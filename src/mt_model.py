"""Sequence-to-sequence model for human motion prediction."""

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter

from typing import *
from enum import IntEnum


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class MTGRU(nn.Module):
    """Multi-task Latent-to-sequence model for human motion prediction"""

    def __init__(self,
                 target_seq_len,
                 rnn_decoder_size,
                 rnn_encoder_size,
                 batch_size,
                 k,
                 n_psi_hidden,
                 n_psi_lowrank,
                 output_dim=64,
                 input_dim=0,
                 dropout=0.0,
                 residual_output=True,
                 init_state_noise=False):
        """Create the model.

        Args:
          target_seq_len: length of the target sequence.
          rnn_decoder_size: number of units in the rnn.
          rnn_encoder_size: number of units in the MT encoder rnn.
          batch_size: the size of the batches used during the forward pass.
          k: the size of the Multi-Task latent space.
          n_psi_hidden: the size of the nonlinear hidden layer for generating parameters psi.
          n_psi_lowrank: the size of the linear subspace in which psi lives (to reduce par count).
          output_dim: instantaneous dimension of output (size of vector emitted at each time t).
          input_dim: size of each input vector at each time t.
          dropout: probability of dropout used for encoding.
          residual_output: passes the inputs directly to output via residual connection. If the output dim > input dim
            as is typically the case when taking the modelled root-feet joints as input, only the first input_dim
            outputs are affected.
        """
        super(MTGRU, self).__init__()

        self.HUMAN_SIZE = output_dim
        self.input_size = input_dim
        self.target_seq_len = target_seq_len
        self.decoder_size = rnn_decoder_size
        self.encoder_size = rnn_encoder_size
        self.batch_size = batch_size
        self.dropout = dropout
        self.residual_output = residual_output
        self.init_state_noise = init_state_noise
        self.k = k

        print("Input size is %d" % self.input_size)
        print('latent_size = {0}'.format(k))
        print('decoder_state_size = {0}'.format(rnn_decoder_size))
        print('encoder_state_size = {0}'.format(rnn_encoder_size))

        # Encoder weights
        self.encoder_cell = nn.GRUCell(self.HUMAN_SIZE, rnn_encoder_size)

        self.to_mu = Parameter(torch.Tensor(rnn_encoder_size, k))
        self.to_lsigma = Parameter(torch.Tensor(rnn_encoder_size, k))
        self.to_lsigma_bias = Parameter(torch.Tensor(k))
        self.init_weights((self.to_mu, self.to_lsigma))
        self.to_lsigma_bias.data = - torch.ones_like(self.to_lsigma_bias.data) * 0.5  # reduce initial std.

        # Psi Decoder weights
        n_psi_pars = sum(self._decoder_par_size())
        self.psi_decoder = torch.nn.Sequential(
            torch.nn.Linear(k, n_psi_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(n_psi_hidden, n_psi_lowrank),
            torch.nn.Linear(n_psi_lowrank, n_psi_pars)
        )

        # GRU Decoder static weights (all except direct recurrent)
        self.gru_Wih = Parameter(torch.Tensor(self.input_size, self.decoder_size * 2))
        self.gru_Whh = Parameter(torch.Tensor(self.decoder_size, self.decoder_size * 2))
        self.gru_bias = Parameter(torch.Tensor(self.decoder_size * 2))
        self.init_weights((self.gru_Wih, self.gru_Whh, self.gru_bias))

        # open GRU Decoder forget gate
        self.gru_bias.data[self.decoder_size:2 * self.decoder_size] = torch.ones_like(
            self.gru_bias.data[self.decoder_size:2 * self.decoder_size])


    def _decoder_par_shape(self):
        Whh = (self.decoder_size, self.decoder_size)
        bh = (self.decoder_size,)
        Wih = (self.input_size, self.decoder_size)
        C = (self.decoder_size, self.HUMAN_SIZE)
        if self.residual_output:
            D = (self.input_size, max(0, self.HUMAN_SIZE - self.input_size))
        else:
            D = (self.input_size, self.HUMAN_SIZE)
        return Whh, bh, Wih, C, D

    def _decoder_par_size(self):
        return [np.prod(x) for x in self._decoder_par_shape()]

    def _decoder_par_slice(self):
        sizes = self._decoder_par_size()
        csizes = np.cumsum(np.hstack((np.zeros((1,), int), sizes)))
        return [slice(csizes[i], csizes[i+1]) for i in range(len(sizes))]

    def _decoder_par_reshape(self, psi):
        shapes = self._decoder_par_shape()
        slices = self._decoder_par_slice()
        return [psi[slice].reshape(shape) for shape, slice in zip(shapes, slices)]

    @staticmethod
    def init_weights(pars):
        for p in pars:
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, inputs, outputs, state=None):
        batch_size = inputs.size(0)  # test time may have a different batch size to train (so don't use self.batch_sz)
        if self.init_state_noise and state is None:
            state = torch.randn(batch_size, self.decoder_size).float().to(inputs.device)
        elif state is None:
            state = torch.zeros(batch_size, self.decoder_size).to(inputs.device)

        # encode outputs into latent z (pseudo-)posterior
        mu, logstd = self.encode(outputs)

        # Sample from (pseudo-)posterior
        eps = torch.randn_like(logstd)
        z = mu + eps * torch.exp(logstd)

        # generate sequence from z
        yhats, states = self.forward_given_z(inputs, z, state)

        return yhats, mu, logstd, states

    def encode(self, outputs):
        # Encode current sequence(s) => latent space
        outputs = torch.transpose(outputs, 0, 1)
        enc_state = self.encoder_cell(outputs[0, :, :])
        for tt in range(1, outputs.shape[0]):
            enc_state = self.encoder_cell(outputs[tt, :, :], enc_state)

        mu, logstd = enc_state @ self.to_mu, enc_state @ self.to_lsigma + self.to_lsigma_bias
        return mu, logstd

    def forward_given_z(self, inputs, z, state):
        batchsize = inputs.shape[0]

        # Decode from sampled z
        psi = self.psi_decoder(z)

        # can't run decoder in batch, since each index has its own parameters
        yhats = []
        states = []
        for bb in range(batchsize):
            Whh, bh, Wih, C, D = self._decoder_par_reshape(psi[bb, :])
            dec = self.partial_GRU(inputs[bb, :, :], Whh, Wih, bh, state[bb, :])
            if self.residual_output:
                yhat_bb = dec @ C + torch.cat((inputs[bb, :, :self.HUMAN_SIZE], inputs[bb, :, :] @ D), 1)
            else:
                yhat_bb = dec @ C + inputs[bb, :, :] @ D
            states.append(dec[-1, :].unsqueeze(0).detach())
            yhats.append(yhat_bb.unsqueeze(0))

        yhats = torch.cat(yhats, dim=0)
        states = torch.cat(states, dim=0)

        return yhats, states

    def partial_GRU(self,
                    x: torch.Tensor,
                    Whh: torch.Tensor,
                    Wih: torch.Tensor,
                    bh: torch.Tensor,
                    init_states: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_sz, d = x.size()
        hidden_seq = []
        h_t = torch.zeros(self.hidden_size).to(x.device) if init_states is None else init_states

        HS = self.decoder_size
        for t in range(seq_sz):
            x_t = x[t, :]

            # batch the static matmuls
            gx, gh = x_t @ self.gru_Wih, h_t @ self.gru_Whh
            _ixr, _ixz = slice(0, HS), slice(HS, 2 * HS)
            r_t = torch.sigmoid(gx[_ixr] + gh[_ixr] + self.gru_bias[_ixr])  # forget
            z_t = torch.sigmoid(gx[_ixz] + gh[_ixz] + self.gru_bias[_ixz])  # convex pass-through

            eta_t = torch.tanh(x_t @ Wih + r_t * (h_t @ Whh) + bh)    # hidden
            h_t = z_t * h_t + (1 - z_t) * eta_t

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        return hidden_seq    # hidden state is simply hidden_seq[-1,:] so no need to return explicitly

    def get_batch(self, data_Y, data_U):
        return self._get_batch(data_Y, data_U, self.batch_size, self.target_seq_len, self.input_size, self.HUMAN_SIZE)

    def get_test_batch(self, data_Y, data_U, batch_size):
        return self._get_test_batch(data_Y, data_U, batch_size, self.target_seq_len, self.input_size, self.HUMAN_SIZE)

    @staticmethod
    def _get_batch(data_Y, data_U, batch_size, target_seq_len, input_size, human_size):

        # Select entries at random
        probs = np.array([y.shape[0] for y in data_Y])
        probs = probs / probs.sum()
        chosen_keys = np.random.choice(len(data_Y), batch_size, p=probs)

        inputs = np.zeros((batch_size, target_seq_len, input_size), dtype=float)
        outputs = np.zeros((batch_size, target_seq_len, human_size), dtype=float)

        for i in range(batch_size):
            the_key = chosen_keys[i]

            # Get the number of frames
            n = data_Y[the_key].shape[0]

            # Sample somewhere in the middle
            idx = np.random.randint(1, n - target_seq_len)

            # append to function outputs
            inputs[i, :, :] = data_U[the_key][idx:idx + target_seq_len, :]
            outputs[i, :, :] = data_Y[the_key][idx:idx + target_seq_len, :]

        return inputs, outputs

    @staticmethod
    def _get_test_batch(data_Y, data_U, batch_size, target_seq_len, input_size, human_size):
        """Get a random batch of data from the specified bucket, prepare for step.

        Args
          data: a list of sequences of size n-by-d to fit the model to.
          actions: a list of the actions we are using
        Returns
          The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
          the constructed batches have the proper format to call step(...) later.
        """

        k_ahead = 6

        if batch_size <= 0:
            batch_size = len(data_Y) - k_ahead + 1
            chosen_keys = range(batch_size)
        else:
            chosen_keys = np.random.choice(len(data_Y) - k_ahead, batch_size,
                                           replace=batch_size > len(data_Y) - k_ahead)

        # How many frames in total do we need?
        target_len = target_seq_len * k_ahead

        inputs = np.zeros((batch_size, target_len, input_size), dtype=float)
        outputs = np.zeros((batch_size, target_len, human_size), dtype=float)

        for i in chosen_keys:
            # Append the data    (batch, tt, dd)
            for k in range(k_ahead):
                inputs[i, (64 * k):(64 * (k+1)), :] = data_U[i + k].T
                outputs[i, (64 * k):(64 * (k+1)), :] = data_Y[i + k].T

        return inputs, outputs


class OpenLoopGRU(nn.Module):
    """Non MT version of the MT model for human motion prediction. Prediction is open loop."""

    def __init__(self,
                 target_seq_len,
                 rnn_decoder_size,
                 batch_size,
                 output_dim=64,
                 input_dim=0,
                 dropout=0.0,
                 residual_output=True,
                 init_state_noise=False):
        """Create the model.

        Args:
          target_seq_len: lenght of the target sequence.
          rnn_decoder_size: number of units in the rnn.
          batch_size: the size of the batches used during the forward pass.
          output_dim: instantaneous dimension of output (size of vector emitted at each time t).
          dropout: probability of dropout used for encoding.
          input_dim: number of input dimensions excl skeleton joints (i.e. trajectory inputs).
          residual_output: passes the inputs directly to output via residual connection. If the output dim > input dim
            as is typically the case when taking the modelled root-feet joints as input, only the first input_dim
            outputs are affected.
        """
        super(OpenLoopGRU, self).__init__()

        self.HUMAN_SIZE = output_dim
        self.input_size = input_dim
        self.target_seq_len = target_seq_len
        self.decoder_size = rnn_decoder_size
        self.batch_size = batch_size
        self.dropout = dropout
        self.residual_output = residual_output
        self.init_state_noise = init_state_noise

        print("Input size is %d" % self.input_size)
        print('decoder_state_size = {0}'.format(rnn_decoder_size))

        self.rnn = nn.GRU(self.input_size, self.decoder_size, batch_first=True)
        self.emission = nn.Linear(self.decoder_size, self.HUMAN_SIZE)

    def forward(self, inputs):
        batch_size = inputs.size(0)  # test time may have a different batch size to train (so don't use self.batch_sz)
        if self.init_state_noise:
            seq, state = self.rnn(inputs, torch.randn(1, batch_size, self.decoder_size).float().to(inputs.device))
        else:
            seq, state = self.rnn(inputs)

        yhats = self.emission(seq)

        if self.residual_output:
            if self.HUMAN_SIZE >= self.input_size:
                yhats = yhats + torch.cat((inputs, torch.zeros(inputs.size(0), inputs.size(1),
                                                               self.HUMAN_SIZE - self.input_size).to(inputs.device)), 2)
            else:
                yhats = yhats + inputs[:, :, :self.HUMAN_SIZE]
        return yhats

    def get_batch(self, data_Y, data_U):
        return MTGRU._get_batch(data_Y, data_U, self.batch_size, self.target_seq_len, self.input_size, self.HUMAN_SIZE)

    def get_test_batch(self, data_Y, data_U, batch_size):
        return MTGRU._get_test_batch(data_Y, data_U, batch_size, self.target_seq_len, self.input_size, self.HUMAN_SIZE)



class DynamicsDict(nn.Module):
    """Multi-task Latent-to-sequence model for human motion prediction"""

    def __init__(self,
                 target_seq_len,
                 rnn_decoder_size,
                 rnn_encoder_size,
                 batch_size,
                 k,
                 n_psi_hidden,
                 n_psi_lowrank,
                 output_dim=64,
                 input_dim=0,
                 dropout=0.0,
                 residual_output=True,
                 init_state_noise=False):
        """Create the model.

        Args:
          target_seq_len: length of the target sequence.
          rnn_decoder_size: number of units in the rnn.
          rnn_encoder_size: number of units in the MT encoder rnn.
          batch_size: the size of the batches used during the forward pass.
          k: the size of the Multi-Task latent space.
          n_psi_hidden: the size of the nonlinear hidden layer for generating parameters psi.
          n_psi_lowrank: the size of the linear subspace in which psi lives (to reduce par count).
          output_dim: instantaneous dimension of output (size of vector emitted at each time t).
          input_dim: size of each input vector at each time t.
          dropout: probability of dropout used for encoding.
          residual_output: passes the inputs directly to output via residual connection. If the output dim > input dim
            as is typically the case when taking the modelled root-feet joints as input, only the first input_dim
            outputs are affected.
        """
        super(DynamicsDict, self).__init__()

        self.HUMAN_SIZE = output_dim
        self.input_size = input_dim
        self.target_seq_len = target_seq_len
        self.decoder_size = rnn_decoder_size
        self.encoder_size = rnn_encoder_size
        self.batch_size = batch_size
        self.dropout = dropout
        self.residual_output = residual_output
        self.init_state_noise = init_state_noise
        self.k = k

        print("Input size is %d" % self.input_size)
        print('latent_size = {0}'.format(k))
        print('decoder_state_size = {0}'.format(rnn_decoder_size))
        print('encoder_state_size = {0}'.format(rnn_encoder_size))

        # Encoder weights
        self.encoder = nn.GRU(self.HUMAN_SIZE, rnn_encoder_size, batch_first=True)

        self.to_mu = Parameter(torch.Tensor(rnn_encoder_size, k))
        self.to_lsigma = Parameter(torch.Tensor(rnn_encoder_size, k))
        self.to_lsigma_bias = Parameter(torch.Tensor(k))
        MTGRU.init_weights((self.to_mu, self.to_lsigma))
        self.to_lsigma_bias.data = - torch.ones_like(self.to_lsigma_bias.data) * 0.5  # reduce initial std.

        # Psi Decoder weights
        n_psi_pars = sum(self._decoder_par_size())
        self.psi_decoder = torch.nn.Sequential(
            torch.nn.Linear(k, n_psi_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(n_psi_hidden, n_psi_lowrank),
            torch.nn.Linear(n_psi_lowrank, n_psi_pars)
        )

        # GRU Decoder static weights (all except direct recurrent)
        self.decoder = nn.GRU(self.input_size, rnn_decoder_size, batch_first=True)


    def _decoder_par_shape(self):
        C = (self.decoder_size, self.HUMAN_SIZE)
        if self.residual_output:
            D = (self.input_size, max(0, self.HUMAN_SIZE - self.input_size))
        else:
            D = (self.input_size, self.HUMAN_SIZE)
        return C, D

    def _decoder_par_size(self):
        return [np.prod(x) for x in self._decoder_par_shape()]

    def _decoder_par_slice(self):
        sizes = self._decoder_par_size()
        csizes = np.cumsum(np.hstack((np.zeros((1,), int), sizes)))
        return [slice(csizes[i], csizes[i+1]) for i in range(len(sizes))]

    def _decoder_par_reshape(self, psi):
        shapes = self._decoder_par_shape()
        slices = self._decoder_par_slice()
        return [psi[slice].reshape(shape) for shape, slice in zip(shapes, slices)]

    def forward(self, inputs, outputs, state=None):
        batch_size = inputs.size(0)  # test time may have a different batch size to train (so don't use self.batch_sz)
        if self.init_state_noise and state is None:
            state = torch.randn(1, batch_size, self.decoder_size).float().to(inputs.device)
        elif state is None:
            state = torch.zeros(1, batch_size, self.decoder_size).to(inputs.device)

        # encode outputs into latent z (pseudo-)posterior
        mu, logstd = self.encode(outputs)

        # Sample from (pseudo-)posterior
        eps = torch.randn_like(logstd)
        z = mu + eps * torch.exp(logstd)

        # generate sequence from z
        yhats, states = self.forward_given_z(inputs, z, state)

        return yhats, mu, logstd, states

    def encode(self, outputs):
        # Encode current sequence(s) => latent space
        # outputs = torch.transpose(outputs, 0, 1)
        seq, enc_state = self.encoder(outputs)

        enc_state = enc_state[0, :, :]  # remove unnecessary first dim (=1)
        mu, logstd = enc_state @ self.to_mu, enc_state @ self.to_lsigma + self.to_lsigma_bias
        return mu, logstd

    def forward_given_z(self, inputs, z, state):
        batchsize = inputs.shape[0]

        # Decode from sampled z
        psi = self.psi_decoder(z)

        dec, state = self.decoder(inputs, state)

        #could run this in batch for efficiency but I'm feeling lazy right now.
        yhats = []
        for bb in range(batchsize):
            C, D = self._decoder_par_reshape(psi[bb, :])
            if self.residual_output:
                yhat_bb = dec[bb,:,:] @ C + torch.cat((inputs[bb, :, :self.HUMAN_SIZE], inputs[bb, :, :] @ D), 1)
            else:
                yhat_bb = dec[bb,:,:] @ C + inputs[bb, :, :] @ D
            yhats.append(yhat_bb.unsqueeze(0))

        yhats = torch.cat(yhats, dim=0)
        return yhats, None

    def get_batch(self, data_Y, data_U):
        return MTGRU._get_batch(data_Y, data_U, self.batch_size, self.target_seq_len, self.input_size, self.HUMAN_SIZE)

    def get_test_batch(self, data_Y, data_U, batch_size):
        return MTGRU._get_test_batch(data_Y, data_U, batch_size, self.target_seq_len, self.input_size, self.HUMAN_SIZE)