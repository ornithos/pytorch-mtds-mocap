"""Sequence-to-sequence model for human motion prediction."""

import numpy as np
import os
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class MTGRU(nn.Module):
    """Multi-task Latent-to-sequence model for human motion prediction"""

    def __init__(self,
                 source_seq_len,
                 target_seq_len,
                 rnn_decoder_size,
                 rnn_encoder_size,
                 batch_size,
                 k,
                 n_psi_hidden,
                 n_psi_lowrank,
                 output_dim=64,
                 dropout=0.0,
                 input_dim=0,
                 residual_output=True):
        """Create the model.

        Args:
          source_seq_len: lenght of the input sequence.
          target_seq_len: lenght of the target sequence.
          rnn_decoder_size: number of units in the rnn.
          rnn_encoder_size: number of units in the MT encoder rnn.
          batch_size: the size of the batches used during the forward pass.
          k: the size of the Multi-Task latent space.
          output_dim: instantaneous dimension of output (size of vector emitted at each time t).
          dropout: probability of dropout used for encoding.
          input_dim: number of input dimensions excl skeleton joints (i.e. trajectory inputs).
          residual_output: passes the inputs directly to output via residual connection. If the output dim > input dim
            as is typically the case when taking the modelled root-feet joints as input, only the first input_dim
            outputs are affected.
        """
        super(MTGRU, self).__init__()

        self.HUMAN_SIZE = output_dim
        self.input_size = input_dim
        self.source_seq_len = source_seq_len
        self.target_seq_len = target_seq_len
        self.decoder_size = rnn_decoder_size
        self.encoder_size = rnn_encoder_size
        self.batch_size = batch_size
        self.dropout = dropout
        self.residual_output = residual_output
        self.k = k

        print("Input size is %d" % self.input_size)
        print('latent_size = {0}'.format(k))
        print('decoder_state_size = {0}'.format(rnn_decoder_size))
        print('encoder_state_size = {0}'.format(rnn_encoder_size))

        # Encoder weights
        self.encoder_cell = nn.GRUCell(self.HUMAN_SIZE, rnn_encoder_size)
        self.to_mu = Parameter(torch.Tensor(k, rnn_encoder_size))
        self.to_lsigma = Parameter(torch.Tensor(k, rnn_encoder_size))
        self.init_weights((self.to_mu, self.to_lsigma))

        # Psi Decoder weights
        n_psi_pars = sum(self._decoder_par_size())
        self.psi_decoder = torch.nn.Sequential(
            torch.nn.Linear(k, n_psi_hidden),
            torch.nn.tanh(),
            torch.nn.Linear(n_psi_hidden, n_psi_lowrank),
            torch.nn.Linear(n_psi_lowrank, n_psi_pars)
        )

        # GRU Decoder static weights (all except direct recurrent)
        self.gru_Wih = Parameter(torch.Tensor(self.decoder_size * 2, self.input_size))
        self.gru_Whh = Parameter(torch.Tensor(self.decoder_size, self.decoder_size * 2))
        self.gru_bias = Parameter(torch.Tensor(self.decoder_size * 2))
        self.init_weights((self.gru_Wih, self.gru_Whh, self.gru_bias))

        # open GRU Decoder forget gate
        self.gru_bias.data[self.hidden_size:2 * self.hidden_size] = torch.ones_like(
            self.gru_bias.data[self.hidden_size:2 * self.hidden_size])


    def _decoder_par_shape(self):
        Whh = (self.decoder_size, self.decoder_size)
        bh = (self.decoder_size,)
        Whi = (self.decoder_size, self.input_size)
        C = (self.decoder_size, self.HUMAN_SIZE)
        D = (self.input_size, (self.HUMAN_SIZE - self.input_size))
        return Whh, bh, Whi, C, D

    def _decoder_par_size(self):
        return [np.prod(x) for x in self._decoder_par_shape()]

    def _decoder_par_slice(self):
        sizes = self._decoder_par_size()
        csizes = np.cumsum(np.vstack(([0], sizes)))
        return [slice(csizes[i], csizes[i+1]) for i in range(len(sizes))]

    def _decoder_par_reshape(self, psi):
        shapes = self._decoder_par_shape()
        slices = self._decoder_par_slice()
        return [psi[slice].reshape(shape) for shape, slice in zip(shapes, slices)]

    @staticmethod
    def init_weights(self, pars):
        for p in pars:
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, inputs, state=None):
        batchsize = inputs.shape[0]
        # inputs = torch.transpose(inputs, 0, 1) # => leftover from Martinez code (delete if still here and unused!)
        state = torch.zeros(batchsize, self.decoder_size).to(inputs.device) if state is None else state

        # Encode current sequence(s) => latent space
        enc_state = self.encoder_cell(inputs)
        mu, logstd = self.to_mu @ enc_state, self.to_lsigma @ enc_state

        # Sample from (pseudo-)posterior
        eps = torch.randn_like(logstd)
        z = mu + eps * torch.exp(logstd)

        # Decode from sampled z
        psi = self.psi_decoder(z)

        # can't run decoder in batch, since each index has its own parameters
        yhats = []
        states = []
        for bb in range(batchsize):
            Whh, bh, Wih, C, D = self._decoder_par_reshape(psi)
            dec = self.partial_GRU(inputs[bb, :, :], Whh, Wih, bh, state[bb, :])
            yhat_bb = dec @ C + torch.cat((torch.zeros(inputs.size(1), self.input_size), inputs[bb, :, :] @ D), 1)
            states.append(dec[-1, :].unsqueeze(0).detach())
            yhats.append(yhat_bb.unsqueeze(0))

        yhats = torch.cat(yhats, dim=0)
        states = torch.cat(states, dim=0)
        # residual connection?
        if self.residual_output:
            yhats = yhats + torch.cat((inputs, torch.zeros(batchsize, inputs.size(0),
                                                           self.HUMAN_SIZE - self.input_size)), 2)

        return yhats, mu, logstd, states

    def partial_GRU(self,
                    x: torch.Tensor,
                    Whh: torch.Tensor,
                    Wih: torch.Tensor,
                    bh: torch.Tensor,
                    init_states: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_sz, d = x.size()
        hidden_seq = []
        h_t = torch.zeros(self.hidden_size).to(x.device) if init_states is None else init_states

        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[t, :]

            # batch the static matmuls
            gx, gh = x_t @ self.gru_Wih, h_t @ self.gru_Whh
            _ixz, _ixr = slice(0, HS), slice(HS, 2 * HS)
            z_t = torch.sigmoid(gx[:, _ixz] + gh[:, _ixz] + self.bias[_ixz])  # convex pass-through
            r_t = torch.sigmoid(gx[:, _ixr] + gh[:, _ixr] + self.bias[_ixr])  # forget

            eta_t = torch.tanh(x @ Wih + r_t * torch.tanh(h_t @ Whh + bh))    # hidden
            h_t = z_t * h_t + (1 - z_t) * eta_t

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        return hidden_seq    # hidden state is simply hidden_seq[-1,:] so no need to return explicitly

    def get_batch(self, data_Y, data_U, actions):
        """Get a random batch of data from the specified bucket, prepare for step.

        Args
          data: a list of sequences of size n-by-d to fit the model to.
          actions: a list of the actions we are using
        Returns
          The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
          the constructed batches have the proper format to call step(...) later.
        """

        # Select entries at random
        probs = np.array([y.shape[0] for y in data_Y])
        probs = probs / probs.sum()
        chosen_keys = np.random.choice(len(data_Y), self.batch_size, p=probs)

        # How many frames in total do we need?
        total_frames = self.source_seq_len + self.target_seq_len
        traj_size = data_U[1].shape[1]

        encoder_inputs = np.zeros((self.batch_size, self.source_seq_len - 1, self.input_size), dtype=float)
        decoder_inputs = np.zeros((self.batch_size, self.target_seq_len, self.input_size), dtype=float)
        decoder_outputs = np.zeros((self.batch_size, self.target_seq_len, self.HUMAN_SIZE), dtype=float)

        for i in range(self.batch_size):
            the_key = chosen_keys[i]

            # Get the number of frames
            n = data_Y[the_key].shape[0]

            # Sample somewherein the middle
            idx = np.random.randint(1, n - total_frames)

            # Select the data around the sampled points
            data_Y_sel = data_Y[the_key][idx:idx + total_frames, :]
            data_U_sel = data_U[the_key][idx:idx + total_frames, :]

            # Add the data
            encoder_inputs[i, :, 0:self.HUMAN_SIZE] = data_Y_sel[0:self.source_seq_len - 1, :]
            encoder_inputs[i, :, self.HUMAN_SIZE:] = data_U_sel[0:self.source_seq_len - 1, :]  # <= done
            decoder_inputs[i, :, 0:self.HUMAN_SIZE] = data_Y_sel[
                                                      self.source_seq_len - 1:self.source_seq_len + self.target_seq_len - 1,
                                                      :]
            decoder_inputs[i, :, self.HUMAN_SIZE:] = data_U_sel[
                                                     self.source_seq_len - 1:self.source_seq_len + self.target_seq_len - 1,
                                                     :]
            decoder_outputs[i, :, 0:self.HUMAN_SIZE] = data_Y_sel[
                                                       self.source_seq_len:self.source_seq_len + self.target_seq_len,
                                                       :]  # <= done

        return encoder_inputs, decoder_inputs, decoder_outputs

    def get_test_batch(self, data_Y, data_U, batch_size):
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
            batch_size = len(data_Y) - k_ahead
            chosen_keys = range(batch_size)
        else:
            chosen_keys = np.random.choice(len(data_Y) - k_ahead, batch_size,
                                           replace=batch_size > len(data_Y) - k_ahead)

        # How many frames in total do we need?
        source_len = 64
        target_len = 64 * k_ahead
        total_frames = self.source_seq_len * (k_ahead + 1)

        encoder_inputs = np.zeros((batch_size, source_len - 1, self.input_size), dtype=float)
        decoder_inputs = np.zeros((batch_size, target_len, self.input_size), dtype=float)
        decoder_outputs = np.zeros((batch_size, target_len, self.HUMAN_SIZE), dtype=float)

        for i in chosen_keys:
            # Add the data    (batch, tt, dd)
            encoder_inputs[i, :, 0:self.HUMAN_SIZE] = data_Y[i].T[0:source_len - 1, :]
            encoder_inputs[i, :, self.HUMAN_SIZE:] = data_U[i].T[0:source_len - 1, :]

            decoder_inputs[i, 0, 0:self.HUMAN_SIZE] = data_Y[i].T[source_len - 1, :]
            decoder_inputs[i, 0, self.HUMAN_SIZE:] = data_U[i].T[source_len - 1, :]
            for k in range(k_ahead - 1):
                decoder_inputs[i, 64 * k + 1:64 * (k + 1) + 1, 0:self.HUMAN_SIZE] = data_Y[i + 1 + k].T
                decoder_inputs[i, 64 * k + 1:64 * (k + 1) + 1, self.HUMAN_SIZE:] = data_U[i + 1 + k].T
                decoder_outputs[i, 64 * k:64 * (k + 1), :] = data_Y[i + 1 + k].T
            decoder_inputs[i, 64 * (k_ahead - 1) + 1:64 * k_ahead + 1, 0:self.HUMAN_SIZE] = data_Y[i + k_ahead].T[0:63,
                                                                                            :]
            decoder_inputs[i, 64 * (k_ahead - 1) + 1:64 * k_ahead + 1, self.HUMAN_SIZE:] = data_U[i + k_ahead].T[0:63,
                                                                                           :]
            decoder_outputs[i, 64 * (k_ahead - 1):64 * k_ahead, :] = data_Y[i + k_ahead].T

        return encoder_inputs, decoder_inputs, decoder_outputs

    def find_indices_srnn(self, data, action):
        """
        Find the same action indices as in SRNN.
        See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
        """

        # Used a fixed dummy seed, following
        # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
        SEED = 1234567890
        rng = np.random.RandomState(SEED)

        subject = 5
        subaction1 = 1
        subaction2 = 2

        T1 = data[(subject, action, subaction1, 'even')].shape[0]
        T2 = data[(subject, action, subaction2, 'even')].shape[0]
        prefix, suffix = 50, 100

        idx = []
        idx.append(rng.randint(16, T1 - prefix - suffix))
        idx.append(rng.randint(16, T2 - prefix - suffix))
        idx.append(rng.randint(16, T1 - prefix - suffix))
        idx.append(rng.randint(16, T2 - prefix - suffix))
        idx.append(rng.randint(16, T1 - prefix - suffix))
        idx.append(rng.randint(16, T2 - prefix - suffix))
        idx.append(rng.randint(16, T1 - prefix - suffix))
        idx.append(rng.randint(16, T2 - prefix - suffix))
        return idx

    def get_batch_srnn(self, data, action):
        """
        Get a random batch of data from the specified bucket, prepare for step.

        Args
          data: dictionary with k:v, k=((subject, action, subsequence, 'even')),
            v=nxd matrix with a sequence of poses
          action: the action to load data from
        Returns
          The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
          the constructed batches have the proper format to call step(...) later.
        """

        actions = ["directions", "discussion", "eating", "greeting", "phoning",
                   "posing", "purchases", "sitting", "sittingdown", "smoking",
                   "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]

        if not action in actions:
            raise ValueError("Unrecognized action {0}".format(action))

        frames = {}
        frames[action] = self.find_indices_srnn(data, action)

        batch_size = 8  # we always evaluate 8 seeds
        subject = 5  # we always evaluate on subject 5
        source_seq_len = self.source_seq_len
        target_seq_len = self.target_seq_len

        seeds = [(action, (i % 2) + 1, frames[action][i]) for i in range(batch_size)]

        encoder_inputs = np.zeros((batch_size, source_seq_len - 1, self.input_size), dtype=float)
        decoder_inputs = np.zeros((batch_size, target_seq_len, self.input_size), dtype=float)
        decoder_outputs = np.zeros((batch_size, target_seq_len, self.input_size), dtype=float)

        # Compute the number of frames needed
        total_frames = source_seq_len + target_seq_len

        # Reproducing SRNN's sequence subsequence selection as done in
        # https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L343
        for i in range(batch_size):
            _, subsequence, idx = seeds[i]
            idx = idx + 50

            data_sel = data[(subject, action, subsequence, 'even')]

            data_sel = data_sel[(idx - source_seq_len):(idx + target_seq_len), :]

            encoder_inputs[i, :, :] = data_sel[0:source_seq_len - 1, :]
            decoder_inputs[i, :, :] = data_sel[source_seq_len - 1:(source_seq_len + target_seq_len - 1), :]
            decoder_outputs[i, :, :] = data_sel[source_seq_len:, :]

        return encoder_inputs, decoder_inputs, decoder_outputs