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
                 hidden_size1,
                 hidden_size2,
                 batch_size,
                 total_num_batches,
                 k,
                 n_psi_hidden,
                 n_psi_lowrank,
                 bottleneck=np.Inf,
                 output_dim=64,
                 input_dim=0,
                 dropout=0.0,
                 residual_output=True,
                 init_state_noise=False,
                 mt_rnn=False,
                 psi_affine=False):
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
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.interlayer_dim = np.min([bottleneck, hidden_size2, hidden_size1])
        self.batch_size = batch_size
        self.dropout = dropout
        self.residual_output = residual_output
        self.init_state_noise = init_state_noise
        self.k = k
        self.mt_vanilla_rnn = mt_rnn

        print("==== MTGRU ====")
        print("Input size is %d" % self.input_size)
        print('hidden state size = {0}'.format(hidden_size1))
        print('layer 1 output size = {0}'.format(self.interlayer_dim))

        self.mt_net = MTModule(
            target_seq_len,
            hidden_size2,
            batch_size,
            total_num_batches,
            k,
            n_psi_hidden,
            n_psi_lowrank,
            psi_affine=psi_affine,
            output_dim=output_dim,
            input_dim=self.interlayer_dim,
            dropout=dropout,
            residual_output=False,
            init_state_noise=init_state_noise,
            is_gru=True if not self.mt_vanilla_rnn else False)

        print(self.mt_net.psi_decoder)
        # Layer 1 GRU
        self.layer1_rnn = nn.GRU(self.input_size, hidden_size1, batch_first=True)
        self.layer1_linear = nn.Linear(self.hidden_size1, self.interlayer_dim)
        if residual_output:
            self.skip_connection = nn.Linear(self.hidden_size1, output_dim)

    def get_params_optim_dicts(self, mt_lr, static_lr, z_lr, zls_lr=None):
        if zls_lr is None:
            zls_lr = z_lr

        return [{'params': self.mt_net.psi_decoder.parameters(), 'lr': mt_lr},
                {'params': self.layer1_rnn.parameters(), 'lr': static_lr},
                {'params': self.layer1_linear.parameters(), 'lr': static_lr},
                {'params': [self.mt_net.Z_mu], 'lr': z_lr},
                {'params': [self.mt_net.Z_logit_s], 'lr': zls_lr}], -1

    def forward(self, inputs, mu, sd, state=None):
        batch_size = inputs.size(0)  # test time may have a different batch size to train (so don't use self.batch_sz)
        if state is not None:
            assert state.size(1) == self.hidden_size1 + self.hidden_size2
            state1, state2 = state[:, :self.hidden_size1].unsqueeze(0), state[:, self.hidden_size1:]
        elif self.init_state_noise and state is None:
            state1, state2 = torch.randn(1, batch_size, self.hidden_size1).float().to(inputs.device), None
        elif state is None:
            state1, state2 = torch.zeros(1, batch_size, self.hidden_size1).float().to(inputs.device), None

        hiddens, state = self.layer1_rnn(inputs, state1)
        intermediate = self.layer1_linear(hiddens)
        yhats, mt_states = self.mt_net(intermediate, mu, sd, state2)
        if self.residual_output:
            yhats = yhats + self.skip_connection(hiddens)

        state = state[0]   # only 1 layer so should be safe.
        return yhats, torch.cat((state, mt_states), dim=1)

    def standardise_aggregate_posterior(self):
        orig_std = self.mt_net.Z_mu.data.std(dim=0)
        self.mt_net.Z_mu.data = self.mt_net.Z_mu.data / orig_std
        self.mt_net.psi_decoder[0].weight.data = self.mt_net.psi_decoder[0].weight.data * orig_std

    def get_batch(self, data_iterator):
        return _get_batch(data_iterator, self.batch_size)


class MTModule(nn.Module):
    """Multi-task Latent-to-sequence model for human motion prediction"""

    def __init__(self,
                 target_seq_len,
                 rnn_decoder_size,
                 batch_size,
                 total_num_batches,
                 k,
                 n_psi_hidden,
                 n_psi_lowrank,
                 psi_affine=False,
                 output_dim=64,
                 input_dim=0,
                 dropout=0.0,
                 residual_output=True,
                 init_state_noise=False,
                 is_gru=True):
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
        super(MTModule, self).__init__()

        self.HUMAN_SIZE = output_dim
        self.input_size = input_dim
        self.target_seq_len = target_seq_len
        self.decoder_size = rnn_decoder_size
        self.batch_size = batch_size
        self.dropout = dropout
        self.residual_output = residual_output
        self.init_state_noise = init_state_noise
        self.k = k
        self.is_gru = is_gru

        print("~~~~ MT Module ~~~~")
        print('hidden state size = {0}'.format(rnn_decoder_size))
        print('hierarchical latent size = {0}'.format(k))
        print('layer 2 output size = {0}'.format(self.HUMAN_SIZE))

        # Posterior params
        self.Z_mu = Parameter(torch.randn(total_num_batches, k) * 0.01).float()
        self.Z_logit_s = Parameter(torch.ones(total_num_batches, k) * -1.76).float()      # \approx 0.005 std

        # Psi Decoder weights
        n_psi_pars = sum(self._decoder_par_size())
        if psi_affine:
            self.psi_decoder = torch.nn.Linear(k, n_psi_pars)
            self.psi_decoder.bias.data = torch.randn(self.psi_decoder.bias.size()) * 0.5e-1
        else:
            self.psi_decoder = torch.nn.Sequential(
                torch.nn.Linear(k, n_psi_hidden),
                torch.nn.Tanh(),
                torch.nn.Linear(n_psi_hidden, n_psi_lowrank),
                torch.nn.Linear(n_psi_lowrank, n_psi_pars)
            )

        # open GRU Decoder forget gate
        if self.is_gru:
            if psi_affine:
                final_layer = self.psi_decoder
            else:
                final_layer = self.psi_decoder[3]

                final_layer.bias.data[self.decoder_size:2 * self.decoder_size] += torch.ones_like(
                    final_layer.bias.data[self.decoder_size:2 * self.decoder_size]) * 1.5

    def _decoder_par_shape(self):
        hidden_size = self.decoder_size
        hidden_stack = hidden_size * 3 if self.is_gru else hidden_size
        Whh = (hidden_size, hidden_stack)
        bh = (hidden_stack,)
        Wih = (self.input_size, hidden_stack)
        C = (hidden_size, self.HUMAN_SIZE)
        d = self.HUMAN_SIZE
        if self.residual_output:
            D = (self.input_size, max(0, self.HUMAN_SIZE - self.input_size))
        else:
            D = (self.input_size, self.HUMAN_SIZE)
        return Whh, bh, Wih, C, D, d

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

    def forward(self, inputs, mu, sd, state=None):
        batch_size = inputs.size(0)  # test time may have a different batch size to train (so don't use self.batch_sz)
        if self.init_state_noise and state is None:
            state = torch.randn(batch_size, self.decoder_size).float().to(inputs.device)
        elif state is None:
            state = torch.zeros(batch_size, self.decoder_size).to(inputs.device)

        # Sample from (pseudo-)posterior
        eps = torch.randn_like(sd)
        z = mu + eps * sd

        # generate sequence from z
        yhats, states = self.forward_given_z(inputs, z, state)

        return yhats, states

    def forward_given_z(self, inputs, z, state):
        batchsize = inputs.shape[0]

        # Decode from sampled z
        psi = self.psi_decoder(z)

        # can't run decoder in batch, since each index has its own parameters
        yhats = []
        states = []
        for bb in range(batchsize):
            Whh, bh, Wih, C, D, d = self._decoder_par_reshape(psi[bb, :])
            if self.is_gru:
                dec = self.mutable_GRU(inputs[bb, :, :], Whh, Wih, bh, state[bb, :])
            else:
                dec = self.mutable_RNN(inputs[bb, :, :], Whh, Wih, bh, state[bb, :])

            if self.residual_output:
                yhat_bb = dec @ C + torch.cat((inputs[bb, :, :self.HUMAN_SIZE], inputs[bb, :, :] @ D), 1) + d
            else:
                yhat_bb = dec @ C + inputs[bb, :, :] @ D + d
            states.append(dec[-1, :].unsqueeze(0).detach())
            yhats.append(yhat_bb.unsqueeze(0))

        yhats1 = torch.cat(yhats, dim=0)
        states = torch.cat(states, dim=0)

        return yhats1, states

    def mutable_GRU(self,
                    x: torch.Tensor,
                    Whh: torch.Tensor,
                    Wih: torch.Tensor,
                    bh: torch.Tensor,
                    init_states: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_sz, d = x.size()
        hidden_seq = []
        h_t = torch.zeros(self.decoder_size).to(x.device) if init_states is None else init_states

        HS = self.decoder_size

        _ixr, _ixz, _ixn = slice(0, HS), slice(HS, 2 * HS), slice(2 * HS, 3 * HS)

        for t in range(seq_sz):
            x_t = x[t, :]

            # batch the static matmuls
            gx, gh = x_t @ Wih, h_t @ Whh
            r_t = torch.sigmoid(gx[_ixr] + gh[_ixr] + bh[_ixr])  # forget
            z_t = torch.sigmoid(gx[_ixz] + gh[_ixz] + bh[_ixz])  # convex pass-through
            eta_t = torch.tanh(gx[_ixn] + r_t * gh[_ixn] + bh[_ixn])    # hidden
            h_t = z_t * h_t + (1 - z_t) * eta_t

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        return hidden_seq    # hidden state is simply hidden_seq[-1,:] so no need to return explicitly

    def mutable_RNN(self,
                    x: torch.Tensor,
                    Whh: torch.Tensor,
                    Wih: torch.Tensor,
                    bh: torch.Tensor,
                    init_states: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_sz, d = x.size()
        hidden_seq = []
        h_t = torch.zeros(self.decoder_size).to(x.device) if init_states is None else init_states

        for t in range(seq_sz):
            x_t = x[t, :]
            h_t = torch.tanh(x_t @ Wih + h_t @ Whh + bh)
            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        return hidden_seq  # hidden state is simply hidden_seq[-1,:] so no need to return explicitly


class MTModule_iid(nn.Module):
    """Multi-task Latent-to-sequence model for human motion prediction"""

    def __init__(self,
                 target_seq_len,
                 hidden_size,
                 batch_size,
                 total_num_batches,
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
        super(MTModule_iid, self).__init__()

        assert init_state_noise is False, "iid MTModule does not support state noise"
        self.HUMAN_SIZE = output_dim
        self.input_size = input_dim
        self.target_seq_len = target_seq_len
        self.decoder_size = hidden_size
        self.batch_size = batch_size
        self.dropout = dropout
        self.residual_output = residual_output
        self.k = k

        print("~~~~ iid MT Module ~~~~")
        print('hidden state size = {0}'.format(hidden_size))
        print('hierarchical latent size = {0}'.format(k))
        print('layer 2 output size = {0}'.format(self.HUMAN_SIZE))

        # Posterior params
        self.Z_mu = Parameter(torch.randn(total_num_batches, k) * 0.01).float()
        self.Z_logit_s = Parameter(torch.ones(total_num_batches, k) * -1.76).float()      # \approx 0.005 std

        # Psi Decoder weights
        n_psi_pars = sum(self._decoder_par_size())
        self.psi_decoder = torch.nn.Sequential(
            torch.nn.Linear(k, n_psi_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(n_psi_hidden, n_psi_lowrank),
            torch.nn.Linear(n_psi_lowrank, n_psi_pars)
        )

    def _decoder_par_shape(self):
        hidden_size = self.decoder_size
        C = (hidden_size, self.HUMAN_SIZE)
        d = self.HUMAN_SIZE
        if self.residual_output:
            D = (self.input_size, max(0, self.HUMAN_SIZE - self.input_size))
        else:
            D = (self.input_size, self.HUMAN_SIZE)
        return C, D, d

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

    def forward(self, hiddens, inputs, mu, sd):

        # Sample from (pseudo-)posterior
        eps = torch.randn_like(sd)
        z = mu + eps * sd

        # generate sequence from z
        yhats = self.forward_given_z(hiddens, inputs, z)

        return yhats

    def forward_given_z(self, hiddens, inputs, z):
        batchsize = inputs.shape[0]

        # Decode from sampled z
        psi = self.psi_decoder(z)

        # can't run decoder in batch, since each index has its own parameters
        yhats = []
        # states = []
        for bb in range(batchsize):
            C, D, d = self._decoder_par_reshape(psi[bb, :])
            if self.residual_output:
                yhat_bb = hiddens[bb,:,:] @ C + torch.cat((inputs[bb, :, :self.HUMAN_SIZE], inputs[bb, :, :] @ D), 1) + d
            else:
                yhat_bb = hiddens[bb,:,:] @ C + inputs[bb, :, :] @ D + d
            # states.append(dec[-1, :].unsqueeze(0).detach())
            yhats.append(yhat_bb.unsqueeze(0))

        yhats1 = torch.cat(yhats, dim=0)
        # states = torch.cat(states, dim=0)

        return yhats1


class MTModule_BiasRNN(nn.Module):
    """Multi-task Latent-to-sequence model for human motion prediction"""

    def __init__(self,
                 target_seq_len,
                 hidden_size,
                 batch_size,
                 total_num_batches,
                 k,
                 n_psi_hidden,
                 n_psi_lowrank,
                 output_dim=64,
                 input_dim=0,
                 dropout=0.0,
                 residual_output=True,
                 init_state_noise=False):


        super(MTModule_BiasRNN, self).__init__()

        self.HUMAN_SIZE = output_dim
        self.input_size = input_dim
        self.target_seq_len = target_seq_len
        self.decoder_size = hidden_size
        self.batch_size = batch_size
        self.dropout = dropout
        self.residual_output = residual_output
        self.init_state_noise = init_state_noise
        self.k = k

        print("~~~~ RNN Bias MT Module ~~~~")
        print('hidden state size = {0}'.format(hidden_size))
        print('hierarchical latent size = {0}'.format(k))
        print('layer 2 output size = {0}'.format(self.HUMAN_SIZE))

        # Posterior params
        self.Z_mu = Parameter(torch.randn(total_num_batches, k) * 0.01).float()
        self.Z_logit_s = Parameter(torch.ones(total_num_batches, k) * -1.76).float()      # \approx 0.005 std

        # Model
        # ==> explicit ints used here because PyTorch's flatten_parameters (modules/rnn.py: 113) errors if np.int64.
        self.rnn = nn.RNN(int(self.input_size + self.k), int(self.decoder_size), batch_first=True)
        self.emission = nn.Linear(int(self.decoder_size), int(self.HUMAN_SIZE))

    def forward(self, inputs, mu, sd, state=None):
        batchsize = inputs.shape[0]
        if self.init_state_noise and state is None:
            state = torch.randn(batchsize, self.decoder_size).float().to(inputs.device)
        elif state is None:
            state = torch.zeros(batchsize, self.decoder_size).to(inputs.device)

        # Sample from (pseudo-)posterior
        eps = torch.randn_like(sd)
        z = mu + eps * sd

        # generate sequence from z
        yhats = self.forward_given_z(state, inputs, z)

        return yhats

    def forward_given_z(self, state, inputs, z):
        batchsize = inputs.shape[0]

        # Decode from sampled z
        z = z.unsqueeze(1)
        z = z.repeat((1, inputs.shape[1], 1))
        inputs = torch.cat((inputs, z), dim=2)

        # Decode
        hiddens, state_enc = self.rnn(inputs, state.unsqueeze(0))
        yhats = self.emission(hiddens)

        return yhats, state_enc[0]   # 1st layer


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

    def get_params_optim_dicts(self, mt_lr, static_lr, z_lr, zls_lr=None):
        return [{'params': self.parameters(), 'lr': mt_lr}], None

    def forward(self, inputs):
        batch_size = inputs.size(0)    # test time may have a different batch size to train (so don't use self.batch_sz)
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
        return yhats, state

    def get_batch(self, data_iterator):
        return _get_batch(data_iterator, self.batch_size)



class DynamicsDict(nn.Module):
    """Multi-task Latent-to-sequence model for human motion prediction"""

    def __init__(self,
                 target_seq_len,
                 rnn_decoder_size,
                 total_num_batches,
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
        self.batch_size = batch_size
        self.dropout = dropout
        self.residual_output = residual_output
        self.init_state_noise = init_state_noise
        self.k = k

        print("Input size is %d" % self.input_size)
        print('latent_size = {0}'.format(k))
        print('decoder_state_size = {0}'.format(rnn_decoder_size))

        self.mt_net = MTModule_iid(
            target_seq_len,
            rnn_decoder_size,
            batch_size,
            total_num_batches,
            k,
            n_psi_hidden,
            n_psi_lowrank,
            output_dim=output_dim,
            input_dim=input_dim,
            dropout=dropout,
            residual_output=residual_output,
            init_state_noise=False)

        # GRU Decoder static weights (all except direct recurrent)
        self.rnn = nn.GRU(self.input_size, rnn_decoder_size, batch_first=True)

    def get_params_optim_dicts(self, mt_lr, static_lr, z_lr, zls_lr=None):
        if zls_lr is None:
            zls_lr = z_lr

        return [{'params': self.mt_net.psi_decoder.parameters(), 'lr': mt_lr},
                {'params': self.rnn.parameters(), 'lr': static_lr},
                {'params': [self.mt_net.Z_mu], 'lr': z_lr},
                {'params': [self.mt_net.Z_logit_s], 'lr': zls_lr}], -1

    def forward(self, inputs, mu, sd, state=None):
        batch_size = inputs.size(0)  # test time may have a different batch size to train (so don't use self.batch_sz)
        if self.init_state_noise and state is None:
            state = torch.randn(1, batch_size, self.decoder_size).float().to(inputs.device)

        dec, state = self.rnn(inputs, state)

        yhats = self.mt_net(dec, inputs, mu, sd)

        return yhats, state

    # def encode(self, outputs):
    #     # Encode current sequence(s) => latent space
    #     # outputs = torch.transpose(outputs, 0, 1)
    #     seq, enc_state = self.encoder(outputs)
    #
    #     enc_state = enc_state[0, :, :]  # remove unnecessary first dim (=1)
    #     mu, logstd = enc_state @ self.to_mu, enc_state @ self.to_lsigma + self.to_lsigma_bias
    #     return mu, logstd

    def get_batch(self, data_iterator):
        return _get_batch(data_iterator, self.batch_size)


class MTGRU_BiasOnly(nn.Module):
    """Multi-task Latent-to-sequence model for human motion prediction"""

    def __init__(self,
                 target_seq_len,
                 hidden_size1,
                 hidden_size2,
                 batch_size,
                 total_num_batches,
                 k,
                 n_psi_hidden,
                 n_psi_lowrank,
                 bottleneck=np.Inf,
                 output_dim=64,
                 input_dim=0,
                 dropout=0.0,
                 residual_output=True,
                 init_state_noise=False,
                 mt_rnn=False):
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
        super(MTGRU_BiasOnly, self).__init__()

        self.HUMAN_SIZE = output_dim
        self.input_size = input_dim
        self.target_seq_len = target_seq_len
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.interlayer_dim = np.min([bottleneck, hidden_size2, hidden_size1])
        self.batch_size = batch_size
        self.dropout = dropout
        self.residual_output = residual_output
        self.init_state_noise = init_state_noise
        self.k = k
        self.mt_vanilla_rnn = mt_rnn

        print("==== MTGRU ====")
        print("Input size is %d" % self.input_size)
        print('hidden state size = {0}'.format(hidden_size1))
        print('layer 1 output size = {0}'.format(self.interlayer_dim))

        self.mt_net = MTModule_BiasRNN(
            target_seq_len,
            hidden_size2,
            batch_size,
            total_num_batches,
            k,
            n_psi_hidden,
            n_psi_lowrank,
            output_dim=output_dim,
            input_dim=self.interlayer_dim,
            dropout=dropout,
            residual_output=False,
            init_state_noise=init_state_noise)

        # Layer 1 GRU
        self.layer1_rnn = nn.GRU(self.input_size, hidden_size1, batch_first=True)
        self.layer1_linear = nn.Linear(self.hidden_size1, self.interlayer_dim)
        if residual_output:
            self.skip_connection = nn.Linear(self.hidden_size1, output_dim)

    def get_params_optim_dicts(self, mt_lr, static_lr, z_lr, zls_lr=None):
        if zls_lr is None:
            zls_lr = z_lr

        return [{'params': [self.mt_net.rnn.weight_ih_l0] , 'lr': mt_lr},
                {'params': [self.mt_net.rnn.weight_hh_l0, self.mt_net.rnn.bias_ih_l0, self.mt_net.rnn.bias_hh_l0] , 'lr': mt_lr},
                {'params': self.mt_net.emission.parameters(), 'lr': mt_lr},
                {'params': self.layer1_rnn.parameters(), 'lr': static_lr},
                {'params': self.layer1_linear.parameters(), 'lr': static_lr},
                {'params': [self.mt_net.Z_mu], 'lr': z_lr},
                {'params': [self.mt_net.Z_logit_s], 'lr': zls_lr}], -1

    def forward(self, inputs, mu, sd, state=None):
        batch_size = inputs.size(0)  # test time may have a different batch size to train (so don't use self.batch_sz)
        if state is not None:
            assert state.size(1) == self.hidden_size1 + self.hidden_size2
            state1, state2 = state[:, :self.hidden_size1].unsqueeze(0), state[:, self.hidden_size1:]
        elif self.init_state_noise and state is None:
            state1, state2 = torch.randn(1, batch_size, self.hidden_size1).float().to(inputs.device), None
        elif state is None:
            state1, state2 = torch.zeros(1, batch_size, self.hidden_size1).float().to(inputs.device), None

        hiddens, state = self.layer1_rnn(inputs, state1)
        intermediate = self.layer1_linear(hiddens)
        yhats, mt_states = self.mt_net(intermediate, mu, sd, state2)
        if self.residual_output:
            yhats = yhats + self.skip_connection(hiddens)

        state = state[0]   # only 1 layer so should be safe.
        return yhats, torch.cat((state, mt_states), dim=1)

    def standardise_aggregate_posterior(self):
        orig_std = self.mt_net.Z_mu.data.std(dim=0)
        self.mt_net.Z_mu.data = self.mt_net.Z_mu.data / orig_std
        self.mt_net.rnn.weight_ih_l0.data[:, self.interlayer_dim:] = \
            self.mt_net.rnn.weight_ih_l0.data[:, self.interlayer_dim:] * orig_std

    def get_batch(self, data_iterator):
        return _get_batch(data_iterator, self.batch_size)


class DataIterator:
    def __init__(self, dataY, dataU, chunk_size, min_size, start=0, checks=True, overlap2=False):
        if checks:
            assert len(dataY) == len(dataU), "Y and U lists have a different number of elements"
            for i in range(len(dataY)):
                assert dataY[i].shape[0] == dataU[i].shape[0], \
                    "item {:d} in data has inconsistent number of rows between Y and U".format(i)

        self.i = start
        self.start = start
        self.element = 0
        self.batch_ix = 0
        self.chunk_size = chunk_size
        self.min_size = min_size
        self.orig_dataY = dataY
        self.orig_dataU = dataU
        self.dataY = dataY
        self.dataU = dataU
        self.y_dim = self.dataY[0].shape[1]
        self.u_dim = self.dataU[0].shape[1]
        self.increment_start = np.array([start, (start + chunk_size // 2) % chunk_size] if overlap2 else [start])
        self.c_overlap = 0
        self.length = self._length()
        self.batch_ids = list(range(self.length))

    def __iter__(self):
        return self

    def __next__(self):
        if self.element < len(self.dataY):
            while self.i + self.min_size > self.dataY[self.element].shape[0]:
                self.element += 1
                self.i = self.start
                if self.element >= len(self.dataY):
                    raise StopIteration()

            i, element = self.i, self.element
            is_new_state = i == self.start  # not in while, since 1st iterate in general won't use this loop.

            chunk_Y = self.dataY[element]
            chunk_U = self.dataU[element]
            eix = min(i + self.chunk_size, chunk_Y.shape[0])
            out = (chunk_Y[i:eix, :], chunk_U[i:eix, :], self.batch_ids[self.batch_ix], is_new_state)
            self.i = eix
            self.batch_ix += 1
            return out
        else:
            raise StopIteration()

    def shuffle(self, increment_start=False):
        self.reset_data(increment_start=increment_start)
        self.reset_indices()
        all_data = list(self)
        self.reset_indices()
        self.batch_ids = np.random.permutation(len(all_data))
        self.dataY = [all_data[i][0] for i in self.batch_ids]
        self.dataU = [all_data[i][1] for i in self.batch_ids]
        self.length = self._length()

        # The data are now partitioned st each partition begins at `start`, so init value should always be zero.
        self.start = 0
        self.i = 0
        assert len(all_data) == self.length, "Error in iterator design. Please contact the library creator."

    def reset_data(self, increment_start=False):
        if increment_start:
            self.c_overlap = (self.c_overlap + 1) % len(self.increment_start)
            self.start = self.increment_start[self.c_overlap]
        self.dataY = self.orig_dataY
        self.dataU = self.orig_dataU
        self.length = self._length()
        self.batch_ids = list(range(self.length))

    def reset_indices(self):
        self.i = self.start
        self.element = 0
        self.batch_ix = 0

    def _length(self, start=None):
        if start is None: start = self.increment_start[self.c_overlap]

        _len = 0
        for y in self.orig_dataY:
            d, r = divmod(y.shape[0] - start, self.chunk_size)
            _len += d + (r >= self.min_size)
        return _len

    def total_length(self):
        _len = 0
        for start in self.increment_start:
            _len += self._length(start=start)
        return _len


def _get_batch(data_iterator, batch_size):

    if batch_size <= 0: batch_size = data_iterator.length

    inputs = np.zeros((batch_size, data_iterator.chunk_size, data_iterator.u_dim), dtype=float)
    outputs = np.zeros((batch_size, data_iterator.chunk_size, data_iterator.y_dim), dtype=float)
    ixs = []
    for i in range(batch_size):
        try:
            y, u, ix, is_new_state = next(data_iterator)
        except StopIteration:
            data_iterator.shuffle(increment_start=True)
            y, u, ix, is_new_state = next(data_iterator)

        inputs[i, :, :] = u
        outputs[i, :, :] = y
        ixs.append(ix)

    return inputs, outputs, ixs
