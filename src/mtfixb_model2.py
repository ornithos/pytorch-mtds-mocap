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


class MTGRU_NoBias(nn.Module):
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
        super(MTGRU_NoBias, self).__init__()

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

        self.mt_net = MTModule_NoBias(
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
            init_state_noise=init_state_noise,
            is_gru=True if not self.mt_vanilla_rnn else False,
            psi_affine=psi_affine)

        print(self.mt_net.psi_decoder)
        print(self.mt_net.psi_decoder.bias)

        # Layer 1 GRU
        self.layer1_rnn = nn.GRU(self.input_size, hidden_size1, batch_first=True)
        self.layer1_linear = nn.Linear(self.hidden_size1, self.interlayer_dim)
        if residual_output:
            self.skip_connection = nn.Linear(self.hidden_size1, output_dim)

    def get_params_optim_dicts(self, mt_lr, static_lr, z_lr, zls_lr=None):
        if zls_lr is None:
            zls_lr = z_lr

        return [{'params': self.mt_net.psi_decoder.parameters(), 'lr': mt_lr},
                {'params': [self.mt_net.Wih, self.mt_net.b], 'lr': mt_lr},
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


class MTModule_NoBias(nn.Module):
    """Multi-task Latent-to-sequence model for human motion prediction"""

    def __init__(self,
                 target_seq_len,
                 rnn_decoder_size,
                 batch_size,
                 total_num_batches,
                 k,
                 n_psi_hidden,
                 n_psi_lowrank,
                 output_dim=64,
                 input_dim=0,
                 dropout=0.0,
                 residual_output=True,
                 init_state_noise=False,
                 is_gru=True,
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
        super(MTModule_NoBias, self).__init__()

        self.HUMAN_SIZE = output_dim
        self.input_size = input_dim
        self.target_seq_len = target_seq_len
        self.decoder_size = rnn_decoder_size
        self.batch_size = batch_size
        self.dropout = dropout
        self.residual_output = residual_output
        self.init_state_noise = init_state_noise
        self.k = k

        assert not is_gru, "GRU not yet implemented for NoBias module."
        self.is_gru = is_gru

        print("~~~~ MT Module ~~~~")
        print('hidden state size = {0}'.format(rnn_decoder_size))
        print('hierarchical latent size = {0}'.format(k))
        print('layer 2 output size = {0}'.format(self.HUMAN_SIZE))

        # Posterior params
        self.Z_mu = Parameter(torch.randn(total_num_batches, k) * 0.01).float()
        self.Z_logit_s = Parameter(torch.ones(total_num_batches, k) * -1.76).float()      # \approx 0.005 std

        # Const wrt Z params
        self.Wih = Parameter(torch.nn.init.xavier_normal_(torch.zeros(input_dim, rnn_decoder_size)))
        self.b = Parameter(torch.zeros(rnn_decoder_size))

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

    def _decoder_par_shape(self):
        hidden_size = self.decoder_size
        hidden_stack = hidden_size * 3 if self.is_gru else hidden_size
        Whh = (hidden_size, hidden_stack)
        C = (hidden_size, self.HUMAN_SIZE)
        d = self.HUMAN_SIZE
        if self.residual_output:
            D = (self.input_size, max(0, self.HUMAN_SIZE - self.input_size))
        else:
            D = (self.input_size, self.HUMAN_SIZE)
        return Whh, C, D, d

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
            Whh, C, D, d = self._decoder_par_reshape(psi[bb, :])
            bh, Wih = self.b, self.Wih
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
