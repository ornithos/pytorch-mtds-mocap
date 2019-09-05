"""Sequence-to-sequence model for human motion prediction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from tensorflow.python.ops import array_ops
# from tensorflow.python.ops import variable_scope as vs

import random

import numpy as np
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import torch
from torch import nn
import torch.nn.functional as F


class Seq2SeqModelOpen(nn.Module):
    """Sequence-to-sequence model for human motion prediction"""

    def __init__(self,
                 architecture,
                 source_seq_len,
                 target_seq_len,
                 rnn_size,  # hidden recurrent layer size
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 loss_to_use,
                 number_of_actions,
                 one_hot=True,
                 residual_velocities=False,
                 output_dim=67,
                 dropout=0.0,
                 dtype=torch.float32,
                 num_traj=0):
        """Create the model.

        Args:
          architecture: [basic, tied] whether to tie the decoder and decoder.
          source_seq_len: lenght of the input sequence.
          target_seq_len: lenght of the target sequence.
          rnn_size: number of units in the rnn.
          num_layers: number of rnns to stack.
          max_gradient_norm: gradients will be clipped to maximally this norm.
          batch_size: the size of the batches used during training;
            the model construction is independent of batch_size, so it can be
            changed after initialization if this is convenient, e.g., for decoding.
          learning_rate: learning rate to start with.
          learning_rate_decay_factor: decay learning rate by this much when needed.
          loss_to_use: [supervised, sampling_based]. Whether to use ground truth in
            each timestep to compute the loss after decoding, or to feed back the
            prediction from the previous time-step.
          number_of_actions: number of classes we have.
          one_hot: whether to use one_hot encoding during train/test (sup models).
          residual_velocities: whether to use a residual connection that models velocities.
          dtype: the data type to use to store internal variables.
        """
        super(Seq2SeqModelOpen, self).__init__()

        self.HUMAN_SIZE = output_dim
        self.input_size = self.HUMAN_SIZE + num_traj + (number_of_actions if one_hot else 0)

        print("Training OPEN LOOP model.")
        print("One hot is ", one_hot)
        print("Input size is %d" % self.input_size)

        # Summary writers for train and test runs

        self.source_seq_len = source_seq_len
        self.target_seq_len = target_seq_len
        self.rnn_size = rnn_size
        self.batch_size = batch_size
        self.dropout = dropout
        self.num_layers = num_layers

        print('rnn_size = {0}'.format(rnn_size))
        self.rnn = nn.GRU(self.input_size, self.rnn_size)
        if num_layers > 1:
            self.rnn2 = nn.GRU(self.rnn_size, self.rnn_size)
        self.fc1 = nn.Linear(self.rnn_size * num_layers, self.HUMAN_SIZE)

    def copy_rnn_weights_to_cells(self, cells):
        cells[0].weight_ih = self.rnn.weight_ih_l0
        cells[0].weight_hh = self.rnn.weight_hh_l0
        cells[0].bias_ih = self.rnn.bias_ih_l0
        cells[0].bias_hh = self.rnn.bias_hh_l0
        if self.num_layers > 1:
            cells[1].weight_ih = self.rnn2.weight_ih_l0
            cells[1].weight_hh = self.rnn2.weight_hh_l0
            cells[1].bias_ih = self.rnn2.bias_ih_l0
            cells[1].bias_hh = self.rnn2.bias_hh_l0

    def forward(self, encoder_inputs, decoder_inputs, use_cuda):

        batchsize = encoder_inputs.shape[0]
        inputs_enc, inputs_dec = torch.transpose(encoder_inputs, 0, 1), torch.transpose(decoder_inputs, 0, 1)

        if self.training:
            # able to take adv of PyTorch's fast GRU implementation. Use two sep GRU modules as want skip conns.
            state1, state_enc = self.rnn(inputs_enc)
            state_out, _ = self.rnn(inputs_dec, state_enc)
            if self.num_layers > 1:
                _, state_enc2 = self.rnn2(state1)
                state_out2, _ = self.rnn2(state_out, state_enc2)
                state_out = torch.cat((state_out, state_out2), dim=2)

            outputs = inputs_dec[:, :, 0:self.HUMAN_SIZE] + self.fc1(
                F.dropout(state_out, self.dropout, training=self.training))

        else:
            # Need to feed predictions back into inputs => (afaik) need to fall back on slower GRUCells.
            # create GRUCells on-the-fly to avoid larger memory footprint
            cells = [torch.nn.GRUCell(self.input_size, self.rnn_size)]
            if self.num_layers >= 1:
                cells.append(torch.nn.GRUCell(self.rnn_size, self.rnn_size))
            self.copy_rnn_weights_to_cells(cells)

            outputs = []

            # Encode seed sequence
            state1, state_enc = self.rnn(inputs_enc)
            if self.num_layers > 1:
                _, state_enc2 = self.rnn2(state1)
                state_enc = torch.cat((state_enc, state_enc2), dim=0)

            # First step t=1
            u = inputs_dec[0]
            state = cells[0](u, state_enc[0])
            if self.num_layers > 1:
                state2 = cells[1](state, state_enc[1])
                state_out = torch.cat((state, state2), dim=1)
            else:
                state_out = state
            output = u[:, 0:self.HUMAN_SIZE] + self.fc1(
                F.dropout(state_out, self.dropout, training=self.training))
            outputs.append(output.view([1, batchsize, self.HUMAN_SIZE]))

            # t > 1
            for i, u in enumerate(inputs_dec[1:]):
                inp = torch.cat((output.detach(), u[:, self.HUMAN_SIZE:]), 1)

                state = cells[0](inp, state)
                if self.num_layers > 1:
                    state2 = cells[1](state, state2)
                    state_out = torch.cat((state, state2), dim=1)
                else:
                    state_out = state

                output = inp[:, 0:self.HUMAN_SIZE] + self.fc1(
                    F.dropout(state_out, self.dropout, training=self.training))

                outputs.append(output.view([1, batchsize, self.HUMAN_SIZE]))

            outputs = torch.cat(outputs, 0)

        outputs = torch.transpose(outputs, 0, 1)
        return outputs


    def get_batch(self, data_Y, data_U, actions, stratify=False):
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
        if not stratify:
            chosen_keys = np.random.choice(len(data_Y), self.batch_size, p=probs)
            bsz = self.batch_size
        else:
            bsz = len(data_Y)
            chosen_keys = list(range(bsz))

        # How many frames in total do we need?
        total_frames = self.source_seq_len + self.target_seq_len
        traj_size = data_U[1].shape[1]

        encoder_inputs = np.zeros((self.batch_size, self.source_seq_len - 1, self.input_size), dtype=float)
        decoder_inputs = np.zeros((self.batch_size, self.target_seq_len, self.input_size), dtype=float)
        decoder_outputs = np.zeros((self.batch_size, self.target_seq_len, self.HUMAN_SIZE), dtype=float)

        for i in xrange(bsz):
            the_key = chosen_keys[i]

            # Get the number of frames
            n = data_Y[the_key].shape[0]
            assert n > total_frames, "n of file {:s} too small.".format(the_key)

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
