"""Simple code for training an RNN for motion prediction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import h5py

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

import seq2seq_model, seq2seq_model_open
import torch
import torch.optim as optim
from torch.autograd import Variable
import parseopts4bmark


def create_model(args, actions, sampling=False):
    """Create translation model and initialize or load parameters in session."""

    if not args.open_loop:
        model = seq2seq_model.Seq2SeqModel(
            args.architecture,
            args.seq_length_in,
            args.seq_length_out,
            args.size,  # hidden layer size
            args.num_layers,
            args.max_gradient_norm,
            args.batch_size,
            args.learning_rate,
            args.learning_rate_decay_factor,
            args.loss_to_use if not sampling else "sampling_based",
            len(actions),
            not args.omit_one_hot,
            args.residual_velocities,
            output_dim=67,
            dtype=torch.float32,
            num_traj=35)
    else:
        model = seq2seq_model_open.Seq2SeqModelOpen(
            args.architecture,
            args.seq_length_in,
            args.seq_length_out,
            args.size,  # hidden layer size
            args.num_layers,
            args.max_gradient_norm,
            args.batch_size,
            args.learning_rate,
            args.learning_rate_decay_factor,
            args.loss_to_use if not sampling else "sampling_based",
            len(actions),
            not args.omit_one_hot,
            args.residual_velocities,
            output_dim=67,
            dtype=torch.float32,
            num_traj=35)

    if len(args.load) <= 0:
        return model

    print("Loading model")
    model = torch.load(args.load, map_location='cpu') if args.use_cpu else torch.load(args.load)
    return model


def train(args):
    """Train a seq2seq model on human motion"""

    actions = define_actions(args.action)

    train_set_Y, train_set_U, test_set_Y, test_set_U = read_all_data(args)

    # Limit TF to take a fraction of the GPU memory

    if True:
        model = create_model(args, actions, args.sample)
        if not args.use_cpu:
            model = model.cuda()

        # === This is the training loop ===

        current_step = 0
        if len(args.load) > 0:
            Exception("Training from load file not supported in this file.")

        previous_losses, val_losses, save_ixs = [], [], []

        step_time, loss = 0, 0
        if args.optimiser.upper() == 'SGD':
            optimiser = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimiser.upper() == 'ADAM':
            optimiser = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
                                   weight_decay=args.weight_decay)
        else:
            Exception('Unknown optimiser specified `{:s}`, please choose `SGD` or `Adam`'.format(args.optimiser))

        for _ in range(args.iterations):
            optimiser.zero_grad()
            model.train()

            start_time = time.time()

            # Actual training

            # === Training step ===
            encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch(train_set_Y, train_set_U,
                                                                              not args.omit_one_hot)
            encoder_inputs = torch.from_numpy(encoder_inputs).float()
            decoder_inputs = torch.from_numpy(decoder_inputs).float()
            decoder_outputs = torch.from_numpy(decoder_outputs).float()
            if not args.use_cpu:
                encoder_inputs = encoder_inputs.cuda()
                decoder_inputs = decoder_inputs.cuda()
                decoder_outputs = decoder_outputs.cuda()
            encoder_inputs = Variable(encoder_inputs)
            decoder_inputs = Variable(decoder_inputs)
            decoder_outputs = Variable(decoder_outputs)

            preds = model(encoder_inputs, decoder_inputs, not args.use_cpu)

            step_loss = (preds - decoder_outputs) ** 2
            step_loss = step_loss.mean()

            # Actual backpropagation
            step_loss.backward()
            optimiser.step()

            step_loss = step_loss.cpu().data.numpy()

            if current_step % 10 == 0:
                print("step {0:04d}; step_loss: {1:.4f}".format(current_step, step_loss))
            if current_step % 50 == 0:
                sys.stdout.flush()

            step_time += (time.time() - start_time) / args.test_every
            loss += step_loss / args.test_every
            current_step += 1
            # === step decay ===
            if current_step % args.learning_rate_step == 0:
                args.learning_rate = args.learning_rate * args.learning_rate_decay_factor
                for param_group in optimiser.param_groups:
                    param_group['lr'] = args.learning_rate
                print("Decay learning rate. New value at " + str(args.learning_rate))

            # cuda.empty_cache()

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % args.test_every == 0:
                model.eval()

                # === Validation with random data from test set ===
                encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch(test_set_Y, test_set_U,
                                                                                  not args.omit_one_hot, stratify=True)
                encoder_inputs = torch.from_numpy(encoder_inputs).float()
                decoder_inputs = torch.from_numpy(decoder_inputs).float()
                decoder_outputs = torch.from_numpy(decoder_outputs).float()
                if not args.use_cpu:
                    encoder_inputs = encoder_inputs.cuda()
                    decoder_inputs = decoder_inputs.cuda()
                    decoder_outputs = decoder_outputs.cuda()
                encoder_inputs = Variable(encoder_inputs)
                decoder_inputs = Variable(decoder_inputs)
                decoder_outputs = Variable(decoder_outputs)

                preds = model(encoder_inputs, decoder_inputs, not args.use_cpu)

                step_loss = (preds - decoder_outputs) ** 2

                val_loss = step_loss.mean()  # Loss book-keeping

                print()
                print("{0: <16} |".format("milliseconds"), end="")
                print((" {:5d} |" * 7).format(*[80, 320, 640, 1000, 1520, 2000, 2520]))
                print()

                mean_loss = step_loss.detach().cpu().mean(dim=0).numpy()
                mean_loss = mean_loss.mean(axis=1)

                # Pretty print of the results for 80, 160, 320, 400, 560 and 1000 ms
                print("{0: <16} |".format(" "), end="")
                for ms in [1, 7, 15, 24, 37, 49, 62]:
                    if args.seq_length_out >= ms + 1:
                        print(" {0:.3f} |".format(mean_loss[ms]), end="")
                    else:
                        print("   n/a |", end="")
                print()

                print()
                print("============================\n"
                      "Global step:         %d\n"
                      "Learning rate:       %.4f\n"
                      "Step-time (ms):     %.4f\n"
                      "Train loss avg:      %.4f\n"
                      "--------------------------\n"
                      "Test loss:            %.4f\n"
                      "============================" % (current_step,
                                                        args.learning_rate, step_time * 1000, loss,
                                                        val_loss))

                torch.save(model, args.train_dir + '/model_' + str(current_step))

                print()
                previous_losses.append(loss)
                val_losses.append(val_loss)
                save_ixs.append(current_step)

                # Reset global time and loss
                step_time, loss = 0, 0

                sys.stdout.flush()

        best_step = save_ixs[np.argmin(val_losses)]
        best_model = torch.load(args.train_dir + '/model_' + str(best_step))
        print("<><><><><><><><><><><><><>\nBest model is at step: {:d}.\n<><><><><><><><><><><><><>\n".formt(best_step))
        torch.save(best_model, args.train_dir + '/model_best')


def sample(args):
    """Sample predictions for srnn's seeds"""
    actions = define_actions(args.action)

    train_set_Y, train_set_U, test_set_Y, test_set_U = read_all_data(
        args.seq_length_in, args.seq_length_out, args.data_dir, args.style_ix)

    if True:
        # === Create the model ===
        print("Creating %d layers of %d units." % (args.num_layers, args.size))
        sampling = True
        model = create_model(args, actions, sampling)
        if not args.use_cpu:
            model = model.cuda()
        print("Model created")

        # Clean and create a new h5 file of samples
        SAMPLES_FNAME = 'samples.h5'
        try:
            os.remove(SAMPLES_FNAME)
        except OSError:
            pass

        # Make prediction with srnn' seeds
        encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch(test_set_Y, test_set_U, -1)
        encoder_inputs = torch.from_numpy(encoder_inputs).float()
        decoder_inputs = torch.from_numpy(decoder_inputs).float()
        decoder_outputs = torch.from_numpy(decoder_outputs).float()
        if not args.use_cpu:
            encoder_inputs = encoder_inputs.cuda()
            decoder_inputs = decoder_inputs.cuda()
            decoder_outputs = decoder_outputs.cuda()
        encoder_inputs = Variable(encoder_inputs)
        decoder_inputs = Variable(decoder_inputs)
        decoder_outputs = Variable(decoder_outputs)

        preds = model(encoder_inputs, decoder_inputs, not args.use_cpu)

        loss = (preds - decoder_outputs) ** 2
        loss.cpu().data.numpy()
        loss = loss.mean()

        preds = preds.cpu().data.numpy()
        preds = preds.transpose([1, 0, 2])

        loss = loss.cpu().data.numpy()

        np.savez("predictions_{0}.npz".format(args.style_ix), preds=preds, actual=decoder_outputs)

    return


def define_actions(action):
    """
    Define the list of actions we are using.

    Args
      action: String with the passed action. Could be "all"
    Returns
      actions: List of strings of actions
    Raises
      ValueError if the action is not included in H3.6M
    """

    actions = ["walking", "eating", "smoking", "discussion", "directions",
               "greeting", "phoning", "posing", "purchases", "sitting",
               "sittingdown", "takingphoto", "waiting", "walkingdog",
               "walkingtogether"]

    if action in actions:
        return [action]

    if action == "all":
        return actions

    if action == "all_srnn":
        return ["walking", "eating", "smoking", "discussion"]

    raise (ValueError, "Unrecognized action: %d" % action)


def read_all_data(args):

    # === Read training data ===
    print("Reading training data (seq_len_in: {0}, seq_len_out {1}).".format(
        args.seq_length_in, args.seq_length_out))

    style_ixs = set(range(1, 9)) - {args.style_ix}
    style_lkp = np.load(os.path.join(args.data_dir, args.stylelkp_fname))

    load_Y = np.load(os.path.join(args.data_dir, args.output_fname))
    load_U = np.load(os.path.join(args.data_dir, args.input_fname))

    if args.train_set_size == -1:
        pct_train = 0.875

        train_ix_end = np.floor([sum([load_Y[str(i)].shape[0] for i in style_lkp[str(j)]]) * pct_train for j in style_ixs])
        train_ix_end = train_ix_end.astype('int')
        train_len_cum = [np.cumsum([load_Y[str(i)].shape[0] for i in style_lkp[str(j)]]) for j in style_ixs]

        train_set_Y, train_set_U, valid_set_Y, valid_set_U = [], [], [], []
        for j, e, cumsumlens in zip(style_ixs, train_ix_end, train_len_cum):
            found_breakpt = False
            cum_prv = 0
            for i, cuml in enumerate(cumsumlens):
                load_ix = str(style_lkp[str(j)][i])
                if cuml < e:
                    train_set_Y.append(load_Y[load_ix])
                    train_set_U.append(load_U[load_ix])
                    cum_prv = cuml
                elif not found_breakpt:
                    train_set_Y.append(load_Y[load_ix][:e - cum_prv, :])
                    train_set_U.append(load_U[load_ix][:e - cum_prv, :])
                    valid_set_Y.append(load_Y[load_ix][e - cum_prv:, :])
                    valid_set_U.append(load_U[load_ix][e - cum_prv:, :])
                    found_breakpt = True
                else:
                    valid_set_Y.append(load_Y[load_ix])
                    valid_set_U.append(load_U[load_ix])
    else:
        train_set_Y, train_set_U = [], []
        num_each = args.train_set_size // 4
        step = args.train_set_size
        for i in np.sort(list(style_ixs)):
            train_set_Y.append(np.concatenate([load_Y[str((i-1) * step + j + 1)] for j in range(num_each)], axis=0))
            train_set_U.append(np.concatenate([load_U[str((i-1) * step + j + 1)] for j in range(num_each)], axis=0))

        valid_Y = np.load(os.path.join(args.data_dir, "edin_Ys_30fps_variableN_test_valids_all.npz"))
        valid_U = np.load(os.path.join(args.data_dir, "edin_Us_30fps_variableN_test_valids_all.npz"))
        valid_set_Y, valid_set_U = [], []
        num_valid_each = 4
        for i in np.sort(list(style_ixs)):
            for j in range(num_valid_each):
                valid_set_Y.append(valid_Y[str((i-1) * num_valid_each + j + 1)])
                valid_set_U.append(valid_U[str((i-1) * num_valid_each + j + 1)])

    print("done reading data.")

    return train_set_Y, train_set_U, valid_set_Y, valid_set_U


def main(args=None):
    args = parseopts4bmark.parse_args(args)
    args = parseopts4bmark.initial_arg_transform(args)

    print(args.train_dir)
    os.makedirs(args.train_dir, exist_ok=True)

    if args.sample:
        sample(args)
    else:
        train(args)

    return args


if __name__ == "__main__":
    main()
