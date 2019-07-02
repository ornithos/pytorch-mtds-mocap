"""Simple code for training an RNN for motion prediction."""

import os
import sys
import time

import numpy as np

import data_utils
import mt_model
import torch
import torch.optim as optim
from torch.autograd import Variable
import argparse

# Learning
parser = argparse.ArgumentParser(description='Train MT-RNN for human pose estimation')
parser.add_argument('--style_ix', dest='style_ix',
                    help='Style index to hold out', type=int, required=True)
parser.add_argument('--learning_rate', dest='learning_rate',
                    help='Learning rate',
                    default=0.005, type=float)
parser.add_argument('--learning_rate_decay_factor', dest='learning_rate_decay_factor',
                    help='Learning rate is multiplied by this much. 1 means no decay.',
                    default=0.95, type=float)
parser.add_argument('--learning_rate_step', dest='learning_rate_step',
                    help='Every this many steps, do decay.',
                    default=10000, type=int)
parser.add_argument('--batch_size', dest='batch_size',
                    help='Batch size to use during training.',
                    default=16, type=int)
parser.add_argument('--iterations', dest='iterations',
                    help='Iterations to train for.',
                    default=1e5, type=int)
parser.add_argument('--test_every', dest='test_every',
                    help='',
                    default=200, type=int)
parser.add_argument('--optimiser', dest='optimiser',
                    help='Optimiser: SGD, Nesterov, or ADAM',
                    default="SGD", type=str)
parser.add_argument('--first3_prec', dest='first3_prec',
                    help='Precision of noise model of first 3 outputs.',
                    default=1.0, type=float)

# Architecture
parser.add_argument('--residual_velocities', dest='residual_velocities',
                    help='Add a residual connection that effectively models velocities', action='store_true',
                    default=True)
parser.add_argument('--latent_k', dest='k',
                    help='Dimension of parameter manifold.', type=int, required=True)
parser.add_argument('--decoder_size', dest='decoder_size',
                    help='Size of decoder recurrent state.',
                    default=1024, type=int)
parser.add_argument('--encoder_size', dest='encoder_size',
                    help='Size of encoder recurrent state.',
                    default=512, type=int)
parser.add_argument('--size_psi_hidden', dest='size_psi_hidden',
                    help='Size of NL hidden layer of psi network.',
                    default=200, type=int)
parser.add_argument('--size_psi_lowrank', dest='size_psi_lowrank',
                    help='Subspace dimension to embed parameter manifold into. This is to reduce par count.',
                    default=30, type=int)
parser.add_argument('--seq_length_out', dest='seq_length_out',
                    help='Number of frames that the decoder has to predict. 25fps',
                    default=64, type=int)
parser.add_argument('--input_size', dest='input_size',
                    help='Input dimension at each timestep',
                    required=True, type=int)
parser.add_argument('--human_size', dest='human_size',
                    help='Output dimension at each timestep',
                    default=64, type=int)
parser.add_argument('--dropout_p', dest='dropout_p',
                    help='Dropout probability for hidden layers',
                    default=0.0, type=float)

# Directories
parser.add_argument('--data_dir', dest='data_dir',
                    help='Data directory',
                    default=os.path.normpath("../../mocap-mtds/"), type=str)
                    # default=os.path.normpath("../../mocap-mtds/data/"), type=str)
parser.add_argument('--train_dir', dest='train_dir',
                    help='Training directory',
                    default=os.path.normpath("./experiments/"), type=str)
parser.add_argument('--use_cpu', dest='use_cpu',
                    help='', action='store_true',
                    default=False)
parser.add_argument('--load', dest='load',
                    help='Try to load a previous checkpoint.',
                    default='', type=str)
parser.add_argument('--sample', dest='sample',
                    help='Set to True for sampling.', action='store_true',
                    default=False)

args = parser.parse_args()
assert args.dropout_p == 0.0, "dropout not implemented yet."

if not os.path.isfile(os.path.join(args.data_dir, "style_lkp.npz")):
    args.data_dir = os.path.normpath("../../mocap-mtds/data/")

train_dir = os.path.normpath(os.path.join(args.train_dir,
                                          'style_{0}'.format(args.style_ix),
                                          'out_{0}'.format(args.seq_length_out),
                                          'iterations_{0}'.format(args.iterations),
                                          'decoder_size_{0}'.format(args.decoder_size),
                                          'zdim_{0}'.format(args.k),
                                          'psi_lowrank_{0}'.format(args.size_psi_lowrank),
                                          'optim_{0}'.format(args.optimiser),
                                          'lr_{0}'.format(args.learning_rate),
                                          'residual_vel' if args.residual_velocities else 'not_residual_vel'))

print(train_dir)
os.makedirs(train_dir, exist_ok=True)


def create_model():
    """Create MT model and initialize or load parameters in session."""

    model = mt_model.MTGRU(
        args.seq_length_out,
        args.decoder_size,
        args.encoder_size,
        args.batch_size,
        args.k,
        args.size_psi_hidden,
        args.size_psi_lowrank,
        args.human_size,
        args.input_size,
        args.dropout_p,
        args.residual_velocities)

    if len(args.load) <= 0:
        return model

    print("Loading model")
    model = torch.load(args.load, map_location='cpu') if args.use_cpu else torch.load(args.load)
    return model


def train():
    """Train a MT model on human motion"""

    train_set_Y, train_set_U, test_set_Y, test_set_U = read_all_data(args.data_dir, args.style_ix, args.human_size)

    model = create_model()
    if not args.use_cpu:
        model = model.cuda()

    assert len(args.load) == 0, "Training from load file no longer supported in this fork."

    has_weight = not np.isclose(args.first3_prec, 1.0)
    current_step = 0
    previous_losses = []

    step_time, loss = 0, 0
    if args.optimiser.upper() == "SGD":
        optimiser = optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimiser.upper() == "NESTEROV":
        optimiser = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.8, nesterov=True)
    elif args.optimiser.upper() == "ADAM":
        optimiser = optim.Adam(model.parameters(), lr=args.learning_rate, betas = (0.9, 0.999))
    else:
        Exception("Unknown optimiser type: {:d}. Try 'SGD', 'Nesterov' or 'Adam'")

    for _ in range(args.iterations):
        optimiser.zero_grad()
        model.train()

        start_time = time.time()

        # ------------------------------------------------------- TRAINING
        inputs, outputs = model.get_batch(train_set_Y, train_set_U)
        inputs = Variable(torch.from_numpy(inputs).float())
        outputs = Variable(torch.from_numpy(outputs).float())
        if not args.use_cpu:
            inputs, outputs = inputs.cuda(), outputs.cuda()

        preds, mu, logstd, state = model(inputs, outputs)

        sqerr = (preds - outputs) ** 2
        if has_weight:
            sqerr = sqerr * torch.cat((torch.ones(1,1,3) * args.first3_prec, torch.ones(1,1,args.human_size-3)),
                                              dim=2).to(sqerr.device)
        step_loss = args.human_size * args.seq_length_out * sqerr.mean() / 2

        # assume \sigma is const. wrt optimisation, and hence normalising constant can be ignored.
        # Now for KL term. Since we're descending *negative* L.B., we need to *ADD* KL to loss:
        KLD = -0.5 * torch.sum(1 + 2*logstd - mu.pow(2) - torch.exp(2*logstd))
        step_loss = step_loss + KLD

        # Actual backpropagation
        step_loss.backward()
        optimiser.step()
        # -------------------------------------------------------

        # Reporting / admin
        step_loss = step_loss.cpu().data.numpy()

        if True: #current_step % 10 == 0:
            print("step {0:04d}; step_loss: {1:.4f}".format(current_step, step_loss))

        step_time += (time.time() - start_time) / args.test_every
        loss += step_loss / args.test_every
        current_step += 1

        # Decay learning rate (if appl.)
        if current_step % args.learning_rate_step == 0:
            args.learning_rate = args.learning_rate * args.learning_rate_decay_factor
            for param_group in optimiser.param_groups:
                param_group['lr'] = args.learning_rate
            print("Decay learning rate. New value at " + str(args.learning_rate))

        # Once in a while, we save checkpoint, print statistics, and run evals.
        if current_step % args.test_every == 0:
            model.eval()

            # === Validation with random data from test set ===
            inputs, outputs = model.get_test_batch(test_set_Y, test_set_U, -1)

            inputs = Variable(torch.from_numpy(inputs).float())
            outputs = Variable(torch.from_numpy(outputs).float())
            if not args.use_cpu:
                inputs, outputs, inputs.cuda(), outputs.cuda()

            preds, mu, logstd, state = model(inputs, outputs)

            sqerr = (preds - outputs) ** 2
            if has_weight:
                sqerr = sqerr * torch.cat((torch.ones(1,1,3) * args.first3_prec, torch.ones(1,1,args.human_size-3)),
                    dim=2).to(sqerr.device)

            val_loss = args.human_size * args.seq_length_out * sqerr.mean() / 2
            KLD = -0.5 * torch.sum(1 + 2 * logstd - mu.pow(2) - torch.exp(2 * logstd))
            val_loss = val_loss + KLD

            print()
            print("{0: <16} |".format("milliseconds"), end="")
            for ms in [60, 240, 480, 750, 990, 1500, 2010]:
                print(" {0:5d} |".format(ms), end="")
            print()

            avg_mse_tt = sqerr.detach().cpu().mean(dim=0).numpy().mean(axis=1)

            # Pretty print of the results for 60, 240, 480, 750, 990, 1500, 2010 ms
            print("{0: <16} |".format(" "), end="")
            for ms in [1, 7, 15, 24, 32, 49, 66]:
                if args.seq_length_out >= ms + 1:
                    print(" {0:.3f} |".format(avg_mse_tt[ms]), end="")
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

            torch.save(model, train_dir + '/model_' + str(current_step))

            print()
            previous_losses.append(loss)

            # Reset global time and loss
            step_time, loss = 0, 0

            sys.stdout.flush()


def sample():

    train_set_Y, train_set_U, test_set_Y, test_set_U = read_all_data(args.data_dir, args.style_ix, args.human_size)

    model = create_model()
    model.eval()
    if not args.use_cpu:
        model = model.cuda()
    print("Model created")

    # Make prediction with srnn' seeds
    inputs, outputs = model.get_test_batch(test_set_Y, test_set_U, -1)

    inputs = Variable(torch.from_numpy(inputs).float())
    outputs = Variable(torch.from_numpy(outputs).float())
    if not args.use_cpu:
        inputs, outputs, inputs.cuda(), outputs.cuda()

    preds, mu, logstd, state = model(inputs, outputs)

    loss = (preds - outputs) ** 2
    loss.cpu().data.numpy()
    loss = loss.mean()

    preds = preds.cpu().data.numpy()
    preds = preds.transpose([1, 0, 2])

    loss = loss.cpu().data.numpy()

    np.savez("mt_predictions_{0}.npz".format(args.style_ix), preds=preds, actual=outputs)

    return


def read_all_data(data_dir, style_ix, njoints):
    """
    Loads data for training/testing and normalizes it.

    Args
      data_dir: directory to load the data from
      style_ix: style index of the test set (and leave out from the training set)
      njoints: number of joints to model (0 or -1 = all)
    Returns
      train_set: dictionary with normalized training data
      test_set: dictionary with test data
      data_mean: d-long vector with the mean of the training data
      data_std: d-long vector with the standard dev of the training data
      dim_to_ignore: dimensions that are not used becaused stdev is too small
      dim_to_use: dimensions that we are actually using in the model
    """

    # === Read training data ===
    print("Reading training data (test index {0:d}).".format(style_ix))

    style_lkp = np.load(os.path.join(data_dir, "styles_lkp.npz"))
    train_ixs = np.concatenate([style_lkp[str(i)] for i in range(1, 9) if i != style_ix])
    train_set_Y = np.load(os.path.join(data_dir, "edin_Ys_30fps.npz"))
    train_set_U = np.load(os.path.join(data_dir, "edin_Us_30fps.npz"))
    njoints = train_set_Y[str(0)].shape[1] if njoints <= 0 else njoints
    train_set_Y = [train_set_Y[str(i)][:, :njoints] for i in train_ixs]
    train_set_U = [train_set_U[str(i)] for i in train_ixs]

    test_set_Y = np.load(os.path.join(data_dir, "test_input_{0}_y.npz".format(style_ix)))
    test_set_U = np.load(os.path.join(data_dir, "test_input_{0}_u.npz".format(style_ix)))
    test_set_Y = [test_set_Y[str(i + 1)][:njoints, :] for i in range(len(test_set_Y.keys()))]  # whatever, apparently test is transpose of train
    test_set_U = [test_set_U[str(i + 1)] for i in range(len(test_set_U.keys()))]

    print("done reading data.")

    return train_set_Y, train_set_U, test_set_Y, test_set_U


def main():
    if args.sample:
        sample()
    else:
        train()


if __name__ == "__main__":
    main()
