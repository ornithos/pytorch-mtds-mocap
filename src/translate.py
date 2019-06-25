
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
from six.moves import xrange # pylint: disable=redefined-builtin

import data_utils
import seq2seq_model
import torch
import torch.optim as optim
from torch.autograd import Variable
import argparse

# Learning
parser = argparse.ArgumentParser(description='Train RNN for human pose estimation')
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
parser.add_argument('--max_gradient_norm', dest='max_gradient_norm',
                  help='Clip gradients to this norm.',
                  default=5, type=float)
parser.add_argument('--iterations', dest='iterations',
                  help='Iterations to train for.',
                  default=1e5, type=int)
parser.add_argument('--test_every', dest='test_every',
                  help='',
                  default=200, type=int)
# Architecture
parser.add_argument('--architecture', dest='architecture',
                  help='Seq2seq architecture to use: [basic, tied].',
                  default='tied', type=str)
parser.add_argument('--loss_to_use', dest='loss_to_use',
                  help='The type of loss to use, supervised or sampling_based',
                  default='sampling_based', type=str)
parser.add_argument('--residual_velocities', dest='residual_velocities',
                  help='Add a residual connection that effectively models velocities',action='store_true',
                  default=True)
parser.add_argument('--size', dest='size',
                  help='Size of each model layer.',
                  default=1024, type=int)
parser.add_argument('--num_layers', dest='num_layers',
                  help='Number of layers in the model.',
                  default=1, type=int)
parser.add_argument('--seq_length_in', dest='seq_length_in',
                  help='Number of frames to feed into the encoder. 25 fp',
                  default=64, type=int)
parser.add_argument('--seq_length_out', dest='seq_length_out',
                  help='Number of frames that the decoder has to predict. 25fps',
                  default=64, type=int)
parser.add_argument('--omit_one_hot', dest='omit_one_hot',
                  help='', action='store_true',
                  default=True)
# Directories
parser.add_argument('--data_dir', dest='data_dir',
                  help='Data directory',
                  default=os.path.normpath("../../mocap-mtds/data/"), type=str)
parser.add_argument('--train_dir', dest='train_dir',
                  help='Training directory',
                  default=os.path.normpath("./experiments/"), type=str)
parser.add_argument('--action', dest='action',
                  help='The action to train on. all means all the actions, all_periodic means walking, eating and smoking',
                  default='walking', type=str)
parser.add_argument('--use_cpu', dest='use_cpu',
                  help='', action='store_true',
                  default=False)
parser.add_argument('--load', dest='load',
                  help='Try to load a previous checkpoint.',
                  default=0, type=int)
parser.add_argument('--sample', dest='sample',
                  help='Set to True for sampling.', action='store_true',
                  default=False)

args = parser.parse_args()
assert args.omit_one_hot, "not implemented yet"
assert args.action == "walking", "not implemented yet"
assert args.residual_velocities, "not implemented yet. (Also not in original fork.)"
assert args.num_layers == 1, "not implemented yet. (Also not in original fork.)"
assert args.use_cpu, "need to check that there are no hardcoded CPU things about."

train_dir = os.path.normpath(os.path.join( args.train_dir, args.action,
  'out_{0}'.format(args.seq_length_out),
  'iterations_{0}'.format(args.iterations),
  args.architecture,
  args.loss_to_use,
  'omit_one_hot' if args.omit_one_hot else 'one_hot',
  'depth_{0}'.format(args.num_layers),
  'size_{0}'.format(args.size),
  'lr_{0}'.format(args.learning_rate),
  'residual_vel' if args.residual_velocities else 'not_residual_vel'))

print(train_dir)
os.makedirs(train_dir, exist_ok=True)

def create_model(actions, sampling=False):
  """Create translation model and initialize or load parameters in session."""

  model = seq2seq_model.Seq2SeqModel(
      args.architecture,
      args.seq_length_in if not sampling else 50,
      args.seq_length_out if not sampling else 100,
      args.size, # hidden layer size
      args.num_layers,
      args.max_gradient_norm,
      args.batch_size,
      args.learning_rate,
      args.learning_rate_decay_factor,
      args.loss_to_use if not sampling else "sampling_based",
      len( actions ),
      not args.omit_one_hot,
      args.residual_velocities,
      dtype=torch.float32,
      num_traj=44)

  if args.load <= 0:
    return model

  print("Loading model")
  model = torch.load(train_dir + '/model_' + str(args.load))
  if sampling:
    model.source_seq_len = 50
    model.target_seq_len = 100
  return model


def train():
  """Train a seq2seq model on human motion"""

  actions = define_actions( args.action )

  number_of_actions = len( actions )

  train_set_Y, train_set_U, test_set_Y, test_set_U = read_all_data(
    args.seq_length_in, args.seq_length_out, args.data_dir, args.style_ix)

  # Limit TF to take a fraction of the GPU memory

  if True:
    model = create_model(actions, args.sample)
    if not args.use_cpu:
        model = model.cuda()

    #=== This is the training loop ===
    current_step = 0 if args.load <= 0 else args.load + 1
    previous_losses = []

    step_time, loss = 0, 0
    optimiser = optim.SGD(model.parameters(), lr=args.learning_rate)
    #optimiser = optim.Adam(model.parameters(), lr=learning_rate, betas = (0.9, 0.999))

    for _ in range( args.iterations ):
      optimiser.zero_grad()
      model.train()

      start_time = time.time()

      # Actual training

      # === Training step ===
      encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch(train_set_Y, train_set_U, not args.omit_one_hot)
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

      step_loss = (preds-decoder_outputs)**2
      step_loss = step_loss.mean()
    
      # Actual backpropagation
      step_loss.backward()
      optimiser.step()

      step_loss = step_loss.cpu().data.numpy()

      if current_step % 10 == 0:
        print("step {0:04d}; step_loss: {1:.4f}".format(current_step, step_loss ))

      step_time += (time.time() - start_time) / args.test_every
      loss += step_loss / args.test_every
      current_step += 1
      # === step decay ===
      if current_step % args.learning_rate_step == 0:
        args.learning_rate = args.learning_rate*args.learning_rate_decay_factor
        optimiser = optim.Adam(model.parameters(), lr=args.learning_rate, betas = (0.9, 0.999))
        print("Decay learning rate. New value at " + str(args.learning_rate))

      #cuda.empty_cache()

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % args.test_every == 0:
        model.eval()

        # === Validation with random data from test set ===
        encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch(test_set_Y, test_set_U, not args.omit_one_hot)
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
  
        step_loss = (preds-decoder_outputs)**2

        val_loss = step_loss.mean() # Loss book-keeping

        print()
        print("{0: <16} |".format("milliseconds"), end="")
        for ms in [80, 320, 640, 1000, 1520, 2000, 2520]:
          print(" {0:5d} |".format(ms), end="")
        print()

        mean_loss = step_loss.mean(axis=0).numpy()
        mean_loss = mean_loss.mean(axis=1)

        # Pretty print of the results for 80, 160, 320, 400, 560 and 1000 ms
        print("{0: <16} |".format(action), end="")
        for ms in [1,7,15,24,37,49,62]:
          if args.seq_length_out >= ms+1:
            print(" {0:.3f} |".format( mean_loss[ms] ), end="")
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
              args.learning_rate, step_time*1000, loss,
              val_loss))

        torch.save(model, train_dir + '/model_' + str(current_step))

        print()
        previous_losses.append(loss)

        # Reset global time and loss
        step_time, loss = 0, 0

        sys.stdout.flush()


def get_srnn_gts( actions, model, test_set, data_mean, data_std, dim_to_ignore, one_hot, to_euler=True ):
  """
  Get the ground truths for srnn's sequences, and convert to Euler angles.
  (the error is always computed in Euler angles).

  Args
    actions: a list of actions to get ground truths for.
    model: training model we are using (we only use the "get_batch" method).
    test_set: dictionary with normalized training data.
    data_mean: d-long vector with the mean of the training data.
    data_std: d-long vector with the standard deviation of the training data.
    dim_to_ignore: dimensions that we are not using to train/predict.
    one_hot: whether the data comes with one-hot encoding indicating action.
    to_euler: whether to convert the angles to Euler format or keep thm in exponential map

  Returns
    srnn_gts_euler: a dictionary where the keys are actions, and the values
      are the ground_truth, denormalized expected outputs of srnns's seeds.
  """
  srnn_gts_euler = {}

  for action in actions:

    srnn_gt_euler = []
    _, _, srnn_expmap = model.get_batch_srnn( test_set, action )

    # expmap -> rotmat -> euler
    for i in np.arange( srnn_expmap.shape[0] ):
      denormed = data_utils.unNormalizeData(srnn_expmap[i,:,:], data_mean, data_std, dim_to_ignore, actions, one_hot )

      if to_euler:
        for j in np.arange( denormed.shape[0] ):
          for k in np.arange(3,97,3):
            denormed[j,k:k+3] = data_utils.rotmat2euler( data_utils.expmap2rotmat( denormed[j,k:k+3] ))

      srnn_gt_euler.append( denormed );

    # Put back in the dictionary
    srnn_gts_euler[action] = srnn_gt_euler

  return srnn_gts_euler


def sample():
  """Sample predictions for srnn's seeds"""
  actions = define_actions( args.action )

  if True:
    # === Create the model ===
    print("Creating %d layers of %d units." % (args.num_layers, args.size))
    sampling     = True
    model = create_model(actions, sampling)
    if not args.use_cpu:
        model = model.cuda()
    print("Model created")

    # Load all the data
    train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data(
      actions, args.seq_length_in, args.seq_length_out, args.data_dir, not args.omit_one_hot )

    # === Read and denormalize the gt with srnn's seeds, as we'll need them
    # many times for evaluation in Euler Angles ===
    srnn_gts_expmap = get_srnn_gts( actions, model, test_set, data_mean,
                              data_std, dim_to_ignore, not args.omit_one_hot, to_euler=False )
    srnn_gts_euler = get_srnn_gts( actions, model, test_set, data_mean,
                              data_std, dim_to_ignore, not args.omit_one_hot )

    # Clean and create a new h5 file of samples
    SAMPLES_FNAME = 'samples.h5'
    try:
      os.remove( SAMPLES_FNAME )
    except OSError:
      pass

    # Predict and save for each action
    for action in actions:

      # Make prediction with srnn' seeds
      encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch_srnn( test_set, action )

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

      srnn_poses = model(encoder_inputs, decoder_inputs, not args.use_cpu)

      srnn_loss = (srnn_poses - decoder_outputs)**2
      srnn_loss.cpu().data.numpy()
      srnn_loss = srnn_loss.mean()

      srnn_poses = srnn_poses.cpu().data.numpy()
      srnn_poses = srnn_poses.transpose([1,0,2])

      srnn_loss = srnn_loss.cpu().data.numpy()
      # denormalizes too
      srnn_pred_expmap = data_utils.revert_output_format(srnn_poses, data_mean, data_std, dim_to_ignore, actions, not args.omit_one_hot )

      # Save the samples
      with h5py.File( SAMPLES_FNAME, 'a' ) as hf:
        for i in np.arange(8):
          # Save conditioning ground truth
          node_name = 'expmap/gt/{1}_{0}'.format(i, action)
          hf.create_dataset( node_name, data=srnn_gts_expmap[action][i] )
          # Save prediction
          node_name = 'expmap/preds/{1}_{0}'.format(i, action)
          hf.create_dataset( node_name, data=srnn_pred_expmap[i] )

      # Compute and save the errors here
      mean_errors = np.zeros( (len(srnn_pred_expmap), srnn_pred_expmap[0].shape[0]) )

      for i in np.arange(8):

        eulerchannels_pred = srnn_pred_expmap[i]

        for j in np.arange( eulerchannels_pred.shape[0] ):
          for k in np.arange(3,97,3):
            eulerchannels_pred[j,k:k+3] = data_utils.rotmat2euler(
              data_utils.expmap2rotmat( eulerchannels_pred[j,k:k+3] ))

        eulerchannels_pred[:,0:6] = 0

        # Pick only the dimensions with sufficient standard deviation. Others are ignored.
        idx_to_use = np.where( np.std( eulerchannels_pred, 0 ) > 1e-4 )[0]

        euc_error = np.power( srnn_gts_euler[action][i][:,idx_to_use] - eulerchannels_pred[:,idx_to_use], 2)
        euc_error = np.sum(euc_error, 1)
        euc_error = np.sqrt( euc_error )
        mean_errors[i,:] = euc_error

      mean_mean_errors = np.mean( mean_errors, 0 )
      print( action )
      print( ','.join(map(str, mean_mean_errors.tolist() )) )

      with h5py.File( SAMPLES_FNAME, 'a' ) as hf:
        node_name = 'mean_{0}_error'.format( action )
        hf.create_dataset( node_name, data=mean_mean_errors )

  return


def define_actions( action ):
  """
  Define the list of actions we are using.

  Args
    action: String with the passed action. Could be "all"
  Returns
    actions: List of strings of actions
  Raises
    ValueError if the action is not included in H3.6M
  """

  actions = ["walking", "eating", "smoking", "discussion",  "directions",
              "greeting", "phoning", "posing", "purchases", "sitting",
              "sittingdown", "takingphoto", "waiting", "walkingdog",
              "walkingtogether"]

  if action in actions:
    return [action]

  if action == "all":
    return actions

  if action == "all_srnn":
    return ["walking", "eating", "smoking", "discussion"]

  raise( ValueError, "Unrecognized action: %d" % action )


def read_all_data(seq_length_in, seq_length_out, data_dir, style_ix):
  """
  Loads data for training/testing and normalizes it.

  Args
    actions: list of strings (actions) to load
    seq_length_in: number of frames to use in the burn-in sequence
    seq_length_out: number of frames to use in the output sequence
    data_dir: directory to load the data from
    one_hot: whether to use one-hot encoding per action
  Returns
    train_set: dictionary with normalized training data
    test_set: dictionary with test data
    data_mean: d-long vector with the mean of the training data
    data_std: d-long vector with the standard dev of the training data
    dim_to_ignore: dimensions that are not used becaused stdev is too small
    dim_to_use: dimensions that we are actually using in the model
  """

  # === Read training data ===
  print ("Reading training data (seq_len_in: {0}, seq_len_out {1}).".format(
           seq_length_in, seq_length_out))

  style_lkp = np.load(os.path.join(data_dir, "styles_lkp.npz"))
  train_ixs = np.concatenate([style_lkp[str(i)] for i in range(1,9) if i != style_ix])
  train_set_Y = np.load(os.path.join(data_dir, "edin_Ys_30fps.npz"))
  train_set_U = np.load(os.path.join(data_dir, "edin_Us_30fps.npz"))

  test_set_Y = [train_set_Y[str(i)] for i in style_lkp[str(style_ix)]]
  test_set_U = [train_set_U[str(i)] for i in style_lkp[str(style_ix)]]
  train_set_Y = [train_set_Y[str(i)] for i in train_ixs]
  train_set_U = [train_set_U[str(i)] for i in train_ixs]


  print("done reading data.")

  return train_set_Y, train_set_U, test_set_Y, test_set_U


def main():
  if args.sample:
    sample()
  else:
    train()

if __name__ == "__main__":
    main()
