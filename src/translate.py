
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
torch.cuda.set_device(1)


learning_rate = .005
learning_rate_decay_factor = 0.95
learning_rate_step = 10000
max_gradient_norm = 5
batch_size = 16
iterations = int(1e5)
architecture = 'tied'
size = 1024
num_layers = 1
seq_length_in = 50
seq_length_out = 25
omit_one_hot = False
residual_velocities = False
data_dir = './data/h3.6m/dataset'
action = 'walking'
loss_to_use = 'sampling_based'
test_every = 1000
save_every = 1000
dosample = False
#dosample = True
use_cpu = False
train_dir = './experiments/' + action + '/'
if dosample:
    load = 1
else:
    load = 0

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

train_dir = os.path.normpath(os.path.join( train_dir, action,
  'out_{0}'.format(seq_length_out),
  'iterations_{0}'.format(iterations),
  architecture,
  loss_to_use,
  'omit_one_hot' if omit_one_hot else 'one_hot',
  'depth_{0}'.format(num_layers),
  'size_{0}'.format(size),
  'lr_{0}'.format(learning_rate),
  'residual_vel' if residual_velocities else 'not_residual_vel'))

def create_model(actions, sampling=False):
  """Create translation model and initialize or load parameters in session."""

  model = seq2seq_model.Seq2SeqModel(
      architecture,
      seq_length_in if not sampling else 50,
      seq_length_out if not sampling else 100,
      size, # hidden layer size
      num_layers,
      max_gradient_norm,
      batch_size,
      learning_rate,
      learning_rate_decay_factor,
      loss_to_use if not sampling else "sampling_based",
      len( actions ),
      not omit_one_hot,
      residual_velocities,
      dtype=torch.float32)

  if load <= 0:
    return model

  print("Loading model")
  model = torch.load(train_dir + 'Best_model')
  return model


def train():
  learning_rate = .005
  learning_rate_decay_factor = 0.95
  learning_rate_step = 10000
  max_gradient_norm = 5
  batch_size = 16
  iterations = int(1e5)
  architecture = 'tied'
  size = 1024
  num_layers = 1
  seq_length_in = 50
  seq_length_out = 25
  omit_one_hot = False
  residual_velocities = False
  data_dir = './data/h3.6m/dataset'
  action = 'walking'
  loss_to_use = 'sampling_based'
  test_every = 1000
  save_every = 1000
  train_dir = './experiments/' + action + '/'
  dosample = True
  use_cpu = False
  load = 0

  """Train a seq2seq model on human motion"""

  actions = define_actions( action )

  number_of_actions = len( actions )

  train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data(
    actions, seq_length_in, seq_length_out, data_dir, not omit_one_hot )

  # Limit TF to take a fraction of the GPU memory

  if True:
    model = create_model(actions)
    if not use_cpu:
        model = model.cuda()

    # === Read and denormalize the gt with srnn's seeds, as we'll need them
    # many times for evaluation in Euler Angles ===
    srnn_gts_euler = get_srnn_gts( actions, model, test_set, data_mean,
                              data_std, dim_to_ignore, not omit_one_hot )

    #=== This is the training loop ===
    step_time, loss, val_loss = 0.0, 0.0, 0.0
    current_step = 0 if load <= 0 else load + 1
    previous_losses = []

    step_time, loss = 0, 0
    optimiser = optim.SGD(model.parameters(), lr=learning_rate)
    #optimiser = optim.Adam(model.parameters(), lr=learning_rate, betas = (0.9, 0.999))
    best_srnn_loss = np.Inf

    for _ in range( iterations ):
      optimiser.zero_grad()
      model.train()

      start_time = time.time()

      # Actual training

      # === Training step ===
      encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch( train_set, not omit_one_hot )
      encoder_inputs = torch.from_numpy(encoder_inputs).float()
      decoder_inputs = torch.from_numpy(decoder_inputs).float()
      decoder_outputs = torch.from_numpy(decoder_outputs).float()
      if not use_cpu:
        encoder_inputs = encoder_inputs.cuda()
        decoder_inputs = decoder_inputs.cuda()
        decoder_outputs = decoder_outputs.cuda()
      encoder_inputs = Variable(encoder_inputs)
      decoder_inputs = Variable(decoder_inputs)
      decoder_outputs = Variable(decoder_outputs)

      preds = model(encoder_inputs, decoder_inputs)

      step_loss = (preds-decoder_outputs)**2
      step_loss = step_loss.mean()
    
      # Actual backpropagation
      step_loss.backward()
      optimiser.step()

      step_loss = step_loss.cpu().data.numpy()

      if current_step % 10 == 0:
        print("step {0:04d}; step_loss: {1:.4f}".format(current_step, step_loss ))

      step_time += (time.time() - start_time) / test_every
      loss += step_loss / test_every
      current_step += 1
      # === step decay ===
      if current_step % learning_rate_step == 0:
        learning_rate = learning_rate*learning_rate_decay_factor
        optimiser = optim.Adam(model.parameters(), lr=learning_rate, betas = (0.9, 0.999))
        print("Decay learning rate. New value at " + str(learning_rate))

      #cuda.empty_cache()

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % test_every == 0:
        model.eval()

        # === Validation with randomly chosen seeds ===
        encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch( test_set, not omit_one_hot )
        encoder_inputs = torch.from_numpy(encoder_inputs).float()
        decoder_inputs = torch.from_numpy(decoder_inputs).float()
        decoder_outputs = torch.from_numpy(decoder_outputs).float()
        if not use_cpu:
          encoder_inputs = encoder_inputs.cuda()
          decoder_inputs = decoder_inputs.cuda()
          decoder_outputs = decoder_outputs.cuda()
        encoder_inputs = Variable(encoder_inputs)
        decoder_inputs = Variable(decoder_inputs)
        decoder_outputs = Variable(decoder_outputs)
  
        preds = model(encoder_inputs, decoder_inputs)
  
        step_loss = (preds-decoder_outputs)**2
        step_loss = step_loss.mean()

        val_loss = step_loss # Loss book-keeping

        print()
        print("{0: <16} |".format("milliseconds"), end="")
        for ms in [80, 160, 320, 400, 560, 1000]:
          print(" {0:5d} |".format(ms), end="")
        print()

        # === Validation with srnn's seeds ===
        for action in actions:

          # Evaluate the model on the test batches
          encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch_srnn( test_set, action )
          #### Evaluate model on action
  
          encoder_inputs = torch.from_numpy(encoder_inputs).float()
          decoder_inputs = torch.from_numpy(decoder_inputs).float()
          decoder_outputs = torch.from_numpy(decoder_outputs).float()
          if not use_cpu:
            encoder_inputs = encoder_inputs.cuda()
            decoder_inputs = decoder_inputs.cuda()
            decoder_outputs = decoder_outputs.cuda()
          encoder_inputs = Variable(encoder_inputs)
          decoder_inputs = Variable(decoder_inputs)
          decoder_outputs = Variable(decoder_outputs)
    
          srnn_poses = model(encoder_inputs, decoder_inputs)


          srnn_loss = (srnn_poses - decoder_outputs)**2
          srnn_loss.cpu().data.numpy()
          srnn_loss = srnn_loss.mean()

          srnn_poses = srnn_poses.cpu().data.numpy()
          srnn_poses = srnn_poses.transpose([1,0,2])

          srnn_loss = srnn_loss.cpu().data.numpy()
          # Denormalize the output
          srnn_pred_expmap = data_utils.revert_output_format( srnn_poses,
            data_mean, data_std, dim_to_ignore, actions, not omit_one_hot )

          # Save the errors here
          mean_errors = np.zeros( (len(srnn_pred_expmap), srnn_pred_expmap[0].shape[0]) )

          # Training is done in exponential map, but the error is reported in
          # Euler angles, as in previous work.
          # See https://github.com/asheshjain399/RNNexp/issues/6#issuecomment-247769197
          N_SEQUENCE_TEST = 8
          for i in np.arange(N_SEQUENCE_TEST):
            eulerchannels_pred = srnn_pred_expmap[i]

            # Convert from exponential map to Euler angles
            for j in np.arange( eulerchannels_pred.shape[0] ):
              for k in np.arange(3,97,3):
                eulerchannels_pred[j,k:k+3] = data_utils.rotmat2euler(
                  data_utils.expmap2rotmat( eulerchannels_pred[j,k:k+3] ))

            # The global translation (first 3 entries) and global rotation
            # (next 3 entries) are also not considered in the error, so the_key
            # are set to zero.
            # See https://github.com/asheshjain399/RNNexp/issues/6#issuecomment-249404882
            gt_i=np.copy(srnn_gts_euler[action][i])
            gt_i[:,0:6] = 0

            # Now compute the l2 error. The following is numpy port of the error
            # function provided by Ashesh Jain (in matlab), available at
            # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/motionGenerationError.m#L40-L54
            idx_to_use = np.where( np.std( gt_i, 0 ) > 1e-4 )[0]
            
            euc_error = np.power( gt_i[:,idx_to_use] - eulerchannels_pred[:,idx_to_use], 2)
            euc_error = np.sum(euc_error, 1)
            euc_error = np.sqrt( euc_error )
            mean_errors[i,:] = euc_error

          # This is simply the mean error over the N_SEQUENCE_TEST examples
          mean_mean_errors = np.mean( mean_errors, 0 )

          # Pretty print of the results for 80, 160, 320, 400, 560 and 1000 ms
          print("{0: <16} |".format(action), end="")
          for ms in [1,3,7,9,13,24]:
            if seq_length_out >= ms+1:
              print(" {0:.3f} |".format( mean_mean_errors[ms] ), end="")
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
              "Val loss:            %.4f\n"
              "srnn loss:           %.4f\n"
              "============================" % (current_step,
              learning_rate, step_time*1000, loss,
              val_loss, srnn_loss))
        print()
        print()
        if best_srnn_loss > srnn_loss:
            best_srnn_loss = srnn_loss
            print("Saving model!")
            torch.save(model, train_dir + 'Best_model')
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
  learning_rate = .005
  learning_rate_decay_factor = 0.95
  learning_rate_step = 10000
  max_gradient_norm = 5
  batch_size = 16
  iterations = int(1e5)
  architecture = 'tied'
  size = 1024
  num_layers = 1
  seq_length_in = 50
  seq_length_out = 25
  omit_one_hot = False
  residual_velocities = False
  data_dir = './data/h3.6m/dataset'
  train_dir = './experiments/'
  action = 'walking'
  loss_to_use = 'sampling_based'
  test_every = 1000
  save_every = 1000
  #dosample = False
  dosample = True
  use_cpu = False
  load = 0

  actions = define_actions( action )

  if True:
    # === Create the model ===
    print("Creating %d layers of %d units." % (num_layers, size))
    sampling     = True
    model = create_model(actions, sampling)
    if not use_cpu:
        model = model.cuda()
    print("Model created")

    # Load all the data
    train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data(
      actions, seq_length_in, seq_length_out, data_dir, not omit_one_hot )

    # === Read and denormalize the gt with srnn's seeds, as we'll need them
    # many times for evaluation in Euler Angles ===
    srnn_gts_expmap = get_srnn_gts( actions, model, test_set, data_mean,
                              data_std, dim_to_ignore, not omit_one_hot, to_euler=False )
    srnn_gts_euler = get_srnn_gts( actions, model, test_set, data_mean,
                              data_std, dim_to_ignore, not omit_one_hot )

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
      if not use_cpu:
        encoder_inputs = encoder_inputs.cuda()
        decoder_inputs = decoder_inputs.cuda()
        decoder_outputs = decoder_outputs.cuda()
      encoder_inputs = Variable(encoder_inputs)
      decoder_inputs = Variable(decoder_inputs)
      decoder_outputs = Variable(decoder_outputs)

      srnn_poses = model(encoder_inputs, decoder_inputs)

      srnn_loss = (srnn_poses - decoder_outputs)**2
      srnn_loss.cpu().data.numpy()
      srnn_loss = srnn_loss.mean()

      srnn_poses = srnn_poses.cpu().data.numpy()
      srnn_poses = srnn_poses.transpose([1,0,2])

      srnn_loss = srnn_loss.cpu().data.numpy()
      # denormalizes too
      srnn_pred_expmap = data_utils.revert_output_format(srnn_poses, data_mean, data_std, dim_to_ignore, actions, not omit_one_hot )

      # Save the conditioning seeds

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


def read_all_data( actions, seq_length_in, seq_length_out, data_dir, one_hot ):
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

  train_subject_ids = [1,6,7,8,9,11]
  test_subject_ids = [5]

  train_set, complete_train = data_utils.load_data( data_dir, train_subject_ids, actions, one_hot )
  test_set,  complete_test  = data_utils.load_data( data_dir, test_subject_ids,  actions, one_hot )

  # Compute normalization stats
  data_mean, data_std, dim_to_ignore, dim_to_use = data_utils.normalization_stats(complete_train)

  # Normalize -- subtract mean, divide by stdev
  train_set = data_utils.normalize_data( train_set, data_mean, data_std, dim_to_use, actions, one_hot )
  test_set  = data_utils.normalize_data( test_set,  data_mean, data_std, dim_to_use, actions, one_hot )
  print("done reading data.")

  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use


def main():
  if dosample:
    sample()
  else:
    train()

if __name__ == "__main__":
    main()
