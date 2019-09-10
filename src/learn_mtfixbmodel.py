"""Simple code for training an RNN for motion prediction."""

import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

import mtfixb_model, mtfixb_model2
import parseopts


def create_model(args, total_num_batches):
    """Create MT model and initialize or load parameters in session."""

    if len(args.load) > 0:
        print("Loading model")
        model = torch.load(args.load, map_location='cpu') if args.use_cpu else torch.load(args.load)
        return model

    if args.k == 0:
        return create_model_k0(args, total_num_batches)

    if args.dynamicsdict:
        return create_model_DD(args, total_num_batches)

    if args.biasonly:
        return create_model_BiasOnly(args, total_num_batches)

    if args.nobias:
        return create_model_NoMTBias(args, total_num_batches)

    model = mtfixb_model.MTGRU(
        args.seq_length_out,
        args.decoder_size,
        args.decoder_size2,
        args.batch_size,
        total_num_batches,
        args.k,
        args.size_psi_hidden,
        args.size_psi_lowrank,
        args.bottleneck,
        output_dim=args.human_size,
        input_dim=args.input_size,
        dropout=args.dropout_p,
        residual_output=args.residual_velocities,
        init_state_noise=args.init_state_noise,
        mt_rnn=args.mt_rnn)

    if len(args.load) <= 0:
        if len(args.load_layer1) > 0:
            print("Loading GRU2 model")
            model = load_layer1(model, args.load_layer1, args.use_cpu)
        return model

    print("Loading model")
    model = torch.load(args.load, map_location='cpu') if args.use_cpu else torch.load(args.load)
    return model


def create_model_k0(args, total_num_batches):
    """Create MT model and initialize or load parameters in session."""

    model = mtfixb_model.OpenLoopGRU(
        args.seq_length_out,
        args.decoder_size,
        args.batch_size,
        args.human_size,
        args.input_size,
        args.dropout_p,
        args.residual_velocities,
        args.init_state_noise)

    return model


def create_model_DD(args, total_num_batches):
    """Create MT model and initialize or load parameters in session."""

    if len(args.load_layer1) > 0:
        NotImplementedError("Layer 1 load not yet implemented for Dynamics Dict.")

    model = mtfixb_model.DynamicsDict(
        args.seq_length_out,
        args.decoder_size,
        total_num_batches,
        args.batch_size,
        args.k,
        args.size_psi_hidden,
        args.size_psi_lowrank,
        args.human_size,
        args.input_size,
        args.dropout_p,
        args.residual_velocities,
        args.init_state_noise)

    return model


def create_model_BiasOnly(args, total_num_batches):
    """Create MT model and initialize or load parameters in session."""

    if len(args.load_layer1) > 0:
        NotImplementedError("Layer 1 load not yet implemented for MT Bias Only.")

    model = mtfixb_model.MTGRU_BiasOnly(
        args.seq_length_out,
        args.decoder_size,
        args.decoder_size2,
        args.batch_size,
        total_num_batches,
        args.k,
        args.size_psi_hidden,
        args.size_psi_lowrank,
        args.bottleneck,
        output_dim=args.human_size,
        input_dim=args.input_size,
        dropout=args.dropout_p,
        residual_output=args.residual_velocities,
        init_state_noise=args.init_state_noise)

    return model


def create_model_NoMTBias(args, total_num_batches):
    """Create MT model and initialize or load parameters in session."""

    if len(args.load_layer1) > 0:
        NotImplementedError("Layer 1 load not yet implemented for MT Bias Only.")

    model = mtfixb_model2.MTGRU_NoBias(
        args.seq_length_out,
        args.decoder_size,
        args.decoder_size2,
        args.batch_size,
        total_num_batches,
        args.k,
        args.size_psi_hidden,
        args.size_psi_lowrank,
        args.bottleneck,
        output_dim=args.human_size,
        input_dim=args.input_size,
        dropout=args.dropout_p,
        residual_output=args.residual_velocities,
        init_state_noise=args.init_state_noise,
        mt_rnn=args.mt_rnn)

    return model


def train(args):
    """Train a MT model on human motion"""

    train_iter = read_all_data(args)
    train_iter.shuffle()

    total_num_batches = train_iter.total_length()

    model = create_model(args, total_num_batches)
    model = model if args.use_cpu else model.cuda()

    has_weight = not np.isclose(args.first3_prec, 1.0)
    is_hard_em = args.hard_em_iters > 0
    is_MT = args.k > 0
    current_step = 0
    previous_losses = []

    step_time, loss = 0, 0

    mt_lr = args.learning_rate_mt if args.learning_rate_mt >= 0 else args.learning_rate
    z_lr = args.learning_rate_z if args.learning_rate_z >= 0 else args.learning_rate
    zls_lr = 0 if is_hard_em else z_lr

    pars_lrs, zls_ix = model.get_params_optim_dicts(mt_lr, args.learning_rate, z_lr, zls_lr=zls_lr)
    if args.optimiser.upper() == "SGD":
        optimiser = optim.SGD(pars_lrs, weight_decay=args.weight_decay)
    elif args.optimiser.upper() == "NESTEROV":
        optimiser = optim.SGD(pars_lrs, momentum=0.8, nesterov=True, weight_decay=args.weight_decay)
    elif args.optimiser.upper() == "ADAM":
        optimiser = optim.Adam(pars_lrs, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    else:
        Exception("Unknown optimiser type: {:d}. Try 'SGD', 'Nesterov' or 'Adam'")

    has_ar_noise = args.ar_coef > 0
    device = "cpu" if args.use_cpu else "cuda"
    if has_ar_noise:
        assert args.ar_coef < 1, "ar_coef must be in [0, 1)."
        # Construct banded AR precision matrix (fn def below)
        Prec = ar_prec_matrix(args.ar_coef, args.seq_length_out).float().to(device)

    for _ in range(args.iterations):
        optimiser.zero_grad()
        model.train()

        start_time = time.time()

        # ------------------------------------------------------- TRAINING
        inputs, outputs, c_ids = model.get_batch(train_iter)
        inputs, outputs = torchify(inputs, outputs, device=device)

        if is_MT:
            mu = model.mt_net.Z_mu[c_ids, :]
            sd = torch.sigmoid(3 * model.mt_net.Z_logit_s[c_ids, :])
            preds, _state = model(inputs, mu, sd)
        else:
            preds, _state = model(inputs)

        err = (preds - outputs)
        if has_weight:
            err = err * torch.cat((torch.ones(1, 1, 3) * np.sqrt(args.first3_prec),
                                   torch.ones(1, 1, args.human_size - 3)), dim=2).to(err.device)
        if not has_ar_noise:
            sqerr = err ** 2
        else:
            sqerr = (Prec @ err) * err

        step_loss = args.human_size * args.seq_length_out * sqerr.mean() / 2

        # assume \sigma is const. wrt optimisation, and hence normalising constant can be ignored.
        # Now for KL term. Since we're descending *negative* L.B., we need to *ADD* KL to loss:
        if is_MT:
            logstd = torch.log(sd)
            KLD = -0.5 * torch.sum(1 + 2 * logstd - mu.pow(2) - torch.exp(2 * logstd))
            step_loss = step_loss + KLD

        # Actual backpropagation
        step_loss.backward()
        optimiser.step()
        # -------------------------------------------------------

        # Reporting / admin
        step_loss = step_loss.cpu().data.numpy()

        if current_step % 10 == 0:
            if is_MT:
                KLD_part = KLD.cpu().data.numpy()
                print("step {0:04d}; step_loss: {1:.4f} ({2:.4f})".format(current_step, step_loss, step_loss-KLD_part))
            else:
                print("step {0:04d}; step_loss: {1:.4f}".format(current_step, step_loss))

        step_time += (time.time() - start_time) / args.test_every
        loss += step_loss / args.test_every
        current_step += 1

        if current_step % 20 == 0:
            sys.stdout.flush()

        # Decay learning rate (if appl.)
        if current_step % args.learning_rate_step == 0:
            for param_group in optimiser.param_groups:
                param_group['lr'] *= args.learning_rate_decay_factor
            print("Decay learning rate. New value at " + str(optimiser.param_groups[0]['lr']))

        # remove Hard EM spec (if appl.)
        if is_hard_em and zls_ix is not None and current_step == args.hard_em_iters:
            optimiser.param_groups[zls_ix]['lr'] = z_lr
            model.standardise_aggregate_posterior()

        # Once in a while, we save checkpoint, print statistics, and run evals.
        if current_step % args.test_every == 0:
            model.eval()

            # === CANNOT DO TEST SET EVALUATION SINCE DONT KNOW LATENT Z ===
            # inputs, outputs = model.get_test_batch(test_set_Y, test_set_U, -1)
            # inputs, outputs = torchify(inputs, outputs, device=device)
            #
            # if is_MT:
            #     preds, state = model(inputs, mu, sd)
            # else:
            #     preds = model(inputs)
            #
            # err = (preds - outputs)
            # if has_weight:
            #     err = err * torch.cat((torch.ones(1, 1, 3) * np.sqrt(args.first3_prec),
            #                            torch.ones(1, 1, args.human_size - 3)), dim=2).to(err.device)
            #
            # if not has_ar_noise:
            #     sqerr = err ** 2
            # else:
            #     Prec_test = ar_prec_matrix(args.ar_coef, err.size(1)).float().to(device)
            #     sqerr = (Prec_test @ err) * err
            #
            # val_loss = args.human_size * args.seq_length_out * sqerr.mean() / 2
            #
            # if is_MT:
            #     logstd = torch.log(sd)
            #     KLD = -0.5 * torch.sum(1 + 2 * logstd - mu.pow(2) - torch.exp(2 * logstd))
            #     val_loss = val_loss + KLD
            #
            # print()
            # print("{0: <16} |".format("milliseconds"), end="")
            # for ms in [60, 240, 480, 750, 990, 1500, 2010]:
            #     print(" {0:5d} |".format(ms), end="")
            # print()
            #
            # avg_mse_tt = sqerr.detach().cpu().mean(dim=0).numpy().mean(axis=1)

            # Pretty print of the results for 60, 240, 480, 750, 990, 1500, 2010 ms
            # print("{0: <16} |".format(" "), end="")
            # for ms in [1, 7, 15, 24, 32, 49, 66]:
            #     if args.seq_length_out >= ms + 1:
            #         print(" {0:.3f} |".format(avg_mse_tt[ms]), end="")
            #     else:
            #         print("   n/a |", end="")
            # print()
            #
            # print()
            # print("============================\n"
            #       "Global step:         %d\n"
            #       "Learning rate:       %.4f\n"
            #       "Step-time (ms):     %.4f\n"
            #       "Train loss avg:      %.4f\n"
            #       "--------------------------\n"
            #       "Test loss:            %.4f\n"
            #       "============================" % (current_step,
            #                                         args.learning_rate, step_time * 1000, loss,
            #                                         val_loss))

            torch.save(model, args.train_dir + '/model_' + str(current_step))

            # print()
            previous_losses.append(loss)

            # Reset global time and loss
            step_time, loss = 0, 0

            sys.stdout.flush()


def sample(args):
    raise NotImplementedError("Sampling not yet implemented: unsure how to deal with unknown latent z.")
    train_set_Y, train_set_U, test_set_Y, test_set_U = read_all_data(args)

    model = create_model(args)
    model.eval()
    if not args.use_cpu:
        model = model.cuda()
    print("Model created")

    inputs, outputs = model.get_test_batch(test_set_Y, test_set_U, -1)

    inputs = Variable(torch.from_numpy(inputs).float())
    outputs = Variable(torch.from_numpy(outputs).float())
    if not args.use_cpu:
        inputs, outputs, inputs.cuda(), outputs.cuda()

    if args.k > 0:
        preds, mu, logstd, state = model(inputs, outputs)
    else:
        preds = model(inputs)

    loss = (preds - outputs) ** 2
    loss.cpu().data.numpy()
    loss = loss.mean()

    preds = preds.cpu().data.numpy()
    preds = preds.transpose([1, 0, 2])

    loss = loss.cpu().data.numpy()

    np.savez("mt_predictions_{0}.npz".format(args.style_ix), preds=preds, actual=outputs)

    return


def ar_prec_matrix(rho, n):
    # Banded covariance construction
    Prec = np.zeros((n, n))
    i, j = np.indices(Prec.shape)
    Prec[i == j] = 1 + rho ** 2
    Prec[i == j - 1] = - rho
    Prec[i == j + 2] = - rho
    return torch.tensor(Prec)


def load_layer1(model, layer1_filename, use_cpu):
    model_gru1 = torch.load(layer1_filename, map_location='cpu') if use_cpu else torch.load(layer1_filename)
    if isinstance(model_gru1, mtfixb_model.OpenLoopGRU):
        model.layer1_rnn = model_gru1.rnn
        # model.layer1_linear = model_gru2.emission
    else:
        model.layer1_rnn = model_gru1.rnn2

    return model

def read_all_data(args):
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
    print("Reading training data (test index {0:d}).".format(args.style_ix))
    input_test_fname = args.input_test_fname
    if input_test_fname == "":
        input_test_fname = "test_input_{0}_u.npz".format(args.style_ix)

    njoints = args.human_size

    style_lkp = np.load(os.path.join(args.data_dir, args.stylelkp_fname))
    train_ixs = np.concatenate([style_lkp[str(i)] for i in range(1, len(style_lkp.keys()) + 1) if
                                i != args.style_ix])  # CAREFUL: jl is 1-based!
    train_set_Y = np.load(os.path.join(args.data_dir, args.output_fname))
    train_set_U = np.load(os.path.join(args.data_dir, args.input_fname))
    njoints = train_set_Y[str(0)].shape[1] if njoints <= 0 else njoints
    train_set_Y = [train_set_Y[str(i)][:, :njoints] for i in train_ixs]
    train_set_U = [train_set_U[str(i)] for i in train_ixs]

    # test_set_Y = np.load(os.path.join(args.data_dir, "test_input_{0}_y.npz".format(args.style_ix)))
    # test_set_U = np.load(os.path.join(args.data_dir, input_test_fname))
    # test_set_Y = [test_set_Y[str(i + 1)][:njoints, :] for i in
    #               range(len(test_set_Y.keys()))]  # whatever, apparently test is transpose of train
    # test_set_U = [test_set_U[str(i + 1)] for i in range(len(test_set_U.keys()))]

    print("done reading data.")

    return mtfixb_model.DataIterator(train_set_Y, train_set_U, 64, min_size=64, overlap2=args.overlap_windows)


def torchify(*args, device="cpu"):
    return [Variable(torch.from_numpy(arg).float()).to(device) for arg in args]


def main(args=None):
    args = parseopts.parse_args(args)
    args = parseopts.initial_arg_transform(args)

    print(args.train_dir)
    os.makedirs(args.train_dir, exist_ok=True)

    if args.sample:
        sample(args)
    else:
        train(args)

    return args


if __name__ == "__main__":
    main()
