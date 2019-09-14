import os, time
import argparse
import learn_mtfixbmodel
import parseopts
import torch
import torch.optim as optim
import numpy as np

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Optimise Z on new data')
    parser.add_argument('--style_ix', dest='style_ix',
                        help='Style index for optimisation (1,..,8).', type=int, required=True)
    parser.add_argument('--latent_k', dest='k',
                        help='Dimension of parameter manifold (choose 3, 5, 7).', type=int, required=True)
    parser.add_argument('--train_set_size', dest='train_set_size',
                        help='Size of training set that model was learned on (choose 4, 8, 16, 32, 64).', type=int)
    parser.add_argument('--test_set_size', dest='test_set_size',
                        help='Size of batch to perform optimisation over.', type=int, default=32)
    parser.add_argument('--B_forward', dest='B_forward',
                        help='number of batches forward from test index chosen. Since test batches are generally not'
                             'contiguous, the default `1` is recommended, although this then only performs density'
                             'estimation-ish.', type=int, default=1)
    parser.add_argument('--use_cpu', dest='use_cpu', help='', action='store_true')
    parser.add_argument('--data_dir', dest='data_dir', help='Data directory', type=str, default="../../mocap-mtds/data")
    parser.add_argument('--training_iters', dest='iternums', help='Iterations to train for.', type=int)
    parser.add_argument('--model_type', dest='model_type', help='`biasonly` or `no_mt_bias`.')
    parser.add_argument('--learning_rate', dest='learning_rate', help='Learning rate for Z optimisation', type=float,
                        default=8e-3)

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    return args


def optimise(args):
    # ---------------------------------------------------------------------------------------------------------------------
    # === Inputs =====================================
    style_ix = args.style_ix                 # 1, 2, 3, 4, 5, 6, 7, 8
    z_dim = args.k                           # 3, 5, 7
    train_set_size = args.train_set_size     # 4, 8, 16, 32, 64
    test_set_size = args.test_set_size       # 32 => cannot do 64 since some styles have less held-out data than this.
    B_forward = args.B_forward               # 1 => we can't really do LT prediction since test cases are not contiguous in general.
    device = "cpu" if args.use_cpu else "cuda"
    model_iternums = args.iternums           # 10000, 20000
    model_type = args.model_type             # "biasonly", "no_mt_bias"
    lr = args.learning_rate                  # 1e-2, 8e-3, 5e-3, 3e-3, 1e-3 => most reliable has been 8e-3 with poss. annealing.
    data_dir = args.data_dir
    # ---------------------------------------------------------------------------------------------------------------------

    # Input checks
    assert model_type in ["biasonly", "no_mt_bias"]
    assert device in ["cpu", "cuda"], "device must be 'cpu' or 'cuda'."
    assert train_set_size in [4, 8, 16, 32, 64]
    assert B_forward == 1, "cannot do LT prediction as not contiguous."
    assert z_dim in [3, 5, 7]

    # Input transformations
    iscpu = device == "cpu"
    biasonly = model_type == "biasonly"
    model_path = "experiments/style_9/out_64/iterations_20000/decoder_size_1024/" + "zdim_{:d}".format(z_dim) + \
        "/ar_coef_0/psi_lowrank_30/optim_Adam/lr_2e-05/std/" + "edin_Us_30fps_N{0:d}/edin_Ys_30fps_N{0:d}".format(
        train_set_size) + "/not_residual_vel/model_{:d}".format(model_iternums)
    # model_path = "../../mocap-mtds/experiments/nobias/style8_k7_40000"

    # Load model
    load_args = ["--style_ix", str(style_ix), "--load", model_path,
                 "--latent_k", str(z_dim), "--input_size", str(35)]
    iscpu and load_args.append("--use_cpu")

    load_args = parseopts.parse_args(load_args)
    load_args = parseopts.initial_arg_transform(load_args)
    model = learn_mtfixbmodel.create_model(load_args, 850)
    iscpu and model.cpu()

    # Set AD off for most parameters
    model.layer1_rnn.requires_grad = False
    model.layer1_linear.requires_grad = False
    model.mt_net.Z_logit_s.data = model.mt_net.Z_logit_s.data * 1e-7
    model.mt_net.Z_logit_s.requires_grad = False
    model.layer1_rnn.train()

    if biasonly:
        model.mt_net.rnn.requires_grad = False
        model.mt_net.emission.requires_grad = False
        model.mt_net.rnn.train()
    else:
        model.mt_net.psi_decoder.requires_grad = False

    # Get test data
    print("Reading test data (test index {0:d}).".format(style_ix))
    output_fname, input_fname, tstix_fname = map(lambda x: x + "_30fps_N{:d}.npz".format(train_set_size),
                                              ["edin_Ys", "edin_Us", "edin_ixs"])

    test_ixs_all = np.load(os.path.join(data_dir, tstix_fname))[str(style_ix)]

    test_set_Y = np.load(os.path.join(data_dir, output_fname))
    test_set_U = np.load(os.path.join(data_dir, input_fname))
    test_set_Y = [test_set_Y[str(i)] for i in test_ixs_all]
    test_set_U = [test_set_U[str(i)] for i in test_ixs_all]

    print("Using files {:s}; {:s}; {:s}".format(input_fname, output_fname, tstix_fname))
    print("done reading data.")

    # Determine which test examples we will use.
    _test_ixs = np.linspace(0, len(test_ixs_all)-1-B_forward, test_set_size).round().astype('int')
    test_ixs = test_ixs_all[_test_ixs]

    # Create inputs/outputs for optimisation
    ysz = test_set_Y[0].shape[0]
    usz = test_set_U[0].shape[0]
    Yb, Ub = torch.zeros(test_set_size, 64, ysz).float(), torch.zeros(test_set_size, 64, usz).float()
    for i in range(test_set_size):
        Ub[i, :, :] = test_set_U[test_ixs[i]]
        Yb[i, :, :] = test_set_Y[test_ixs[i]]
    if not iscpu:
        Ub.cuda()
        Yb.cuda()

    # Generate initial Z and set-up for optimisation.
    Z = torch.randn(test_set_size, model.k).to(device)
    Z.requires_grad = True
    sd = torch.ones_like(Z).float().to(device) * 1e-7
    pars = [{'params': [Z], 'lr': lr}]

    # Set-up optimiser
    if biasonly:
        iters = [1000, 1000, 1000]
    else:
        iters = [750, 500, 200]
    optimiser = optim.Adam(pars, betas=(0.9, 0.999), weight_decay=0)

    # Perform optimisation
    for j in range(len(iters)):
        start_time = time.time()
        for i in range(iters[j]):
            optimiser.zero_grad()

            preds, _state = model(Ub, Z, sd)
            err = (preds - Yb)
            sqerr = err.pow(2)
            step_loss = ysz * 64 * sqerr.mean() / 2

            # Actual backpropagation
            step_loss.backward()
            optimiser.step()
            i % 5 == 0 and print("step {:d}: {:02.3f}".format(i, step_loss.cpu().data.numpy()[1]))

        print("Inner loop {:d}/{:d} took {:03.1f} seconds".format(j, len(iters), time.time() - start_time))

        # Avoid local minima
        if j != len(iters) - 1:
            cross_errors = np.zeros(test_set_size, test_set_size)
            for i in range(test_set_size):
                _Z = Z[i, :].repeat(test_set_size, 1)   # duplicate i'th particle for all sequences.
                preds, _state = model(Ub, _Z, sd)
                err = (preds - Yb)
                sqerr = err.pow(2)
                cross_errors[:, i] = sqerr.mean(dim=2).mean(dim=1).cpu().detach().numpy()

            choose_z = np.argmin(cross_errors, axis=1)
            Z.data = Z[choose_z, :].detach()

    return Z.cpu.detach().numpy(), test_ixs


if __name__ == "__main__":
    args = parse_args()
    Z, test_ixs = optimise(args)
    savenm = "{:s}_N{:d}_k{:d}_i{:d}".format(args.model_type, args.train_set_size, args.latent_k, args.style_ix)
    np.savez(os.path.join(args.data_dir, "optim_Z", savenm), Z)