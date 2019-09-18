import os, sys, time
import argparse

import learn_mtfixbmodel
import mtfixb_model
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
    parser.add_argument('--train_set_size', dest='train_set_size', default=-1,
                        help='Size of training set that model was learned on (choose 4, 8, 16, 32, 64).', type=int)
    parser.add_argument('--test_set_size', dest='test_set_size',
                        help='Size of batch to perform optimisation over.', type=int, default=32)
    parser.add_argument('--B_forward', dest='B_forward',
                        help='number of batches forward from test index chosen. Since test batches are generally not'
                             'contiguous, the default `1` is recommended, although this then only performs density'
                             'estimation-ish.', type=int, default=1)
    parser.add_argument('--use_cpu', dest='use_cpu', help='', action='store_true')
    parser.add_argument('--data_dir', dest='data_dir', help='Data directory', type=str, default="../../mocap-mtds/data")
    parser.add_argument('--training_iters', dest='iternums', help='Num Iterations model was trained for.', type=int)
    parser.add_argument('--model_type', dest='model_type', help='`biasonly` or `no_mt_bias`.')
    parser.add_argument('--learning_rate', dest='learning_rate', help='Learning rate for Z optimisation', type=float,
                        default=8e-3)
    parser.add_argument('--model_path', dest='model_path', help='["Advanced"] Script usually constructs the path for' +
                        'the model from your specification. Otherwise supply a custom path here', type=str, default='')
    parser.add_argument('--devmode', dest='devmode', help='Used for development on local machine: changes modelpath.',
                       action='store_true')
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
    assert train_set_size in [-1, 4, 8, 16, 32, 64]
    assert B_forward == 1, "cannot do LT prediction as not contiguous."
    assert z_dim in [3, 5, 7, 8]

    # Input transformations
    iscpu = device == "cpu"
    biasonly = model_type == "biasonly"
    is_mtl = train_set_size > 0
    model_iternums = 20000 if is_mtl else model_iternums

    # Construct model path
    if len(args.model_path) == 0:
        datafiles = "edin_Us_30fps_N{0:d}/edin_Ys_30fps_N{0:d}".format(train_set_size) if train_set_size > 0 else \
            "edin_Us_30fps_final/edin_Ys_30fps_final"
        model_path = "experiments/style_{:d}".format(9 if train_set_size > 0 else style_ix) + \
                     "/out_64/iterations_{:d}".format(model_iternums) + \
                     "/decoder_size_1024/zdim_{:d}".format(z_dim) + \
                     "/ar_coef_0/psi_lowrank_30/optim_Adam/lr_{:.0e}/std/".format(2e-5 if not biasonly else 5e-5) + \
                     datafiles + "/not_residual_vel/model_{:d}".format(model_iternums)
        if args.devmode:
            if model_type == "biasonly":
                model_path = "../../mocap-mtds/experiments/biasonly/style8_128_8e-4_k3_20000"
            else:
                model_path = "../../mocap-mtds/experiments/nobias/style8_k7_40000"

    print("model: {:s}".format(model_path))

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
    if is_mtl:
        input_fname_ = "edin_Us_30fps_variableN_test_seeds_{:d}.npz"
        output_fname_ = "edin_Ys_30fps_variableN_test_seeds_{:d}.npz"
        test_set_Y = [np.load(os.path.join(data_dir, output_fname_.format(i))) for i in range(1, 8+1)]
        test_set_Y = [npz[str(j)] for npz in test_set_Y for j in range(1, 4+1)]
        test_set_U = [np.load(os.path.join(data_dir, input_fname_.format(i))) for i in range(1, 8 + 1)]
        test_set_U = [npz[str(j)] for npz in test_set_U for j in range(1, 4 + 1)]

        # Create inputs/outputs for optimisation
        ysz = test_set_Y[0].shape[1]
        usz = test_set_U[0].shape[1]
        bsz = 4 * 8   # each style has 4 seed sequences.
        Yb, Ub = torch.zeros(bsz, 64, ysz).float(), torch.zeros(bsz, 64, usz).float()
        for i in range(bsz):
            Ub[i, :, :] = torch.from_numpy(test_set_U[i])
            Yb[i, :, :] = torch.from_numpy(test_set_Y[i])

        output_fname, input_fname = output_fname_.format(0), input_fname_.format(0)
        test_set_size = bsz
    else:
        output_fname, input_fname = "edin_Ys_30fps_final.npz", "edin_Us_30fps_final.npz"
        style_lkp = np.load(os.path.join(data_dir, "styles_lkp.npz"))
        test_set_Y = np.load(os.path.join(data_dir, output_fname))
        test_set_U = np.load(os.path.join(data_dir, input_fname))
        test_set_Y = [test_set_Y[str(i)] for i in style_lkp[str(style_ix)]]
        test_set_U = [test_set_U[str(i)] for i in style_lkp[str(style_ix)]]
        all_data = list(mtfixb_model.DataIterator(test_set_Y, test_set_U, 64, min_size=64, overlap2=False))
        test_set_Y = [all_data[i][0] for i in range(len(all_data))]
        test_set_U = [all_data[i][1] for i in range(len(all_data))]

        # Determine which test examples we will use.
        test_ixs = np.linspace(0, len(test_set_Y) - 1 - B_forward, test_set_size).round().astype('int')

        # Create inputs/outputs for optimisation
        ysz = test_set_Y[0].shape[1]
        usz = test_set_U[0].shape[1]
        Yb, Ub = torch.zeros(test_set_size, 64, ysz).float(), torch.zeros(test_set_size, 64, usz).float()
        for i in range(test_set_size):
            Ub[i, :, :] = torch.from_numpy(test_set_U[test_ixs[i]])
            Yb[i, :, :] = torch.from_numpy(test_set_Y[test_ixs[i]])

    print("Using files {:s}; {:s}".format(input_fname, output_fname))
    print("done reading data.")

    if not iscpu:
        Ub = Ub.cuda()
        Yb = Yb.cuda()

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
            i % 5 == 0 and print("step {:d}: {:02.3f}".format(i, step_loss.cpu().data.numpy())); sys.stdout.flush()

        print("Inner loop {:d}/{:d} took {:03.1f} seconds".format(j+1, len(iters), time.time() - start_time))

        # Avoid local minima
        if j != len(iters) - 1:
            cross_errors = np.zeros((test_set_size, test_set_size))
            for i in range(test_set_size):
                _Z = Z[i, :].repeat(test_set_size, 1)   # duplicate i'th particle for all sequences.
                preds, _state = model(Ub, _Z, sd)
                err = (preds - Yb)
                sqerr = err.pow(2)
                cross_errors[:, i] = sqerr.mean(dim=2).mean(dim=1).cpu().detach().numpy()

            choose_z = np.argmin(cross_errors, axis=1)
            Z.data = Z[choose_z, :].detach()

    if is_mtl:
        return Z.cpu().detach().numpy(), np.ones(1) * NaN
    else:
        return Z.cpu().detach().numpy(), test_ixs


if __name__ == "__main__":
    args = parse_args()
    Z, test_ixs = optimise(args)
    Ntype = "N{:d}".format(args.train_set_size) if args.train_set_size > 0 else "TL"
    savenm = "{:s}_{:s}_k{:d}_i{:d}".format(args.model_type, Ntype, args.k, args.style_ix)
    dir = os.path.join(args.data_dir, "optim_Z")
    os.makedirs(dir, exist_ok=True)
    np.savez(os.path.join(dir, savenm), Z)