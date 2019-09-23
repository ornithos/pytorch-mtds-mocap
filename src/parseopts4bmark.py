import os
import argparse
import configparser

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Train RNN for human pose estimation')
    parser.add_argument('--style_ix', dest='style_ix',
                        help='Style index to hold out', type=int, required=True)
    parser.add_argument('--weight_decay', dest='weight_decay', help='regularisation', default=0.0, type=float)
    parser.add_argument('--optimiser', dest='optimiser', help='optimisation algorithm', default='SGD', type=str)
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
    parser.add_argument('--train_set_size', dest='train_set_size', type=int, default=-1,
                        help='Number of training instances (length 64) per style')
    parser.add_argument('--stl', dest='stl', help='Single Task Learning: Learn only a model for chosen style',
                        action='store_true')

    # Architecture
    parser.add_argument('--architecture', dest='architecture',
                        help='Seq2seq architecture to use: [basic, tied].',
                        default='tied', type=str)
    parser.add_argument('--loss_to_use', dest='loss_to_use',
                        help='The type of loss to use, supervised or sampling_based',
                        default='sampling_based', type=str)
    parser.add_argument('--residual_velocities', dest='residual_velocities',
                        help='Add a residual connection that effectively models velocities', action='store_true',
                        default=True)
    parser.add_argument('--size', dest='size',
                        help='Size of each model layer.',
                        default=1024, type=int)
    parser.add_argument('--num_layers', dest='num_layers',
                        help='Number of layers in the model.',
                        default=1, type=int)
    parser.add_argument('--open_loop', dest='open_loop',
                        help='Training style: open loop (sometimes called "teacher forcing") or closed loop (' +
                        'related to scheduled sampling).', action='store_true')
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
                        default=os.path.normpath("../../mocap-mtds/"), type=str)
                        # default=os.path.normpath("../../mocap-mtds/data/"), type=str)
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
                        default='', type=str)
    parser.add_argument('--sample', dest='sample',
                        help='Set to True for sampling.', action='store_true',
                        default=False)
    parser.add_argument('--stylelkp_fname', dest='stylelkp_fname', type=str,
                        help="name of style_lkp file", default="styles_lkp.npz")

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)


    # relevant argument checks
    assert args.omit_one_hot, "not implemented yet"
    assert args.action == "walking", "not implemented yet"
    assert args.residual_velocities, "not implemented yet. (Also not in original fork.)"
    assert (not args.stl) or args.style_ix in range(1, 9), "style_ix must be in 1:8 if running STL."

    return args


def initial_arg_transform(args):
    if not os.path.isfile(os.path.join(args.data_dir, "styles_lkp.npz")):
        print("Moving datadir from {:s} => ../../mocap-mtds/data/".format(args.data_dir))
        args.data_dir = os.path.normpath("../../mocap-mtds/data/")

    if args.train_set_size == -1:
        args.input_fname = "edin_Us_30fps_final.npz"
        args.output_fname = "edin_Ys_30fps_final.npz"
    elif args.train_set_size == 0:
        args.input_fname = "edin_Us_30fps_variableN_test_complement_stitched.npz"
        args.output_fname = "edin_Ys_30fps_variableN_test_complement_stitched.npz"
    else:
        args.input_fname = "edin_Us_30fps_N{:d}.npz".format(args.train_set_size)
        args.output_fname = "edin_Ys_30fps_N{:d}.npz".format(args.train_set_size)

    args.train_dir = get_model_save_dir(args)

    return args



def get_model_save_dir(args):
    return os.path.normpath(os.path.join( args.train_dir, args.action,
                  'style_{0}{1:s}'.format(args.style_ix, "" if not args.stl else "/stl"),
                  'out_{0}'.format(args.seq_length_out),
                  'iterations_{0}'.format(args.iterations),
                  'optimiser_{0}'.format(args.optimiser),
                  'weightdecay_{0}'.format(args.weight_decay),
                  'omit_one_hot' if args.omit_one_hot else 'one_hot',
                  'depth_{0}'.format(args.num_layers),
                  'size_{0}'.format(args.size),
                  'lr_{0}'.format(args.learning_rate),
                  'residual_vel' if args.residual_velocities else 'not_residual_vel',
                  'train_{0}'.format('all' if args.train_set_size == -1 else args.train_set_size),
                  'open_loop' if not args.open_loop else 'closed_loop')    # note labels are wrong way in code!
                  )
