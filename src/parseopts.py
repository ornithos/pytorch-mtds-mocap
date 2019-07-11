import os
import argparse
import configparser


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Train MT-RNN for human pose estimation')
    parser.add_argument('--style_ix', dest='style_ix',
                        help='Style index to hold out', type=int, required=True)
    parser.add_argument('--latent_k', dest='k',
                        help='Dimension of parameter manifold.', type=int, required=True)
    parser.add_argument('--input_size', dest='input_size',
                        help='Input dimension at each timestep', required=True, type=int)

    parser.add_argument('--learning_rate', dest='learning_rate',
                        help='Learning rate', type=float)
    parser.add_argument('--learning_rate_decay_factor', dest='learning_rate_decay_factor',
                        help='Learning rate is multiplied by this much. 1 means no decay.', type=float)
    parser.add_argument('--learning_rate_step', dest='learning_rate_step',
                        help='Every this many steps, do decay.', type=int)
    parser.add_argument('--batch_size', dest='batch_size',
                        help='Batch size to use during training.', type=int)
    parser.add_argument('--iterations', dest='iterations',
                        help='Iterations to train for.', type=int)
    parser.add_argument('--test_every', dest='test_every',
                        help='how often to evaluate the test set (and save the model).', type=int)
    parser.add_argument('--optimiser', dest='optimiser',
                        help='Optimiser: SGD, Nesterov, or Adam', type=str)
    parser.add_argument('--first3_prec', dest='first3_prec',
                        help='Precision of noise model of first 3 outputs.', type=float)

    # Architecture
    parser.add_argument('--no_residual_velocities', dest='residual_velocities',
                        help='Add a residual connection that effectively models velocities', action='store_false')

    parser.add_argument('--decoder_size', dest='decoder_size',
                        help='Size of decoder recurrent state.', type=int)
    parser.add_argument('--encoder_size', dest='encoder_size',
                        help='Size of encoder recurrent state.', type=int)
    parser.add_argument('--size_psi_hidden', dest='size_psi_hidden',
                        help='Size of NL hidden layer of psi network.', type=int)
    parser.add_argument('--size_psi_lowrank', dest='size_psi_lowrank',
                        help='Subspace dimension to embed parameter manifold into. This is to reduce par count.',
                        type=int)
    parser.add_argument('--seq_length_out', dest='seq_length_out',
                        help='Number of frames that the decoder has to predict. 25fps', type=int)
    parser.add_argument('--human_size', dest='human_size',
                        help='Output dimension at each timestep', type=int)
    parser.add_argument('--dropout_p', dest='dropout_p',
                        help='Dropout probability for hidden layers', type=float)
    parser.add_argument('--weight_decay', dest='weight_decay',
                        help='Weight decay amount for regularisation', type=float)
    parser.add_argument('--ar_coef', dest='ar_coef',
                        help='Autoregressive coefficient (default is off)', type=float)
    parser.add_argument('--dynamicsdict', dest='dynamicsdict', action="store_true",
                        help='Dynamics Dictionary Architecture')

    # Directories
    parser.add_argument('--data_dir', dest='data_dir',
                        help='Data directory', type=str)
    parser.add_argument('--train_dir', dest='train_dir',
                        help='Training directory', type=str)
    parser.add_argument('--use_cpu', dest='use_cpu',
                        help='', action='store_true')
    parser.add_argument('--load', dest='load',
                        help='Try to load a previous checkpoint.', type=str)
    parser.add_argument('--sample', dest='sample',
                        help='Set to True for sampling.', action='store_true')
    parser.add_argument('--input_fname', dest='input_fname', type=str, help="name of input file")
    parser.add_argument('--output_fname', dest='output_fname', type=str, help="name of output file")
    parser.add_argument('--stylelkp_fname', dest='stylelkp_fname', type=str, help="name of style_lkp file")
    parser.add_argument('--data_augmentation', dest='data_augmentation', action='store_true')
    parser.add_argument('--input_test_fname', dest='input_test_fname', type=str, help="name of test input file")

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    # ========= Now get defaults from config file =============
    config = configparser.ConfigParser(allow_no_value=True)
    # config.optionxform = lambda option: option    # permit uppercase: i.e. no conversion in parsing
    config.read('default.ini')

    # flatten config dicts
    configdict = {}
    for sec in config.sections():
        configdict.update(dict(config[sec]))

    # convert config to correct types :/
    config_types = _get_types_from_argparse(parser)
    for key, value in configdict.items():
        configdict[key] = config_types[key](value)

    # update config with command line args
    vargs = vars(args)
    for key, value in vargs.items():
        if value is not None:
            configdict[key] = value

    configdict["data_dir"] = os.path.normpath(configdict["data_dir"])
    configdict["train_dir"] = os.path.normpath(configdict["train_dir"])

    return _dictNamespace(configdict)


def initial_arg_transform(args):
    if not os.path.isfile(os.path.join(args.data_dir, "styles_lkp.npz")):
        print("Moving datadir from {:s} => ../../mocap-mtds/data/".format(args.data_dir))
        args.data_dir = os.path.normpath("../../mocap-mtds/data/")

    if args.data_augmentation:
        def append_DA(x):
            base, ext = os.path.splitext(x)
            return base + "_DA" + ext

        args.input_fname = append_DA(args.input_fname)
        args.output_fname = append_DA(args.output_fname)
        args.stylelkp_fname = append_DA(args.stylelkp_fname)

    args.train_dir = get_model_save_dir(args)

    return args


def _get_types_from_argparse(parser):
    out = {}
    for a in parser._actions:
        ctype = _get_type_from_action(a)
        if ctype is not None:
            out[a.dest] = ctype
    return out


def _get_type_from_action(action):
    if isinstance(action, argparse._StoreAction):
        return action.type
    elif isinstance(action, argparse._StoreTrueAction) or isinstance(action, argparse._StoreFalseAction):
        return bool
    elif isinstance(action, argparse._HelpAction):
        return None
    else:
        raise Exception("Unexpected ArgParse action {:s}".format(type(action)))


def get_model_save_dir(args):
    return os.path.normpath(os.path.join(args.train_dir,
                                         'style_{0}'.format(args.style_ix),
                                         'out_{0}'.format(args.seq_length_out),
                                         'iterations_{0}'.format(args.iterations),
                                         'decoder_size_{0}'.format(args.decoder_size),
                                         'zdim_{0}'.format(args.k),
                                         'ar_coef_{:.0f}'.format(args.ar_coef * 1e3),
                                         'psi_lowrank_{0}'.format(args.size_psi_lowrank),
                                         'optim_{0}'.format(args.optimiser),
                                         'lr_{0}'.format(args.learning_rate),
                                         '{0}'.format("archDD" if args.dynamicsdict else "std"),
                                         '{0}'.format(args.input_fname.split(".")[0]),
                                         '{0}'.format(args.output_fname.split(".")[0]),
                                         'residual_vel' if args.residual_velocities else 'not_residual_vel'))


class _dictNamespace(object):
    """
    converts a dictionary into a namespace
    """

    def __init__(self, adict):
        self.__dict__.update(adict)
