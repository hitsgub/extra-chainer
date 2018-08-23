import argparse
from datasets import dataset
from utils import io
from utils import utils


def get_arguments(header=''):
    "Argument parser."
    parser = argparse.ArgumentParser(description='Chainer CIFAR example:')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Set GPU device number. Default is 0.')
    parser.add_argument('--augment', type=str, default='t', choices='tf',
                        help='Whether to use data augmentation. '
                        "Default is 't'.")
    parser.add_argument('--normalize', type=str, default='t', choices='tf',
                        help='Whether to use data normalization. '
                        "Default is 't'.")
    parser.add_argument('--batchsize', type=int, default=128,
                        help='Mini-batch size. Default is 128.')
    parser.add_argument('--epoch', type=int, default=300,
                        help='Number of sweeps over the dataset to train.'
                        'Default is 300.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Initial learning rate. Default is 0.05.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay parameter. Default is 5e-4.')
    parser.add_argument('--model', type=str, default='models/ResNet20.py',
                        help='Path of NN model. '
                        "Default is 'models/ResNet20.py'.")
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory to output the results.'
                        "Default is 'results'.")
    parser.add_argument('--logname', type=str, default='log.txt',
                        help="File name to output log. Default is 'log.txt'.")
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'SVHN'],
                        help="Dataset name. Default is 'cifar10'.")
    parser.add_argument('--resume', type=str, default='',
                        help='Snapshot path to resume the training.'
                        "Default is ''.")
    parser.add_argument('--interval', type=int, default=10,
                        help='Update interval of progress bar. Default is 10.')
    parser.add_argument('--slack_interval', type=int, default=0,
                        help='Interval of slack log. If 0, not log. '
                        'Default is 0.')
    args = parser.parse_args()

    args.augment = utils.tf2bool(args.augment)
    args.normalize = utils.tf2bool(args.normalize)
    header = '{}{}'.format(header, args.dataset)
    args.outdir = io.create_result_dir(args.outdir, args.model, header)
    args.classes = dataset.num_of_class(args.dataset)

    io.create_log(args.outdir, args.logname)

    return args
