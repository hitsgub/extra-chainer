# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 06:58:40 2018

@author: HITS
"""
import argparse
import logging
from pathlib import Path
import datetime


def tf2bool(tf):
    "Convert string 't(True)' or 'f(False)' to bool."
    return tf == 't'


def create_log(outdir, logname='log.txt'):
    "Create log file and set logging target to the file."
    fname = Path(outdir, logname)
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        filename=fname, level=logging.DEBUG)
    return


def create_result_dir(outdir, modelpath, header=''):
    "Create result directory using the current time to set the unique name."
    modelname = Path(modelpath).stem
    now = datetime.datetime.now()
    strnow = now.strftime('%y%m%d_%H%M%S_%f')
    result_dir = Path(outdir, '{}_{}_{}'.format(header, modelname, strnow))
    if not result_dir.exists():
        result_dir.mkdir(parents=True)
    return result_dir


def get_classes(dataset):
    "Get # of classes on the given dataset."
    dic = {'cifar10': 10, 'cifar100': 100, 'SVHN': 10}
    return dic[dataset]


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
    args = parser.parse_args()

    args.augment = tf2bool(args.augment)
    args.normalize = tf2bool(args.normalize)
    header = '{}{}'.format(header, args.dataset)
    args.outdir = create_result_dir(args.outdir, args.model, header)
    args.classes = get_classes(args.dataset)

    create_log(args.outdir, args.logname)

    return args
