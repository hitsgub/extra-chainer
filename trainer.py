# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 10:08:02 2018

@author: HITS
"""
import chainer
import chainer.datasets as D
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.backends import cuda
import numpy as np
import get_argments
from utils.model_info import str_info
from datasets.transformer import Transformer
from models.get_model import get_model


def get_dataset(dataset):
    "Get dataset."
    if dataset == 'cifar10':
        sets = D.get_cifar10(scale=255.)
    elif dataset == 'cifar100':
        sets = D.get_cifar100(scale=255.)
    elif dataset == 'SVHN':
        sets = D.get_svhn(scale=255.)
    else:
        raise RuntimeError('Invalid dataset.')
    return sets


def main(args):
    # print learning settings.
    print('GPU: {}'.format(args.gpu))
    print('# Mini-batch size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('Using {} dataset.'.format(args.dataset))
    print('')
    # Load datasets.
    trainset, testset = get_dataset(args.dataset)
    # Data transfomer
    transformer = Transformer(trainset,
                              normalize=args.normalize, trans=args.augment)
    # Make transform datasets.
    trainset = D.TransformDataset(trainset, transformer.train)
    testset = D.TransformDataset(testset, transformer.test)
    # Set CNN model.
    model = L.Classifier(get_model(args.model, args.classes))
    # Setup GPU
    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
        xp = cuda.cupy
    else:
        xp = np
    # Run to get model information.
    model.predictor(xp.array(trainset[0][:1]))
    print(str_info(model))
    # Set optimizer
    optimizer = chainer.optimizers.NesterovAG(args.lr)
    optimizer.setup(model)
    if args.weight_decay != 0:
        optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
    # Setup dataset iterators.
    train_iter = chainer.iterators.SerialIterator(trainset, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(testset, args.batchsize,
                                                 False, False)
    # Setup trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.outdir)
    # Set extension: Evaluater
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu),
                   name='val')
    # Set extension: learning rate decay
    points = [args.epoch // 2, args.epoch * 3 // 4]
    trigger = training.triggers.ManualScheduleTrigger(points, 'epoch')
    trainer.extend(extensions.ExponentialShift('lr', 0.1), trigger=trigger)
    # Set extension: Dump graph
    trainer.extend(extensions.dump_graph('main/loss'))
    # Set extension: Snapshot
    trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))
    # Set extension: log
    trainer.extend(extensions.LogReport())
    # Set extension: observe_lr
    trainer.extend(extensions.observe_lr())
    # Set extension: Print Report
    trainer.extend(extensions.PrintReport(
        ['epoch', 'lr', 'main/loss', 'val/main/loss',
         'main/accuracy', 'val/main/accuracy', 'elapsed_time']))
    # Set extension: Progress bar
    trainer.extend(extensions.ProgressBar(update_interval=args.interval))
    # Set extension: Save train curve graph.
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'val/main/loss'],
        x_key='epoch', file_name='loss.png', marker=''))
    trainer.extend(extensions.PlotReport(
        ['main/accuracy', 'val/main/accuracy'],
        x_key='epoch', file_name='accuracy.png', marker=''))
    # run
    trainer.run()


if __name__ == '__main__':
    args = get_argments.get_arguments()
    main(args)
