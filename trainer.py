import chainer
from chainer.backends import cuda
import chainer.datasets as D
from chainer import training
from chainer.training import extensions

import numpy as np

from datasets.transformer import Transformer
import get_argments
from links.multiplex_classifier import MultiplexClassifier
from models.get_model import get_model
from training.extensions.print_stack_report import PrintStackReport
from utils.model_info import str_info
from utils.slack import SlackOut


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


def postprocess_loss(fig, axes, obj):
    "Modify graph of loss."
    axes.set_xlim(xmin=0)
    axes.set_yscale('log')
    axes.grid(which='both')
    return


def postprocess_accuracy(fig, axes, obj):
    "Modify graph of accuracy."
    axes.set_xlim(xmin=0)
    axes.set_ylim(ymax=1)
    axes.grid(which='both')
    return


def setup_trainer(args, train_iter, test_iter, model):
    # Set optimizer
    optimizer = chainer.optimizers.NesterovAG(args.lr)
    optimizer.setup(model)
    if args.weight_decay != 0:
        optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
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
    trainer.extend(PrintStackReport(
        ['epoch', 'lr', 'main/loss', 'val/main/loss',
         'main/accuracy', 'val/main/accuracy', 'elapsed_time']))
    if args.slack_interval:
        # Set extension: Print Report
        trainer.extend(PrintStackReport(
            ['epoch', 'lr', 'main/loss', 'val/main/loss',
             'main/accuracy', 'val/main/accuracy', 'elapsed_time'],
            out=SlackOut()), trigger=(args.slack_interval, 'epoch'))
    # Set extension: Progress bar
    trainer.extend(extensions.ProgressBar(update_interval=args.interval))
    # Set extension: Save train curve graph.
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'val/main/loss'], x_key='epoch',
        postprocess=postprocess_loss, file_name='loss.png',
        marker='', grid=False))
    trainer.extend(extensions.PlotReport(
        ['main/accuracy', 'val/main/accuracy'], x_key='epoch',
        postprocess=postprocess_accuracy, file_name='accuracy.png',
        marker='', grid=False))
    return trainer


def one_predict(model, iterator, device):
    xp = cuda.cupy if device >= 0 else np
    batch = iterator.next()
    model.predictor(xp.array([x for x, t in batch]))
    iterator.reset()


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
    transformer = Transformer(trainset, pca=False,
                              normalize=args.normalize, trans=args.augment)
    # Make transform datasets.
    trainset = D.TransformDataset(trainset, transformer.train)
    testset = D.TransformDataset(testset, transformer.test)
    # Setup dataset iterators.
    train_iter = chainer.iterators.SerialIterator(trainset, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(testset, args.batchsize,
                                                 False, False)
    # Set CNN model.
    model = MultiplexClassifier(get_model(args.model, args.classes))
    # Setup GPU
    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    # setup trainer
    trainer = setup_trainer(args, train_iter, test_iter, model)
    # Run to get model information.
    one_predict(model, train_iter, args.gpu)
    print(str_info(model))
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
    # run
    trainer.run()


if __name__ == '__main__':
    args = get_argments.get_arguments()
    main(args)
