import chainer
from chainer.backends import cuda
import chainer.datasets as D

import chainer_computational_cost as c3

import numpy as np

from datasets.dataset import get_dataset
from datasets.transformer import Transformer
import get_argments
from links.multiplex_classifier import MultiplexClassifier
from models.get_model import get_model
from utils.model_info import str_info
from utils.setup_trainer import setup_trainer


def one_predict(model, iterator, device):
    xp = cuda.cupy if device >= 0 else np
    batch = iterator.next()
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        with c3.ComputationalCostHook(fma_1flop=True) as cch:
            model.predictor(xp.array([x for x, t in batch]))
            cch.show_summary_report(n_digits=3, mode='table')
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
