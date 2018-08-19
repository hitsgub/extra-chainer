# -*- coding: utf-8 -*-
import numpy

from chainer.dataset import iterator
from chainer.iterators import SerialIterator
from functools import partial
from utils import mixers


class BCIterator(iterator.Iterator):

    """Dataset iterator that serially reads the examples.

    This is a simple implementation of :class:`~chainer.dataset.Iterator`
    that just visits each example in either the order of indexes or a shuffled
    order.

    To avoid unintentional performance degradation, the ``shuffle`` option is
    set to ``True`` by default. For validation, it is better to set it to
    ``False`` when the underlying dataset supports fast slicing. If the
    order of examples has an important meaning and the updater depends on the
    original order, this option should be set to ``False``.

    This iterator saves ``-1`` instead of ``None`` in snapshots since some
    serializers do not support ``None``.

    Args:
        dataset: Dataset to iterate.
        batch_size (int): Number of examples within each batch.
        repeat (bool): If ``True``, it infinitely loops over the dataset.
            Otherwise, it stops iteration at the end of the first epoch.
        shuffle (bool): If ``True``, the order of examples is shuffled at the
            beginning of each epoch. Otherwise, examples are extracted in the
            order of indexes. If ``None`` and no ``order_sampler`` is given,
            the behavior is the same as the case with ``shuffle=True``.
        order_sampler (callable): A callable that generates the order
            of the indices to sample in the next epoch when a epoch finishes.
            This function should take two arguements: the current order
            and the current position of the iterator.
            This should return the next order. The size of the order
            should remain constant.
            This option cannot be used when ``shuffle`` is not ``None``.

    """

    def __init__(self, dataset, batch_size,
                 repeat=True, shuffle=None, order_sampler=None, _range=0.5,
                 mixer_image=mixers.mix_plus, mixer_label=mixers.mix_labels,
                 classes=1):
        self.base_iter = SerialIterator(dataset, batch_size, repeat, shuffle,
                                        order_sampler)
        self.dataset = dataset
        self.mixer_image = mixer_image
        self.mixer_label = partial(mixer_label, classes=classes)
        self._range = _range

    def get_sub(self, label):
        while True:
            x, t = self.dataset[numpy.random.randint(len(self.dataset))]
            if t != label:
                return x, t

    def mix_sample(self, sample, r):
        xb, tb = sample
        xs, ts = self.get_sub(tb)
        return self.mixer_image(xb, xs, r), self.mixer_label(tb, ts, r)

    def __next__(self):
        base = self.base_iter.__next__()
        rands = numpy.random.uniform(0, self._range, len(base))
        batch = [self.mix_sample(sample, r) for sample, r in zip(base, rands)]

        return batch

    next = __next__

    @property
    def epoch_detail(self):
        return self.base_iter.epoch_detail

    @property
    def previous_epoch_detail(self):
        return self.base_iter.previous_epoch_detail

    def serialize(self, serializer):
        self.base_iter.serialize(serializer)

    def reset(self):
        self.base_iter.reset()

    @property
    def _epoch_size(self):
        return self.base_iter._epoch_size

    @property
    def epoch(self):
        return self.base_iter.epoch

    @property
    def is_new_epoch(self):
        return self.base_iter.is_new_epoch
