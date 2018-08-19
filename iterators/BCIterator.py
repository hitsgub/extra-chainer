import numpy

from chainer.iterators import SerialIterator
from functools import partial
from utils import mixers


class BCIterator(SerialIterator):

    """Dataset iterator that serially reads the examples and mix sub-examples.

    This is a simple implementation of :class:`~chainer.dataset.Iterator`
    that just visits each example in either the order of indexes or a shuffled
    order and mix sub-examples.

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
        mixer_image (callable): A callable that mix two images.
            This function should take three arguements: base-image, sub-image
            and ratio of mixing.
            This should return the mixed image.
        mixer_label (callable): A callable that mix two labels.
            This function should take four arguements: base-label, sub-label,
            ratio of mixing and number of classes.
            This should return the mixed label. The format of mixed label will
            not (int) and should match the format of args of ``lossfun`` and
            ``accfun`` of Classifier.
        force_2class (bool): If ``True``, sub-examples are extracted by
            iterative random-choice until sub-label and base-label
            are different. If ``False``, sub-examples are extracted from the
            ``SerialIterator``.
        _range (float): The max ratio of mixing sub-examples. If ``_range=0``,
            ``BCIterator`` can be used for non-mix iterator which return same
            format with mix iterator.
        classes (int): Number of classes. It is necessary for mixer_label.

    """

    def __init__(self, dataset, batch_size,
                 repeat=True, shuffle=None, order_sampler=None,
                 mixer_image=mixers.mix_plus, mixer_label=mixers.mix_labels,
                 force_2class=False, _range=0.5, classes=1):
        super(BCIterator, self).__init__(
            dataset, batch_size, repeat, shuffle, order_sampler)
        self.mixer_image = mixer_image
        self.mixer_label = partial(mixer_label, classes=classes)
        self._range = _range
        self.force_2class = force_2class
        if not force_2class:
            self.sub_iter = SerialIterator(
                dataset, batch_size, repeat, shuffle, order_sampler)

    def get_sub(self, label):
        while True:
            x, t = self.dataset[numpy.random.randint(self._epoch_size)]
            if t != label:
                return x, t

    def mix_sample(self, base, sub=None, r=0):
        xb, tb = base
        xs, ts = sub or self.get_sub(tb)
        return self.mixer_image(xb, xs, r), self.mixer_label(tb, ts, r)

    def __next__(self):
        bases = super(BCIterator, self).__next__()
        rands = numpy.random.uniform(0, self._range, self.batch_size)
        if self.force_2class:
            return [self.mix_sample(b, None, r) for b, r in zip(bases, rands)]
        subs = self.sub_iter.__next__()
        return [self.mix_sample(b, s, r) for b, s, r
                in zip(bases, subs, rands)]

    next = __next__
