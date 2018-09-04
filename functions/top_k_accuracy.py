import six

from chainer.backends import cuda
from chainer import function
from chainer.utils import type_check


class TopKAccuracy(function.Function):

    def __init__(self, top_k=1, ignore_label=None):
        self.top_k = top_k
        self.ignore_label = ignore_label

    def check_type_forward(self, in_types):
        type_check.argname(in_types, ('x', 't'))
        x_type, t_type = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            t_type.dtype.kind == 'i'
        )

        t_ndim = type_check.eval(t_type.ndim)
        type_check.expect(
            x_type.ndim >= t_type.ndim,
            x_type.shape[0] == t_type.shape[0],
            x_type.shape[2: t_ndim + 1] == t_type.shape[1:]
        )
        for i in six.moves.range(t_ndim + 1, type_check.eval(x_type.ndim)):
            type_check.expect(x_type.shape[i] == 1)

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        y, t = inputs

        ignore_cnt = 0
        correct_cnt = 0
        if self.ignore_label is not None:
            mask = (t == self.ignore_label)
            ignore_cnt = mask.sum()
            preds = xp.argsort(y, axis=1)[:, :-1 - self.top_k:-1]
            preds = xp.where(mask[:, None], -1, preds)
        else:
            preds = xp.argsort(y, axis=1)[:, :-1 - self.top_k:-1]
        for i in six.moves.range(self.top_k):
            correct_cnt += (preds[:, i] == t).sum()
        total = t.size - ignore_cnt
        if total == 0:
            return xp.asarray(0.0, dtype=y.dtype),
        else:
            return xp.asarray(float(correct_cnt) / total, dtype=y.dtype),


def top_k_accuracy(y, t, top_k=1, ignore_label=None):
    """Computes multiclass classification top-K accuracy of the minibatch.

    Args:
        y (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Array whose (i, j, k, ...)-th element indicates the score of
            the class j at the (i, k, ...)-th sample.
            The ordered prediction labels :math:`\\hat ts` is calculated by
            the formula :math:`\\hat t(i, k, ...) = \
            \\operatorname{\\mathrm{argsort}}_j y(i, j, k, ...)`.
        t (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray` of signed integer):
            Array of ground truth labels.
        top_k (int): Calculating as correct
            if the true label is amongst this top ``top_k``.
        ignore_label (int or None): Skip calculating accuracy
            if the true label is ``ignore_label``.

    Returns:
        Variable: A variable holding a scalar array of the accuracy.

    .. note:: This function is non-differentiable.

    .. admonition:: Example

        We show the most common case, when ``y`` is the two dimensional array.

        >>> import numpy as np
        >>> y = np.array([[0.1, 0.7, 0.2], # prediction labels are [1, 2, 0]
        ...               [8.0, 1.0, 2.0], # prediction labels are [0, 2, 1]
        ...               [-8.0, -1.0, -2.0], # prediction labels are [1, 2, 0]
        ...               [-8.0, 1.0, 2.0]]) # prediction labels are [2, 1, 0]
        >>> t = np.array([1, 0, 1, 2], np.int32)
        >>> top_k_accuracy(y, t, 1).data \
# 100% accuracy because all samples are correct in 1st-label.
        array(1.)
        >>> t = np.array([1, 0, 2, 1], np.int32)
        >>> top_k_accuracy(y, t, 1).data \
# 50% accuracy because 1st and 2nd samples are correct in 1st-label.
        array(0.5)
        >>> top_k_accuracy(y, t, 2).data \
# 100% accuracy because all samples are correct in 1st or 2nd labels.
        array(1.)
        >>> top_k_accuracy(y, t, 1, ignore_label=1).data \
# 50% accuracy because of ignoring the 1st and 4th sample.
        array(0.5)
    """
    return TopKAccuracy(top_k=top_k, ignore_label=ignore_label)(y, t)
