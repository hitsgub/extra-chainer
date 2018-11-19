import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer import config


def _check_indices(indices):
    if len(indices) == 0:
        return
    # TODO: Chack indices without cpu
    indices = cuda.to_cpu(indices)
    for i in indices:
        if 0 <= i < len(indices):
            continue
        raise ValueError('Out of bounds index: {}'.format(i))
    sort = numpy.sort(indices)
    for s, t in six.moves.zip(sort, sort[1:]):
        if s == t:
            raise ValueError('indices contains duplicate value: {}'.format(s))


def _inverse_indices(indices):
    xp = cuda.get_array_module(indices)
    r = xp.empty_like(indices)
    if xp is numpy:
        r[indices] = numpy.arange(len(indices))
    else:
        cuda.elementwise(
            'S ind', 'raw S r',
            'r[ind] = i',
            'inverse_indices'
        )(indices, r)
    return r


class MixFeat(function_node.FunctionNode):

    """MixFeat function."""

    def __init__(self, sigma, inv=False, inds=None, ra1=None, rb=None):
        self.sigma = sigma
        self.inv = inv
        self.inds = inds
        self.ra1 = ra1
        self.rb = rb

    def _perturber(self, x, indices, randoms, inv):
        if inv:
            indices = _inverse_indices(indices)
            y = x * randoms
        else:
            y = x
        y = y[indices]
        if inv:
            return y
        return y * randoms

    def forward(self, inputs):
        self.retain_inputs(())
        x, = inputs
        xp = cuda.get_array_module(x)
        if self.inds is None:
            self.inds = xp.random.permutation(x.shape[0])
        if self.ra1 is None:
            shape = [s if i == 0 else 1 for i, s in enumerate(x.shape)]
            rate = xp.random.normal(0, self.sigma, shape, dtype=xp.float32)
            theta = xp.random.uniform(-numpy.pi, numpy.pi, shape,
                                      dtype=xp.float32)
            self.ra1 = 1 + rate * xp.sin(theta)
            self.rb = rate * xp.cos(theta)

        if chainer.is_debug():
            _check_indices(self.inds)

        return x * self.ra1 + self._perturber(x, self.inds, self.rb, self.inv),

    def backward(self, indexes, grad_outputs):
        g, = grad_outputs
        gx, = MixFeat(
            self.sigma, not self.inv, self.inds, self.ra1, self.rb).apply((g,))
        return gx,


def mixfeat(x, sigma=0.2):
    """Mix features a given variable along the axis.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Variable to mix features.
            A :math:`(s_1, s_2, ..., s_N)` -shaped float array.
        sigma (float): Gaussian parameter :math:`\\sigma` for mixing ratio.

    Returns:
        ~chainer.Variable: Output variable.
    """
    if not config.train:
        return x
    y, = MixFeat(sigma).apply((x,))
    return y
