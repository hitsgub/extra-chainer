# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 08:00:56 2018

@author: HITS
"""

import six
import chainer 
import chainer.links as L
import chainer.functions as F
from chainer import config
import numpy as np

class SeparableLink(chainer.ChainList):
    def __init__(self, link=L.Convolution2D, axis=1, n=2, *args, **kwargs):
        super(SeparableLink, self).__init__()
        for _ in six.moves.range(n):
            self.append(link(*args, **kwargs))
        self.axis = axis

    def __call__(self, x):
        # computating dividing point on the axis.
        divides = np.linspace(0, x.shape[self.axis], len(self) + 1)
        # computating divided slices.
        slices = (slice(i, j) for i, j in zip(divides, divides[1:]))
        slice_base = (slice(None),) * self.axis
        # applying each link on the each divided slice, and concatenation.
        y = F.concat((f(x[slice_base + (s,)]) for f, s in zip(self, slices)), self.axis)
        return y

class SeparableSampleLink(chainer.ChainList):
    def __init__(self, link=L.Convolution2D, _reduce=sum, _normalize=True,
                 n=2, *args, **kwargs):
        super(SeparableSampleLink, self).__init__()
        for _ in six.moves.range(n):
            self.append(link(*args, **kwargs))
        self._reduce = _reduce
        self._normalize = _normalize

    def __call__(self, x):
        if config.train:
            # computating dividing point on axis=0.
            divides = np.linspace(0, x.shape[0], len(self) + 1)
            # computating divided slices.
            slices = (slice(i,j) for i, j in zip(divides, divides[1:]))
            # applying each link on the each divided slice, and concatenation.
            y = F.concat((f(x[s]) for f, s in zip(self, slices)), 0)
        else:
            # applying links on the batch, and reduce.
            y = self._reduce(f(x) for f in self)
            # normalizing if it is needed.
            if self._normalize:
                y /= len(self)
        return y
