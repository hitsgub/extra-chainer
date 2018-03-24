# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 21:23:40 2018

@author: HITS
"""
from chainer import function
from chainer import config
import numpy as np
from chainer.backends import cuda
from utils.utils import attention_shape


class ShakeDrop(function.Function):
    """shake drop regularization."""
    def __init__(self, depth_rate, axes=(0, 1, 2, 3), pL=0.5,
                 a_range=(-1, 1), b_range=(0, 1)):
        self.axes = axes
        self.pl = 1 - depth_rate * (1 - pL)
        self.a_range = a_range
        self.b_range = b_range
        self.E = self.pl + (1 - self.pl) * np.mean(self.a_range)
        self.gate = 1

    def forward(self, xs):
        x = xs[0]
        xp = cuda.get_array_module(x)
        self.retain_inputs(())
        if config.train:
            self.gate = np.random.binomial(1, self.pl)
            if not self.gate:
                self.shape = attention_shape(self.axes, x.shape)
                alpha = xp.random.uniform(
                    *self.a_range, self.shape).astype(np.float32)
                y = x * alpha
            else:
                y = x
        else:
            y = x * self.E
        return y,

    def backward(self, xs, gys):
        gy = gys[0]
        xp = cuda.get_array_module(gy)
        if not self.gate:
            beta = xp.random.uniform(
                *self.b_range, self.shape).astype(np.float32)
            gx = gy * beta
        else:
            gx = gy
        return gx,
