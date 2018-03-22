# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 20:30:45 2018

@author: HITS
"""
from chainer import function
from chainer import config
import numpy as np
from chainer import cuda
import chainer.links as L
from models.network_templates import ResNet
from links.separable_link import SeparableLink


class ShakeShake(function.Function):
    """shake shake regularization."""
    def __init__(self, axes=(0, 1), a_range=(0, 1), b_range=(0, 1)):
        self.axes = axes
        self.a_range = a_range
        self.b_range = b_range
        self.E = np.mean(a_range)

    def forward(self, xs):
        x = xs[0]
        xp = cuda.get_array_module(x)
        self.retain_inputs(())
        half = x.shape[1] // 2
        if config.train:
            self.shape = [s if i in self.axes else 1 for i, s in
                          enumerate(x.shape)]
            if 1 in self.axes:
                self.shape[1] //= 2
            a = xp.random.uniform(*self.a_range, self.shape).astype(xp.float32)
            y = a * x[:, :half] + (1 - a) * x[:, half:]
        else:
            y = (x[:, :half] + x[:, half:]) * self.E
        return y,

    def backward(self, xs, gys):
        gy = gys[0]
        xp = cuda.get_array_module(gy)
        b = xp.random.uniform(*self.b_range, self.shape).astype(xp.float32)
        gx0 = gy * b
        gx1 = gy * (1 - b)
        return xp.concatenate((gx0, gx1), axis=1),


def shakeshake(x, axes=(0, 1), a_range=(0, 1), b_range=(0, 1)):
    return ShakeShake(axes, a_range, b_range)(x)


Ns = (3,) * 3
channels = (16, 32, 64)
firsts = 'CBR'
mains = 'I+BRWbRSs'
lasts = 'BRP'
nobias = False
# Twice channels convolution
W = lambda _self: L.Convolution2D(None, _self.ch * 2, 3, _self.stride, 1,
                                  nobias, _self.initialW)
# Twice channels Batch normalization
b = lambda _self: L.BatchNormalization(_self.ch * 2)
# Separable convolution
S = lambda _self: SeparableLink(L.Convolution2D, 1, 2, None,
                                _self.ch * 2, 3, _self.stride, 1,
                                nobias, _self.initialW)
# shakeshake
s = lambda _self: ShakeShake()


def model(classes):
    "Definition of 20-layer pre-activation ShakeShake ResNets."
    return ResNet(classes, Ns, channels, firsts, mains, lasts, nobias=nobias,
                  conv_keys='WS', W=W, b=b, S=S, s=s)
