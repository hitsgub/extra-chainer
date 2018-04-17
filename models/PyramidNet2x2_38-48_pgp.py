# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 18:25:58 2018

@author: HITS
"""
from models.network_templates import PyramidNet
from functions.pgp import PGP
import chainer.links as L

Ns = (6,) * 3
first_channels = 16
alpha = 48
firsts = 'CBR'
mains = 'I+BRcBRD'
lasts = 'BRP'
keys_join = 'G'
nobias = True
dic = {}

dic['D'] = lambda _self: \
    L.Convolution2D(None, _self.ch, 2, _self.stride, 1, _self.nobias,
                    _self.initialW)

dic['c'] = lambda _self: \
    L.Convolution2D(None, _self.ch, 2, _self.stride, 0, _self.nobias,
                    _self.initialW)

dic['G'] = lambda _self: \
    PGP(2)


def model(classes):
    """
    Definition of 38-48 PyramidNets with 2x2 convs.
    Test error in CIFAR-10 is around 5%,
    despite the model size is nearly equal to 20-layer-ResNet.
    """
    return PyramidNet(classes, Ns, first_channels, alpha, firsts, mains,
                      lasts, keys_join, nobias=nobias, conv_keys='D', **dic)
