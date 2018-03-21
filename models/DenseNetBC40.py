# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 22:25:26 2018

@author: HITS
"""
from models.network_templates import DenseNet
import chainer.links as L

Ns = (6,) * 3
channels = 12
first_channels = channels * 2
firsts = 'CBR'
mains = 'I,BRXYRC'
lasts = 'BRP'
keys_join = 'BRcA'
trans_theta = 0.5
nobias = True
# Bottleneck convolution
X = lambda _self: L.Convolution2D(None, _self.ch * 4, 1, 1, 0, nobias=nobias)
# Bottleneck batch normalization
Y = lambda _self: L.BatchNormalization(_self.ch * 4)


def model(classes):
    "Definition of 40-layer DenseNets."
    return DenseNet(classes, Ns, first_channels, channels, firsts, mains,
                    lasts, keys_join, trans_theta, nobias=nobias, X=X, Y=Y)
