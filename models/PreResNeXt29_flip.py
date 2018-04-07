# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 21:31:41 2018

@author: HITS
"""
from models.network_templates import ResNet
import chainer.links as L
from functions.flip_augmentation import FlipAugmentation

group = 8
Ns = (3,) * 3
channels = (64, 128, 256)
firsts = 'CBR'
mains = 'I+BR{0}cBR{0}GBR4c'.format(group)
lasts = 'BRP'
keys_join = 'gA'
nobias = False


def G(_self):
    "Group Convolution."
    return L.Convolution2D(None, _self.ch, 3, _self.stride, 1, _self.nobias,
                           _self.initialW, group=group)


def g(_self):
    return FlipAugmentation()


def model(classes):
    "Definition of 29-layer pre-activation ResNeXts with flip."
    return ResNet(classes, Ns, channels, firsts, mains, lasts, keys_join,
                  nobias=nobias, conv_keys='G', G=G, g=g)
