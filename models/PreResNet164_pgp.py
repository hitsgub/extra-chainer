# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 21:31:41 2018

@author: HITS
"""
from models.network_templates import ResNet
from functions.pgp import PGP

Ns = (18,) * 3
channels = (16, 32, 64)
firsts = 'CBR'
mains = 'I+BRcBRCBR4c'
lasts = 'BRP'
keys_join = 'G'
nobias = False


def G(_self):
    return PGP(2)


def model(classes):
    "Definition of 164-layer bottleneck pre-activation ResNets with PGP."
    return ResNet(classes, Ns, channels, firsts, mains, lasts, keys_join,
                  nobias=nobias, G=G)
