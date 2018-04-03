# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 18:45:32 2018

@author: HITS
"""
from models.network_templates import ResNet
from functions.pgp import PGP

Ns = (3,) * 3
channels = (16, 32, 64)
firsts = 'CBR'
mains = 'I+BRCBRC'
lasts = 'BRP'
keys_join = 'G'
nobias = False


def G(_self):
    return PGP(2)


def model(classes):
    "Definition of 20-layer pre-activation ResNets."
    return ResNet(classes, Ns, channels, firsts, mains, lasts, keys_join,
                  nobias=nobias, G=G)
