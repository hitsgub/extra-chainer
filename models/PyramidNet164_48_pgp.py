# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 22:25:26 2018

@author: HITS
"""
from models.network_templates import PyramidNet
from functions.pgp import PGP

Ns = (18,) * 3
first_channels = 16
alpha = 48
firsts = 'CBR'
mains = 'I+BRcBRCBR4c'
lasts = 'BRP'
keys_join = 'G'
nobias = True


def G(_self):
    return PGP(2)


def model(classes):
    "Definition of 164-layer PyramidNets with PGP."
    return PyramidNet(classes, Ns, first_channels, alpha, firsts, mains,
                      lasts, keys_join, nobias=nobias, G=G)
