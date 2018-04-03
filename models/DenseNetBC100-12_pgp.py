# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 22:25:26 2018

@author: HITS
"""
from models.network_templates import DenseNet
from functions.pgp import PGP

Ns = (16,) * 3
channels = 12
first_channels = channels * 2
firsts = 'CBR'
mains = 'I,BR4cBRC'
lasts = 'BRP'
keys_join = 'BRcG'
trans_theta = 0.5
nobias = True


def G(_self):
    return PGP(2)


def model(classes):
    "Definition of 100-layer DenseNets with PGP."
    return DenseNet(classes, Ns, first_channels, channels, firsts, mains,
                    lasts, keys_join, trans_theta, nobias=nobias, G=G)