# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 21:31:41 2018

@author: HITS
"""
from models.network_templates import ResNet

Ns = (3,) * 3
channels = (16, 32, 64)
firsts = 'CBR'
mains = 'I+CBRCB>R'
lasts = 'P'
nobias = False


def model(classes):
    "Definition of 20-layer ResNets."
    return ResNet(classes, Ns, channels, firsts, mains, lasts, nobias=nobias)
