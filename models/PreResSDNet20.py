# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 21:31:41 2018

@author: HITS
"""
from models.network_templates import ResNet
from functions.shake_drop import ShakeDrop

Ns = (3,) * 3
channels = (16, 32, 64)
firsts = 'CBR'
mains = 'I+BRCBRCS'
lasts = 'BRP'
nobias = False


def S(_self):
    return ShakeDrop(_self.depth_rate)


def model(classes):
    "Definition of 20-layer pre-activation shake-drop ResNets."
    return ResNet(classes, Ns, channels, firsts, mains, lasts, nobias=nobias,
                  S=S)
