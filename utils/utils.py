# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 22:27:20 2018

@author: HITS
"""


def attention_shape(axes, shape):
    return [L if i in axes else 1 for i, L in enumerate(shape)]
