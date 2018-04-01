# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 07:57:24 2018

@author: HITS
"""
from chainer import link


def numParams(model):
    "Total number of parameters in the model."
    return sum(p.data.size for p in model.params())


def numWs(model):
    "Number of ``W`` in the model."
    n = 0
    for child in model.children():
        if hasattr(child, 'children'):
            n += numWs(child)
        if hasattr(child, 'W'):
            n += 1
    return n


def numLinks(model):
    "Number of ``chainer.link`` in the model."
    n = 0
    for child in model.children():
        if isinstance(child, link.Chain) or isinstance(child, link.ChainList):
            n += numLinks(child)
        else:
            n += 1
    return n


def str_info(model):
    "get information of the model."
    nParams = numParams(model)
    # Number which has ``W``, Conv or FC in default chainer.
    nLayers = numWs(model)
    nLinks = numLinks(model)
    return 'params:{:,}\tlayers:{}\tLinks:{}'.format(nParams, nLayers, nLinks)


def print_infos(models):
    "print information of the models."
    for name, m in models.items():
        s = str_info(m)
        strlog = '[{}]\t{}'.format(name, s)
        print(strlog)
