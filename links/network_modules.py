# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 06:15:49 2018

@author: HITS
"""

import six
from links.chain_modules import Module, SequentialChainList


class DepthRate(object):
    "get current depth rate in whole networks."
    def __init__(self, total, _range=(0, 1)):
        self.begin, end = _range
        self.sub = float(end - self.begin)
        self.total = total
        self.offset = 0

    def __call__(self, depth=0):
        return self.sub * (depth + self.offset) / self.total + self.begin

    def add_offset(self, pos):
        self.offset += pos


def force_tuple(src, n):
    if not hasattr(src, '__iter__'):
        src = (src,) * n
    else:
        diff = max(0, n - len(src))
        src = tuple(src[:n]) + (src[-1],) * diff
    return src


class RuleChannels(object):
    pass


class StaticChannels(RuleChannels):
    """
    getter of static number of out_channels.

    Args:
        out_channels (int): # of output channels in all layers.
    """
    def __init__(self, out_channels):
        self.out_channels = out_channels

    def __call__(self, in_channels):
        return self.out_channels


class DynamicChannels(RuleChannels):
    """
    getter of dynamic number of out_channels.
    out = int(in * scale) + bias.

    Args:
        scale (float): incremental scale between in- and out- channels.
        bias (int): additional number between in- and out- channels.
    """
    def __init__(self, scale=1, bias=0):
        self.scale = scale
        self.bias = bias

    def __call__(self, in_channels):
        return int(in_channels * self.scale) + self.bias


class DynamicRatioChannels(RuleChannels):
    """
    getter of dynamic number of out_channels, for pyramid-nets.
    out = int(init + alpha * k / N)

    Args:
        alpha (float): additional channels ratio.
        N (float): block size.
    """
    def __init__(self, alpha=48, N=9):
        self.alpha = float(alpha)
        self.N = N
        self.k = 0

    def __call__(self, in_channels):
        self.k += 1
        if not hasattr(self, 'in_channels'):
            self.in_channels = in_channels
        return int(self.in_channels + self.alpha * self.k / self.N)


class Block(SequentialChainList):
    "N iteration of Module."
    def __init__(self, N, in_channels, rule_channels, keys='I+BRCBRC',
                 stride=1, nobias=False, conv_keys='',
                 dr=DepthRate(1), **dic):
        super(Block, self).__init__()
        if not isinstance(rule_channels, RuleChannels):
            rule_channels = StaticChannels(rule_channels)
        for i in six.moves.range(N):
            out_channels = rule_channels(in_channels)
            self.append(Module(in_channels, out_channels, keys, stride, nobias,
                               conv_keys, dr(i), **dic))
            stride = 1
            in_channels = self[-1].out_channels
        self.out_channels = in_channels


class Encoder(SequentialChainList):
    """Encoder, consist of multiple ``Block``."""
    def __init__(self, Ns, in_channels, rules_channels, keys='I+BRCBRC',
                 keys_join='', rule_channels_join=None, strides=(1, 2, 2),
                 nobias=False, conv_keys='', **dic):
        super(Encoder, self).__init__()
        rules_channels = force_tuple(rules_channels, len(Ns))
        if not isinstance(rule_channels_join, RuleChannels):
            rule_channels_join = StaticChannels(rule_channels_join)
        dr = DepthRate(sum(Ns))
        for N, rule, stride in zip(Ns, rules_channels, strides):
            if len(self) > 0:
                out_channels = rule_channels_join(in_channels)
                self.append(Module(in_channels, out_channels, keys_join, 1,
                                   nobias, conv_keys, dr(0), **dic))
                in_channels = self[-1].out_channels
            self.append(Block(N, in_channels, rule, keys, stride, nobias,
                              conv_keys, dr, **dic))
            in_channels = self[-1].out_channels
            dr.add_offset(N)
        self.out_channels = in_channels
