# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 20:19:18 2018

@author: HITS
"""

import six
import chainer
import chainer.links as L
import chainer.functions as F
from functions.exadd import exadd_maxshape


def force_tuple(src, n):
    if not hasattr(src, '__iter__'):
        src = (src,) * n
    else:
        diff = max(0, n - len(src))
        src = tuple(src[:n]) + (src[-1],) * diff
    return src


class LinkDict():
    "Link dictionary class."
    def preset(self):
        initialW = chainer.initializers.HeNormal()
        self.initialW = initialW
        dic = {}
        # preset keys (is to be overrided).
        # 3x3 Convolution
        # when you add 'link' to dic, use 'lambda _self: link(...)' form.
        dic['C'] = lambda _self: \
            L.Convolution2D(None, _self.ch, 3, _self.stride, 1, self.nobias,
                            initialW)
        # 1x1 Convolution
        dic['c'] = lambda _self: \
            L.Convolution2D(None, _self.ch, 1, _self.stride, 0, self.nobias,
                            initialW)
        # Batch Normalization
        dic['B'] = lambda _self: L.BatchNormalization(_self.ch)
        # Rectified Linear Unit
        # when you add 'func', that is not 'link', to dic,
        # use 'lambda _self: (lambda x: func(x, ...))' form.
        dic['R'] = lambda _self: (lambda x: F.relu(x))
        # Average pooling
        dic['A'] = lambda _self: (lambda x: F.average_pooling_2d(x, 2))
        # Max pooling
        dic['M'] = lambda _self: (lambda x: F.max_pooling_2d(x, 2))
        # Global average pooling
        dic['P'] = lambda _self: \
            (lambda x: F.average_pooling_2d(x, x.shape[2:4]))
        # Identity mapping (average pooling when stride > 1)
        dic['I'] = lambda _self: \
            (lambda x: x if _self.stride == 1 else
             lambda x: F.average_pooling_2d(x, 1, _self.stride))
        # Identity mapping (1x1 convolution when stride > 1)
        dic['i'] = lambda _self: \
            (lambda x: x if _self.stride == 1 else
             L.Convolution2D(None, _self.ch, 1, _self.stride, 0, self.nobias,
                             initialW))
        return dic

    def merge_dic(self, *dicts):
        return six.moves.reduce(lambda lhs, rhs: dict(lhs, **rhs), dicts)

    def __init__(self, in_channels, out_channels, keys, stride=1, nobias=False,
                 conv_keys='', depthrate=0, **dic):
        self.ch = in_channels
        # If out_channels is None, output keeps input channels.
        self.out_ch = out_channels or in_channels
        self.keys = keys
        self.stride = stride
        self.nobias = nobias
        self.conv_keys = conv_keys + 'Cci'
        self.depth_rate = depthrate
        self.dic = self.merge_dic(self.preset(), dic)

    def __iter__(self):
        factor = 1
        for k in self.keys:
            if k.isdigit():
                factor = int(k)
                continue
            if k in self.conv_keys:
                # 'out_channels' changes when convolution applying.
                self.ch = self.out_ch * factor
                # reset factor of channel
                factor = 1
            # create the 'link' instance (or 'lambda x' function).
            link = self.dic[k](self)
            if k in self.conv_keys:
                # Assigined 'stride' is applied only on the first convolution.
                self.stride = 1
            yield link
        raise StopIteration()


class SeriesLink(chainer.ChainList):
    "Series of link."
    def __init__(self, in_channels, out_channels, keys='BRCBRC', stride=1,
                 nobias=False, conv_keys='', depthrate=0, **dic):
        super(SeriesLink, self).__init__()
        dic = LinkDict(in_channels, out_channels, keys, stride, nobias,
                       conv_keys, depthrate, **dic)
        self.series = []
        for link in dic:
            self.series.append(link)
            # append only chainer.link.Link to self.
            if isinstance(link, chainer.link.Link):
                self.append(link)
        self.out_channels = dic.ch

    def __call__(self, x):
        return six.moves.reduce(lambda h, f: f(h), self.series, x)


class SumSeries(chainer.ChainList):
    "Sum of Series."
    def __init__(self, in_channels, outs_channels, keys='I+BRCBRC', stride=1,
                 nobias=False, conv_keys='', depthrate=0, **dic):
        super(SumSeries, self).__init__()
        if isinstance(keys, str):
            keys = keys.split('+')
        outs_channels = force_tuple(outs_channels, len(keys))
        for k, o in zip(keys, outs_channels):
            self.append(SeriesLink(in_channels, o, k, stride, nobias,
                                   conv_keys, depthrate, **dic))
        self.out_channels = max(branch.out_channels for branch in self)

    def __call__(self, x):
        return exadd_maxshape([f(x) for f in self])


class ConcatSeries(chainer.ChainList):
    "Concatenation of Series."
    def __init__(self, in_channels, outs_channels, keys='I,BRCBRC', stride=1,
                 nobias=False, conv_keys='', depthrate=0, **dic):
        super(ConcatSeries, self).__init__()
        if isinstance(keys, str):
            keys = keys.split(',')
        outs_channels = force_tuple(outs_channels, len(keys))
        for k, o in zip(keys, outs_channels):
            series = SumSeries if '+' in k else SeriesLink
            self.append(series(in_channels, o, k, stride, nobias, conv_keys,
                               depthrate, **dic))
        self.out_channels = sum(branch.out_channels for branch in self)

    def __call__(self, x):
        return F.concat([f(x) for f in self])


class SequentialChainList(chainer.ChainList):

    """Sequential executer of ChainList.

    Args:
        links: Initial child links.
    """

    def __init__(self, *links):
        super(SequentialChainList, self).__init__(*links)

    def __call__(self, x):
        return six.moves.reduce(lambda h, f: f(h), self, x)


class Module(SequentialChainList):
    """
    Sequence of Series.
    Priority of the joiner as shown: '+(sum)' > ',(concat)' > '>(sequence)'.

    Args:
        in_channels (int or None): Number of channels of input arrays.
            If ``None``, parameter initialization will be deferred until the
            first forward data pass at which time the size will be determined,
            if the 'chainer.link' of the first key supports lazy
            initialization.
        out_channels ((nest of) tuple of ints): Array of the Numbers of
            channels of the each output arrays of the branches.
            If ``None``, out_channels = in_channels.
            (Note that, it is forbidden both of them are ``None``.)
        keys (str): Array of keys,
            each key corresponds to definition of the layer.
            '>,+' are Special keys,
            '>' joins both sides of '>' sequencialy,
            ',' joins both sides by concatenation,
            '+' joins both sides by summation,
            and priority order of them is '+' > ',' > '>'.
        stride (int or pair of ints): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
            stride will be applied to the first branch.
            If the first branch is parallel,
            stride will be applied to each branch in parallel branches.
        nobias (bool): If ``True``,
            then links in this module does not use the bias term.
        conv_keys (str): The keys corresponds to the layers which includes
            convolution. This information is used to stride invalidation and
            out_channel updating in the first convolution layer.
        depthrate (float): The depth rate of current module in whole networks.
        print_debug (bool): Whether print or not the series of the module
            information. default is True.
        dic (dict as {char: chainer.link generator}):
            Dictionary of key to the layer definition.
            For example, if you need chainer.link,
                dic['B'] = lambda _self: L.BatchNormalization(_self.ch)
            if you need function which contains no updating parameter,
                dic['R'] = lambda _self: (lambda x: F.relu(x))
            '_self' has current status including
            in_channels, out_channels, stride, nobias, conv_keys.
    """
    def __init__(self, in_channels, outs_channels, keys='I+CBRCB>R', stride=1,
                 nobias=False, conv_keys='', depthrate=0,
                 print_debug=True, **dic):
        super(Module, self).__init__()
        if isinstance(keys, str):
            keys = keys.split('>')
        outs_channels = force_tuple(outs_channels, len(keys))
        for k, o in zip(keys, outs_channels):
            if print_debug:
                print('{:.4f}'.format(depthrate), in_channels, k, o)
            if ',' in k:
                series = ConcatSeries
            elif '+' in k:
                series = SumSeries
            else:
                series = SeriesLink
            self.append(series(in_channels, o, k, stride, nobias, conv_keys,
                               depthrate, **dic))
            stride = 1
            in_channels = self[-1].out_channels
        self.out_channels = in_channels
