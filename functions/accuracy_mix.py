# -*- coding: utf-8 -*-
from chainer import config
import chainer.functions as F


def accuracy_mix(y, t, ignore_label=None, force=False):
    if config.train or force:
        t = F.argmax(t, axis=1)
    return F.accuracy(y, t, ignore_label)
