import chainer
import chainer.functions as F
import chainer.links as L

from functions.mixfeat import mixfeat
from functions.repeat_nd import upsample
from links.chain_modules import Module


class FishNet(chainer.Chain):
    """
    FishNet.
    """

    def __init__(self, classes=10, ch0=16,
                 firsts='CBR', mains='X+BRCBRC', bottoms='BRPc', lasts='BRP',
                 nobias=False, **dic):
        super(FishNet, self).__init__()
        with self.init_scope():
            self.f0 = Module(None, ch0, firsts, 1, nobias, **dic)
            # tail
            self.t0 = Module(ch0    , ch0    , mains, 1, nobias, **dic)
            self.t1 = Module(ch0    , ch0 * 2, mains, 1, nobias, **dic)
            self.t2 = Module(ch0 * 2, ch0 * 4, mains, 1, nobias, **dic)
            self.t3 = Module(ch0 * 4, ch0 * 4, bottoms, 1, nobias, **dic)
            # body
            self.b2 = Module(ch0 * 8, ch0 * 4, mains, 1, nobias, **dic)
            self.b1 = Module(ch0 * 6, ch0 * 3, mains, 1, nobias, **dic)
            self.b0 = Module(ch0 * 4, ch0 * 2, mains, 1, nobias, **dic)
            # head
            self.h1 = Module(ch0 * 5, ch0 * 5, mains, 1, nobias, **dic)
            self.h2 = Module(ch0 * 9, ch0 * 9, mains, 1, nobias, **dic)
            self.h3 = Module(ch0 * 9, ch0 * 9, lasts, 1, nobias, **dic)
            # linear
            self.fin = L.Linear(classes)

    def __call__(self, x):
        xs = []
        h = self.f0(x)
        # tail
        xs.append(self.t0(h))
        xs.append(self.t1(F.max_pooling_2d(xs[0], 2)))
        xs.append(self.t2(F.max_pooling_2d(xs[1], 2)))
        b = self.t3(xs[2])
        # body
        xs[2] = self.b2(F.concat([xs[2], F.broadcast_to(b, xs[2].shape)]))
        xs[1] = self.b1(F.concat([xs[1], upsample(xs[2], 2)]))
        xs[0] = self.b0(F.concat([xs[0], upsample(xs[1], 2)]))
        # head
        xs[1] = self.h1(F.concat([xs[1], F.max_pooling_2d(xs[0], 2)]))
        xs[2] = self.h2(F.concat([xs[2], F.max_pooling_2d(xs[1], 2)]))
        h = self.h3(xs[2])
        y = self.fin(h)
        return y


def E(module):
    return mixfeat


def X(module):
    if module.ch <= module.out_ch:
        return None
    def func(x):
        ch = x.shape[1] // 2
        return x[:, :ch] + x[:, ch:]
    return func


ch0 = 16
firsts = 'CEBR'
mains = 'X+BRCEBRCE'
bottoms = 'BRPcE'
lasts = 'BRP'
nobias = False


def model(classes):
    "Definition of X-layer FishNets."
    return FishNet(classes, ch0, firsts, mains, bottoms, lasts, nobias, X=X, E=E)
