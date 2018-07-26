import chainer
import chainer.functions as F

from functions.exadd import exadd_maxshape
from links.chain_modules import Module
from models.network_templates import ResNet

Ns = (3,) * 3
channels = (16, 32, 64)
firsts = 'CBR'
mains = 'Xi'
lasts = 'BRP'
nobias = False
se_ratio = 16


def S(_self):
    return (lambda x: F.sigmoid(x))


class CSE_block(chainer.Chain):
    "competitive squeeze-and-excitation block"
    def __init__(self, in_channels, out_channels, keys='BRCBRC', stride=1,
                 nobias=False, conv_keys='', depthrate=0, se_ratio=16, **dic):
        super(CSE_block, self).__init__()
        se_channels = out_channels // se_ratio
        with self.init_scope():
            self.res = Module(in_channels, out_channels, keys, stride,
                              nobias, conv_keys, depthrate, **dic)
            self.ide = Module(in_channels, out_channels, 'I', stride,
                              nobias, conv_keys, depthrate, **dic)
            self.rSE = Module(out_channels, se_channels, 'PcR', 1,
                              nobias, conv_keys, depthrate, **dic)
            self.iSE = Module(in_channels, se_channels, 'PcR', 1,
                              nobias, conv_keys, depthrate, **dic)
            self.SE = Module(se_channels * 2, out_channels, 'cS', 1,
                             nobias, conv_keys, depthrate, S=S, **dic)
        self.out_channels = out_channels
        self.channels = [m.channels for m in
                         (self.res, self.ide, self.rSE, self.iSE, self.SE)]

    def __call__(self, x):
        res = self.res(x)
        ide = self.ide(x)
        se_res = self.rSE(res)
        se_ide = self.iSE(ide)
        se = F.concat((se_res, se_ide), axis=1)
        excite = self.SE(se)
        return exadd_maxshape([res * F.broadcast_to(excite, res.shape), ide])


def X(_self):
    return CSE_block(_self.ch, _self.out_ch, 'BRCBRC', _self.stride,
                     _self.nobias, _self.conv_keys, _self.depth_rate, se_ratio,
                     **_self.dic)


def model(classes):
    "Definition of 20-layer pre-activation ResNets."
    return ResNet(classes, Ns, channels, firsts, mains, lasts, nobias=nobias,
                  X=X)
