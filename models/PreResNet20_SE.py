import chainer.functions as F
import chainer.links as L

from links.chain_modules import Module
from models.network_templates import ResNet

Ns = (3,) * 3
channels = (16, 32, 64)
firsts = 'CBR'
mains = 'I+BRCBRCE'
lasts = 'BRP'
nobias = False
se_ratio = 16


def S(_self):
    return (lambda x: F.sigmoid(x))


def s(_self):
    return L.Convolution2D(
        None, _self.ch // se_ratio, 1, 1, 0, _self.nobias, _self.initialW)


def E(_self):
    return Module(_self.ch, _self.ch, 'I*PsRcS', s=s, S=S)


def model(classes):
    "Definition of 20-layer pre-activation ResNets."
    return ResNet(classes, Ns, channels, firsts, mains, lasts, nobias=nobias,
                  E=E)
