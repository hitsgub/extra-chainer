from chainer import functions as F
from functions.perturb import Perturb
from models.network_templates import ResNet

Ns = (3,) * 3
channels = (16, 32, 64)
firsts = 'CBR'
mains = 'I+aBpRcaBpRc'
lasts = 'BRP'
nobias = False


def a(_self):
    return lambda x: F.average_pooling_2d(x, 3, 1, 1)


def p(_self):
    return Perturb(0.1, True)


def model(classes):
    "Definition of 20-layer pnn-pre-activation ResNets."
    return ResNet(classes, Ns, channels, firsts, mains, lasts, nobias=nobias,
                  a=a, p=p)
