from functions.pgp import PGP
from models.network_templates import ResNet

Ns = (18,) * 3
channels = (16, 32, 64)
firsts = 'CBR'
mains = 'I+BRcBRCBR4c'
lasts = 'BRP'
keys_join = 'G'
nobias = False


def G(_self):
    return PGP(2)


def model(classes):
    "Definition of 164-layer bottleneck pre-activation ResNets with PGP."
    return ResNet(classes, Ns, channels, firsts, mains, lasts, keys_join,
                  nobias=nobias, G=G)
