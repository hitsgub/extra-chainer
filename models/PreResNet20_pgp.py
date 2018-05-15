from functions.pgp import PGP
from models.network_templates import ResNet

Ns = (3,) * 3
channels = (16, 32, 64)
firsts = 'CBR'
mains = 'I+BRCBRC'
lasts = 'BRP'
keys_join = 'G'
nobias = False


def G(_self):
    return PGP(2)


def model(classes):
    "Definition of 20-layer pre-activation ResNets with PGP."
    return ResNet(classes, Ns, channels, firsts, mains, lasts, keys_join,
                  nobias=nobias, G=G)
