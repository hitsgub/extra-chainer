from functions.pgp import PGP
from models.network_templates import PyramidNet

Ns = (18,) * 3
first_channels = 16
alpha = 48
firsts = 'CBR'
mains = 'I+BRcBRCBR4c'
lasts = 'BRP'
keys_join = 'G'
nobias = True


def G(_self):
    return PGP(2)


def model(classes):
    "Definition of 164-layer PyramidNets with PGP."
    return PyramidNet(classes, Ns, first_channels, alpha, firsts, mains,
                      lasts, keys_join, nobias=nobias, G=G)
