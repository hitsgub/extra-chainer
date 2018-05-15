from functions.shake_drop import ShakeDrop
from models.network_templates import PyramidNet

Ns = (30,) * 3
first_channels = 16
alpha = 200
firsts = 'CBR'
mains = 'I+BcBRCBR4cBS'
lasts = 'BRP'
keys_join = 'A'
nobias = True


def S(_self):
    return ShakeDrop(_self.depth_rate)


def model(classes):
    "Definition of 110-layer PyramidNets."
    return PyramidNet(classes, Ns, first_channels, alpha, firsts, mains,
                      lasts, keys_join, nobias=nobias, S=S)
