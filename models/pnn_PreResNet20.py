from functions.perturb import Perturb
from models.network_templates import ResNet

Ns = (3,) * 3
channels = (16, 32, 64)
firsts = 'CBR'
mains = 'I+BpRcBpRc'
lasts = 'BRP'
nobias = False


def p(_self):
    return Perturb(0.1, True)


def model(classes):
    "Definition of 20-layer pnn-pre-activation ResNets."
    return ResNet(classes, Ns, channels, firsts, mains, lasts, nobias=nobias,
                  p=p)
