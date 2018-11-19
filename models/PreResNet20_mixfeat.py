from functions.mixfeat import mixfeat
from models.network_templates import ResNet

Ns = (3,) * 3
channels = (16, 32, 64)
firsts = 'CXBR'
mains = 'I+BRCXBRCX'
lasts = 'BRP'
nobias = False


def X(module):
    return mixfeat


def model(classes):
    "Definition of 20-layer pre-activation ResNets with mixfeat."
    return ResNet(classes, Ns, channels, firsts, mains, lasts, nobias=nobias,
                  X=X)
