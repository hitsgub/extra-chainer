from models.network_templates import ResNet

Ns = (3,) * 3
channels = (16, 32, 64)
firsts = 'CBR'
mains = 'I+BRcBRCBR4c'
lasts = 'BRP'
nobias = False


def model(classes):
    "Definition of 29-layer bottleneck pre-activation ResNets."
    return ResNet(classes, Ns, channels, firsts, mains, lasts, nobias=nobias)
