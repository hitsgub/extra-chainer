from models.network_templates import ResNet

Ns = (3,) * 3
channels = (16, 32, 64)
firsts = 'CBR'
mains = 'I+BRCBRC'
lasts = 'BRP'
nobias = False


def model(classes):
    "Definition of 20-layer pre-activation ResNets."
    return ResNet(classes, Ns, channels, firsts, mains, lasts, nobias=nobias)
