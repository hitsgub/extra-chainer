from models.network_templates import ResNet

Ns = (2,) * 4
channels = (64, 128, 256, 512)
firsts = 'CBR'
mains = 'I+BRCBRC'
lasts = 'BRP'
nobias = False
strides = [1] * 4


def model(classes):
    "Definition of pre-activation ResNets."
    return ResNet(classes, Ns, channels, firsts, mains, lasts,
                  strides=strides, nobias=nobias)
