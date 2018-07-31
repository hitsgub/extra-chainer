from links.shuffle_v2_block import Shuffle_v2_block
from models.network_templates import ResNet

Ns = (3,) * 3
channels = (16, 32, 64)
firsts = 'CBR'
mains = 'Xi'
lasts = 'BRCP'
nobias = False


def X(_self):
    return Shuffle_v2_block(
        _self.ch, _self.out_ch, 0.5, 'BRCBRC', _self.stride,
        _self.nobias, _self.conv_keys, _self.depth_rate, **_self.dic)


def model(classes):
    "Definition of 20-layer pre-activation ResNets."
    return ResNet(classes, Ns, channels, firsts, mains, lasts, nobias=nobias,
                  X=X)
