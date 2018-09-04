import chainer
import chainer.functions as F

from functions.shuffle import shuffle
from links.chain_modules import Module


class Shuffle_v2_block(chainer.Chain):
    "basic unit in shuffle-net v2."
    def __init__(self, in_channels, out_channels, conv_ratio=0.5,
                 keys='BRCBRC', stride=1, nobias=False,
                 conv_keys='', depth_ratio=0, **dic):
        super(Shuffle_v2_block, self).__init__()
        self.copy = (out_channels > in_channels)
        if self.copy:
            self.conv_in = in_channels
            conv_out = out_channels - in_channels
        else:
            self.conv_in = int(in_channels * conv_ratio)
            conv_out = out_channels - in_channels + self.conv_in
        with self.init_scope():
            self.convs = Module(self.conv_in, conv_out, keys, stride, nobias,
                                conv_keys, depth_ratio, **dic)

    def __call__(self, x):
        x, i = [x, x] if self.copy else \
            F.split_axis(x, (self.conv_in,), axis=1)
        x = self.convs(x)
        y = F.concat((x, i), axis=1)
        return shuffle(y)
