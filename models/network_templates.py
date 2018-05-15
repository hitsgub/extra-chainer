import chainer.links as L

from links.chain_modules import Module, SequentialChainList
from links.network_modules import DynamicChannels, DynamicRatioChannels
from links.network_modules import Encoder


class ResNet(SequentialChainList):
    """
    Template implementation example of residual networks.
    """
    def __init__(self, classes, Ns=(3,) * 3, channels=(16, 32, 64),
                 firsts='CBR', mains='I+BRCBRC', lasts='BRP',
                 keys_join='A', rule_channels_join=None, strides=(1, 1, 1),
                 nobias=False, conv_keys='', **dic):
        super(ResNet, self).__init__()
        # Module before main.
        self.append(Module(None, channels[0], firsts, 1, True, conv_keys,
                           **dic))
        # main networks.
        self.append(Encoder(Ns, self[-1].out_channels, channels, mains,
                            keys_join, rule_channels_join, strides, nobias,
                            conv_keys, **dic))
        # Module after main.
        self.append(Module(self[-1].out_channels, None, lasts, 1, nobias,
                           conv_keys, **dic))
        # Classifier.
        self.append(L.Linear(classes))


class DenseNet(SequentialChainList):
    """
    Template implementation example of densely connected networks.
    """
    def __init__(self, classes, Ns=(12,) * 3, first_channels=16, channels=12,
                 firsts='CBR', mains='I,BRC', lasts='BRP',
                 keys_join='BRcA', trans_theta=1, strides=(1, 1, 1),
                 nobias=True, conv_keys='', **dic):
        super(DenseNet, self).__init__()
        rule_channels_join = DynamicChannels(trans_theta)
        # Module before main.
        self.append(Module(None, first_channels, firsts, 1, True, conv_keys,
                           **dic))
        # main networks.
        self.append(Encoder(Ns, self[-1].out_channels, channels, mains,
                            keys_join, rule_channels_join, strides, nobias,
                            conv_keys, **dic))
        # Module after main.
        self.append(Module(self[-1].out_channels, None, lasts, 1, nobias,
                           conv_keys, **dic))
        # Classifier.
        self.append(L.Linear(classes))


class PyramidNet(SequentialChainList):
    """
    Template implementation example of pyramidal residual networks.
    """
    def __init__(self, classes, Ns=(18,) * 3, first_channels=16, alpha=48,
                 firsts='CBR', mains='I+BCBRCB', lasts='BRP',
                 keys_join='A', rule_channels_join=None, strides=(1, 1, 1),
                 nobias=False, conv_keys='', **dic):
        super(PyramidNet, self).__init__()
        rule_channels = DynamicRatioChannels(alpha, sum(Ns))
        # Module before main.
        self.append(Module(None, first_channels, firsts, 1, True, conv_keys,
                           **dic))
        # main networks.
        self.append(Encoder(Ns, self[-1].out_channels, rule_channels, mains,
                            keys_join, rule_channels_join, strides, nobias,
                            conv_keys, **dic))
        # Module after main.
        self.append(Module(self[-1].out_channels, None, lasts, 1, nobias,
                           conv_keys, **dic))
        # Classifier.
        self.append(L.Linear(classes))
