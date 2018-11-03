from models.network_templates import DenseNet
import numpy as np

Ns = (16,) * 3
channels = 12
first_channels = channels * 2
firsts = 'CBR'
mains = 'I,XBR4cBRC'
lasts = 'BRP'
keys_join = 'BRcA'
trans_theta = 0.5
nobias = True
ch_biases = [12, 96, 138]

def X(module):
    ch0 = module.ch
    ch_bias = ch_biases[int(module.depth_rate * 3)]
    ch_base = ch0 - ch_bias
    ch = module.out_ch
    k = (ch_base // ch).bit_length()
    indexes = [np.arange(ch * (2**n - 1), ch * 2**n) for n in np.arange(k)]
    indexes = (ch0 - 1 - np.concatenate(indexes))[::-1]
    if indexes[0] == ch_bias:
        indexes = np.concatenate((np.arange(ch_bias), indexes))
    module.ch = len(indexes)
    def func(x):
        return x[:, indexes]
    return func


def model(classes):
    "Definition of 100-layer SparseNets."
    return DenseNet(classes, Ns, first_channels, channels, firsts, mains,
                    lasts, keys_join, trans_theta, nobias=nobias, X=X)
