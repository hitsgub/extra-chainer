from models.network_templates import PyramidNet

Ns = (18,) * 3
first_channels = 16
alpha = 48
firsts = 'CBR'
mains = 'I+BCBRCB'
lasts = 'BRP'
keys_join = 'A'
nobias = True


def model(classes):
    "Definition of 110-layer PyramidNets."
    return PyramidNet(classes, Ns, first_channels, alpha, firsts, mains,
                      lasts, keys_join, nobias=nobias)
