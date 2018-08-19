from chainer import config
import chainer.functions as F


def kl_divergence(y, t, force=False):
    if not (config.train or force):
        return F.softmax_cross_entropy(y, t)
    ratios = t[t.nonzero()]
    entropy = - F.sum(ratios * F.log(ratios))
    crossEntropy = - F.sum(t * F.log_softmax(y))
    loss = (crossEntropy - entropy) / y.shape[0]
    return loss
