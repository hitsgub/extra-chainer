from chainer.backends import cuda
from chainer import config
from chainer import function
import numpy as np

from utils.utils import attention_shape


class ShakeDrop(function.Function):
    """shake drop regularization."""
    def __init__(self, depth_rate, axes=(0, 1, 2, 3), pL=0.5,
                 a_range=(-1, 1), b_range=(0, 1)):
        self.axes = axes
        self.pl = 1 - depth_rate * (1 - pL)
        self.a_range = a_range
        self.b_range = b_range
        self.E = self.pl + (1 - self.pl) * np.mean(self.a_range)
        self.gate = 1

    def forward(self, xs):
        x = xs[0]
        self.retain_inputs(())
        if not config.train:
            return x * self.E,
        self.gate = np.random.binomial(1, self.pl)
        if self.gate:
            return x,
        self.shape = attention_shape(self.axes, x.shape)
        xp = cuda.get_array_module(x)
        alpha = xp.random.uniform(*self.a_range, self.shape).astype(np.float32)
        return x * alpha,

    def backward(self, xs, gys):
        gy = gys[0]
        xp = cuda.get_array_module(gy)
        if self.gate:
            return gy,
        beta = xp.random.uniform(*self.b_range, self.shape).astype(np.float32)
        return gy * beta,
