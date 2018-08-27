from chainer.backends import cuda
from chainer import function

from chainer import config


class Perturb(function.Function):
    """Perturbation layer."""
    def __init__(self, level=0.1, pass_inference=False):
        self.level = 0.1
        self.noise = None
        self.pass_inference = pass_inference

    def make_noise(self, x):
        xp = cuda.get_array_module(x)
        _, c, h, w = x.shape
        return xp.random.uniform(
            -self.level, self.level, (1, c, h, w)).astype(xp.float32)

    def forward(self, xs):
        x = xs[0]
        self.retain_inputs(())
        if self.noise is None:
            self.noise = self.make_noise(x)
        if config.train or not self.pass_inference:
            x = x + self.noise
        return x,

    def backward(self, xs, gys):
        return gys
