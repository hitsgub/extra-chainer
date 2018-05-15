from chainer.backends import cuda
from chainer import function
from functions.pgp import pgp_forward, pgp_backward


class PGP_flip(function.Function):
    """PGP with flip."""
    def __init__(self, stride=2, axes=(3,)):
        self.stride = stride
        self.axes = axes

    def forward(self, xs):
        self.retain_inputs(())
        x = xs[0]
        xp = cuda.get_array_module(x)
        y = pgp_forward(x, self.stride)
        for i in self.axes:
            slices = [slice(None)] * i + [slice(None, None, -1)]
            y = xp.concatenate((y, y[slices]), axis=0)
        return y,

    def backward(self, xs, gys):
        gy = gys[0]
        for i in reversed(self.axes):
            N = gy.shape[0] // 2
            gy0 = gy[:N]
            slices = [slice(N, None)] + [slice(None)] * (i - 1) + \
                [slice(None, None, -1)]
            gy1 = gy[slices]
            gy = gy0 + gy1
        gx = pgp_backward(gy, self.stride)
        return gx,
