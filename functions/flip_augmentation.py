from chainer.backends import cuda
from chainer import function


class FlipAugmentation(function.Function):
    """intermediate augmentation by flipping."""
    def __init__(self, axes=(3,)):
        self.axes = axes

    def forward(self, xs):
        self.retain_inputs(())
        x = xs[0]
        xp = cuda.get_array_module(x)
        for i in self.axes:
            slices = [slice(None)] * i + [slice(None, None, -1)]
            y = xp.concatenate((x, x[slices]), axis=0)
        return y,

    def backward(self, xs, gys):
        gx = gys[0]
        for i in reversed(self.axes):
            N = gx.shape[0] // 2
            gx0 = gx[:N]
            slices = [slice(N, None)] + [slice(None)] * (i - 1) + \
                [slice(None, None, -1)]
            gx1 = gx[slices]
            gx = gx0 + gx1
        return gx,
