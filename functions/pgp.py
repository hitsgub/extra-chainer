from chainer.backends import cuda
from chainer import function


def zero_padding(x, pad, axis):
    "zero padding."
    shape = list(x.shape)
    shape[axis] = pad
    xp = cuda.get_array_module(x)
    x_pad = xp.zeros(shape, dtype=x.dtype)
    return xp.concatenate((x_pad, x), axis=axis)


def pgp_forward(x, s):
    "parallel grid pooling."
    for i, p in enumerate(x.shape[2:], 2):
        if p % s:
            x = zero_padding(x, p % s, i)
    n_in, c, h_in, w_in = x.shape
    n_out = n_in * (s ** 2)
    h_out, w_out = h_in // s, w_in // s
    y = x.reshape(n_in, c, h_out, s, w_out, s)
    y = y.transpose(3, 5, 0, 1, 2, 4)
    return y.reshape(n_out, c, h_out, w_out)


def pgp_backward(x, s):
    "parallel grid pooling, inverse."
    n_in, c, h_in, w_in = x.shape
    n_out = n_in // (s ** 2)
    h_out, w_out = h_in * s, w_in * s
    y = x.reshape(s, s, n_out, c, h_in, w_in)
    y = y.transpose(2, 3, 4, 0, 5, 1)
    return y.reshape(n_out, c, h_out, w_out)


class PGP(function.Function):
    """PGP function."""
    def __init__(self, stride=2):
        self.stride = stride

    def forward(self, xs):
        self.retain_inputs(())
        x = xs[0]
        return pgp_forward(x, self.stride),

    def backward(self, xs, gys):
        gy = gys[0]
        return pgp_backward(gy, self.stride),


class PGP_inv(function.Function):
    """Inverse PGP function."""
    def __init__(self, stride=2):
        self.stride = stride

    def forward(self, xs):
        self.retain_inputs(())
        x = xs[0]
        return pgp_backward(x, self.stride),

    def backward(self, xs, gys):
        gy = gys[0]
        return pgp_forward(gy, self.stride),
