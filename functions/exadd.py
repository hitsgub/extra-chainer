from chainer.backends import cuda
from chainer import function


class ExAdd(function.Function):

    """
    Add given variables with extra zero padding or cutting excessive
    for mismatch dimensions.
    Output variable shape is given shape or same as xs[0].
    For example, it is able to used for merging branches with
    different channels in ResNetA.
    """

    def __init__(self, shape=None):
        self._shape = shape

    def forward(self, xs):
        xp = cuda.get_array_module(xs[0])
        self.retain_inputs(())
        self.shapes = (x.shape for x in xs)
        y = xp.zeros(self._shape, dtype=xs[0].dtype) if self._shape else 0
        for x in xs:
            if y is 0:
                y = x
            else:
                # Compute intersection between shape of x and y.
                slices = [slice(min(lx, ly)) for lx, ly in
                          zip(x.shape, y.shape)]
                y[slices] += x[slices]
        return y,

    def backward(self, xs, gys):
        gy = gys[0]
        gxs = []
        for xshape in self.shapes:
            if any(lx <= ly for lx, ly in zip(xshape, gy.shape)):
                # gx is subslices of gy.
                slices = [slice(lx) for lx in xshape]
                gx = gy[slices]
            else:
                xp = cuda.get_array_module(gy)
                gx = xp.zeros(xshape, dtype=gy.dtype)
                # Compute intersection between shape of x and y.
                slices = [slice(min(lx, ly)) for lx, ly in
                          zip(xshape, gy.shape)]
                gx[slices] = gy[slices]
            gxs.append(gx)
        return gxs


def exadd(xs, shape=None):
    """
    Add given variables with extra zero padding or cutting excessive
    on mismatch channels.

    Args:
        xs (tuple of Variables): Variables to be added.
            If shape is not given, first variable is basis.
        shape (tuple of ints, optional): output shape.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return ExAdd(shape)(*xs)


def exadd_maxshape(xs):
    """
    Add given variables with extra zero padding on mismatch channels,
    arrange to largest shape.

    Args:
        xs (tuple of Variables): Variables to be added.

    Returns:
        ~chainer.Variable: Output variable.
    """
    shapes = [x.shape for x in xs]
    shape = [max(ls) for ls in zip(*shapes)]
    return ExAdd(shape)(*xs)
