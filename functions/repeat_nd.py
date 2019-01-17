import six

import chainer.functions as F
from chainer import function
from chainer.utils import type_check


class RepeatND(function.Function):
    """Repeating of an array in ND."""

    def __init__(self, reps):
        if isinstance(reps, six.integer_types):
            self.reps = (reps,)
        elif isinstance(reps, tuple) and all(
                isinstance(x, six.integer_types) for x in reps):
            self.reps = reps
        else:
            raise TypeError('reps must be int or tuple of ints')
        if not all(x >= 0 for x in self.reps):
            raise ValueError('all elements in reps must be zero or larger')

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

    def forward(self, inputs):
        self.retain_inputs(())
        x = inputs[0]
        self.ndim = x.ndim
        self.shape = x.shape
        for i, r in enumerate(self.reps):
            x = x.repeat(r, axis=i)
        return x,

    def backward(self, inputs, grads):
        reps = self.reps

        # Ensure input and reps have the same length.
        if self.ndim > len(reps):
            reps += (1,) * (self.ndim - len(reps))

        if grads[0].shape == ():
            # This case should be treated differently because numpy.num would
            # return a scalar (even if keepdims=True).
            return grads[0],

        # Reshape so that base axis and reps axis can be distinguished.
        new_shape = []
        for i in range(grads[0].ndim):
            new_shape.append(self.shape[i])
            new_shape.append(reps[i])
        new_shape = tuple(new_shape)

        # Sum along reps axis
        reps_axis = tuple(range(1, 2 * grads[0].ndim, 2))
        gy = grads[0].reshape(new_shape).sum(axis=reps_axis)

        return gy,


def repeat_nd(x, reps):
    """Construct an array by repeating a given array.

    Args:
        x (chainer.Variable or :class:`numpy.ndarray` or cupy.ndarray):
            Input data.
        reps (int or tuple of ints): The number of times for each axis with
            which x is replicated.

    Returns:
        ~chainer.Variable: Variable repeated the given array.
    """
    return RepeatND(reps)(x)


def upsample(x, width):
    return RepeatND((1, 1, width, width))(x)


def upscale_old(x, width):
    return F.average_pooling_2d(upsample(x, width), width + 1, 1, width // 2)


def upscale(x, width):
    _, _, h, w = x.shape
    return F.average_pooling_2d(
        upsample(x, width), 2 * width, 1, width)[..., :width * h, :width * w]


def upscales(x, width, iteration=1):
    for _ in six.moves.range(iteration):
        x = upscale(x, width)
    return x
