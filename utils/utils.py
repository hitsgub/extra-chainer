def attention_shape(axes, shape):
    return [L if i in axes else 1 for i, L in enumerate(shape)]


def tf2bool(tf):
    "Convert string 't(True)' or 'f(False)' to bool."
    return tf == 't'
