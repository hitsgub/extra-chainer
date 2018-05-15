def attention_shape(axes, shape):
    return [L if i in axes else 1 for i, L in enumerate(shape)]
