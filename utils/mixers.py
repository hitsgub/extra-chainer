import numpy


def mix_labels(a, b, ratio=0., classes=1):
    eye = numpy.eye(classes)
    labels = (eye[a] * (1 - ratio) + eye[b] * ratio).astype(numpy.float32)
    return labels


def mix_mean(a, b, ratio=0.):
    return (a * (1 - ratio) + b * ratio).astype(numpy.float32)


def mix_plus(a, b, ratio=0.):
    ga = numpy.std(a)
    gb = numpy.std(b)
    if ratio == 0:
        return a
    p = 1.0 / (1 + gb / ga * (1 - ratio) / ratio)
    denomi = numpy.sqrt(p ** 2 + (1 - p) ** 2)
    y = ((a * (1 - p) + b * p) / denomi).astype(numpy.float32)
    return y


def mix_stack(a, b, ratio=0.):
    a = a * (1 - ratio)
    b = b * ratio
    if numpy.random.binomial(1, 0.5):
        a, b = b, a
    y = numpy.concatenate((a, b), axis=0)
    return y
