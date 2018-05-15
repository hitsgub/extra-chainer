from chainercv import transforms
from functools import partialmethod
import numpy as np


class Transformer(object):
    """Dataset transformer."""
    def __init__(self, trainset=None, pca=True, normalize=True, trans=True):
        normalize = normalize and (trainset is not None)
        trainimg = [x for x, _ in trainset]
        self.mean = np.mean(trainimg, axis=(0, 2, 3), keepdims=True)[0] \
            if normalize else 0
        self.invstd = 1 / np.std(trainimg, axis=(0, 2, 3), keepdims=True)[0] \
            if normalize else 1
        self.pca = pca
        self.trans = trans

    def __call__(self, inputs, train=True):
        img, label = inputs
        img = img.copy()
        # Color augmentation
        if train and self.pca:
            img = transforms.pca_lighting(img, 76.5)
        # Standardization
        img -= self.mean
        img *= self.invstd
        # Random crop
        if train and self.trans:
            img = transforms.random_flip(img, x_random=True)
            img = transforms.random_expand(img, max_ratio=1.5)
            img = transforms.random_crop(img, (28, 28))
        return img, label

    # partial methods
    train = partialmethod(__call__, train=True)
    test = partialmethod(__call__, train=False)
