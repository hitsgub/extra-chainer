import chainer.datasets as D


def num_of_class(dataset):
    "Get # of classes on the given dataset."
    dic = {'cifar10': 10, 'cifar100': 100, 'SVHN': 10}
    return dic[dataset]


def get_dataset(dataset):
    "Get dataset."
    if dataset == 'cifar10':
        return D.get_cifar10(scale=255.)
    if dataset == 'cifar100':
        return D.get_cifar100(scale=255.)
    if dataset == 'SVHN':
        return D.get_svhn(scale=255.)
    raise RuntimeError('Invalid dataset.')
