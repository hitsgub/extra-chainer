def num_of_class(dataset):
    "Get # of classes on the given dataset."
    dic = {'cifar10': 10, 'cifar100': 100, 'SVHN': 10}
    return dic[dataset]
