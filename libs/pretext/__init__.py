from .cifar10 import DataPreprocess as cifar10
from .mlcc import DataPreprocess as mlcc


def get_data_preprocess(args):
    if args.dataset == 'mlcc':
        return mlcc
    elif args.dataset == 'cifar10':
        return cifar10
