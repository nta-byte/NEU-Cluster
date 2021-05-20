from .neu_cls import Relabel as neu_cls
from .cifar10 import Relabel as cifar10
from .stl10 import Relabel as stl10
from .mlcc import Relabel as mlcc


def get_relabeling(args):
    if args.dataset == 'neu-cls':
        return neu_cls
    elif args.dataset == 'cifar10':
        return cifar10
    elif args.dataset == 'stl10':
        return stl10
    elif args.dataset == 'mlcc':
        return mlcc
