from .neu_cls import Relabel as neu_cls
from .cifar10 import Relabel as cifar10


def get_relabeling(args):
    if args.dataset == 'neu-cls':
        return neu_cls
    elif args.dataset == 'cifar10':
        return cifar10
