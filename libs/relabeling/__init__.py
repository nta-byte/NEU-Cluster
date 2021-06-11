from .neu_cls import Relabel as neu_cls
from .cifar10 import Relabel as cifar10
from .stl10 import Relabel as stl10
from .mlcc import Relabel as mlcc


def get_relabeling(cfg):
    if cfg['master_model_params'].DATASET.DATASET == 'mlcc':
        return mlcc
    elif cfg['master_model_params'].DATASET.DATASET == 'cifar10':
        return cifar10
    elif cfg['master_model_params'].DATASET.DATASET == 'stl10':
        return stl10
    elif cfg['master_model_params'].DATASET.DATASET == 'neu-cls':
        return neu_cls
