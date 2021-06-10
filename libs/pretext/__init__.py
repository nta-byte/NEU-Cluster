from .cifar10 import DataPreprocess as cifar10
from .mlcc import DataPreprocess as mlcc
from .neu_cls import DataPreprocess as neu_cls
from .stl10 import DataPreprocess as stl10


def get_data_preprocess(cfg):
    if cfg['master_model_params'].DATASET.DATASET == 'mlcc':
        return mlcc
    elif cfg['master_model_params'].DATASET.DATASET == 'cifar10':
        return cifar10
    elif cfg['master_model_params'].DATASET.DATASET == 'stl10':
        return stl10
    elif cfg['master_model_params'].DATASET.DATASET == 'neu-cls':
        return neu_cls

