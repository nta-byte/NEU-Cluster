from .cifar10 import PretextCreater as cifar10


# from .mlcc import PretextCreater as mlcc
# from .neu_cls import PretextCreater as neu_cls
from .stl10 import PretextCreater as stl10


def get_PretextCreater(cfg):
    if cfg['master_model_params'].DATASET.DATASET == 'cifar10':
        return cifar10
    elif cfg['master_model_params'].DATASET.DATASET == 'mlcc':
        return mlcc
    elif cfg['master_model_params'].DATASET.DATASET == 'stl10':
        return stl10
    elif cfg['master_model_params'].DATASET.DATASET == 'neu-cls':
        return neu_cls
