import numpy as np
import os
from pathlib import Path
import pickle
from PIL import Image
# from torchvision import models, transforms
# import torch.nn as nn
import torch
from libs.helper.classification_tools import CustomLabelEncoder
from libs.utils.yaml_config import init
from libs.dataset.preprocess import get_list_files
# from libs.pretext.processing import DataPreprocess, get_model, infer
from libs.pretext.utils import get_model
from libs.pretext import get_PretextCreater


# from libs.pretext.cifar10 import DataPreprocessFlow4


def extract_feature(cfg, logging, class_merging=False, shuffle_train=False):
    device = torch.device('cuda:{}'.format(cfg['master_model_params'].GPUS) if torch.cuda.is_available() else 'cpu')

    model = get_model(cfg, activation=False)

    if cfg['reduce_dimension_params']['type'] == 'vae':
        # tmp_cluster_dataset = cfg.cluster_dataset
        # cfg.cluster_dataset = 'train_test'
        DataPreprocess = get_PretextCreater(cfg)
        dp = DataPreprocess(cfg, class_merging=class_merging, shuffle_train=shuffle_train,
                            dataset_part=cfg['reduce_dimension_params']['dataset_part'])
        # dp = DataPreprocess(cfg, class_merging=class_merging, )
        dp.infer(model, device)
        dp.save_pretext_for_vae()
        # cfg.cluster_dataset = tmp_cluster_dataset
    DataPreprocess = get_PretextCreater(cfg)
    dp = DataPreprocess(cfg,
                        class_merging=class_merging,
                        shuffle_train=shuffle_train,
                        dataset_part=cfg['pretext_params']['dataset_part'])
    # dp.evaluate(model, device)
    dp.infer(model, device)
    dp.save_output()
    del dp


def extract_feature_flow4(cfg, shuffle_train=False):
    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
    add_noise = cfg['1st_train_params']['add_noise']
    model = get_model(cfg, activation=False)

    DataPreprocess = get_PretextCreater(cfg)
    dp = DataPreprocess(cfg,
                        noise='add_noise',
                        add_noise=add_noise,
                        renew_noise=False,
                        shuffle_train=shuffle_train,
                        dataset_part=cfg['pretext_params']['dataset_part'])
    # dp.evaluate(model, device)
    dp.infer(model, device)
    dp.save_output()
    del dp


if __name__ == '__main__':
    args, logging = init("experiments/neu-cls/flow1_resnet18.yaml")
    extract_feature(args, logging)
