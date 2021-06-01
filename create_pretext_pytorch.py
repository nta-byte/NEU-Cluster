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
from libs.pretext import get_data_preprocess
from libs.pretext.cifar10 import DataPreprocessFlow4


def extract_feature(args, config, logging, class_merging=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = get_model(args, activation=False)

    if args.reduce_dimension == 'vae':
        tmp_cluster_dataset = args.cluster_dataset
        args.cluster_dataset = 'train_test'
        DataPreprocess = get_data_preprocess(args)
        dp = DataPreprocess(args, config, class_merging=class_merging)
        dp.infer(model, device)
        dp.save_pretext_for_vae()
        args.cluster_dataset = tmp_cluster_dataset
    DataPreprocess = get_data_preprocess(args)
    dp = DataPreprocess(args, config, class_merging=class_merging)
    # dp.evaluate(model, device)
    dp.infer(model, device)
    dp.save_output()
    del dp


def extract_feature_flow4(args, config, logging, active_data='train'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = get_model(args, activation=False)

    # if args.reduce_dimension == 'vae':
    #     tmp_cluster_dataset = args.cluster_dataset
    #     args.cluster_dataset = 'train'
    #     # DataPreprocess = get_data_preprocess(args)
    #     dp = DataPreprocessFlow4(args, config)
    #     dp.infer(model, device)
    #     dp.save_pretext_for_vae()
    #     args.cluster_dataset = tmp_cluster_dataset
    # DataPreprocess = get_data_preprocess(args)
    dp = DataPreprocessFlow4(args, config, active_data=active_data)
    # dp.evaluate(model, device)
    dp.infer(model, device)
    dp.save_output()
    del dp


if __name__ == '__main__':
    args, logging = init("experiments/neu-cls/flow1_resnet18.yaml")
    extract_feature(args, logging)
