import numpy as np
import os
from pathlib import Path
import pickle
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import torch
from libs.helper.classification_tools import CustomLabelEncoder
from libs.utils.yaml_config import init
from libs.dataset.preprocess import get_list_files
# from libs.pretext.processing import DataPreprocess, get_model, infer
from libs.pretext.utils import get_model
from libs.pretext import get_data_preprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def extract_feature(args, logging):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    DataPreprocess = get_data_preprocess(args)
    dp = DataPreprocess(args)

    model = get_model(args)

    dp.infer(model, device)
    dp.save_output()


if __name__ == '__main__':
    args, logging = init("experiments/cifar10/resnet50.yaml")
    extract_feature(args, logging)
