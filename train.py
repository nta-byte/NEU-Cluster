import os, sys
import logging

from datetime import datetime
import argparse
import numpy as np
from PIL import Image
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms

from imgaug import augmenters as iaa

from training.model import get_model
# from training.utils.loader import DataLoader, NewPad, data_loader_idcard, ImgAugTransform
from training.utils.core import train_step, evaluation
from training.utils.optimizer import get_optimizer
from training.config import update_config, config
from libs.utils.yaml_config import init
from libs.dataset import DataPreprocess


def parse_args():
    args, logging = init("experiments/cifar10/flow1_resnet50.yaml")

    update_config(config, args)

    return args, config


def train():
    args, config = parse_args()
    clusters = 10
    config.defrost()
    config.DATASET.NUM_CLASSES = clusters
    config.DATASET.TRAIN_LIST = os.path.join(args.relabel_dir, str(clusters) + '_train.txt')
    config.DATASET.VAL_LIST = os.path.join(args.relabel_dir, str(clusters) + '_valid.txt')
    config.MODEL.PRETRAINED = False
    # config.freeze()

    # Init save dir
    save_dir_root = args.training_ouput_dir
    save_dir = os.path.join(save_dir_root,
                            f'train_{clusters}_cluster')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # with open(args.le_path, 'rb') as f:
    #     le = pickle.load(f)
    # Init logging
    logname = os.path.join(save_dir, 'log_train.txt')
    handlers = [logging.FileHandler(logname), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(
        handlers=handlers,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(args)
    logging.info(config)

    # Device configuration
    device = torch.device('cuda:{}'.format(config.GPUS))

    data_preprocess = DataPreprocess(config, args)
    train_loader, val_loader = data_preprocess.train_loader, data_preprocess.val_loader
    # mapper = data_preprocess.classes
    classNum = config.DATASET.NUM_CLASSES
    # print(classNum)
    model = get_model(config)
    # print(model)
    model = model.to(device)
    if config.MODEL.PRETRAINED:
        logging.info(f"Finetune from the model {config.MODEL.PRETRAINED}")

    # Loss and optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN.LR)
    optimizer = get_optimizer(config, model)

    criterion = nn.CrossEntropyLoss()
    # print(mapper)
    # Train the model
    total_step = len(train_loader)
    # steps = total_step * 2
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=5,
    #                                               mode="exp_range", gamma=0.85)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=.995)
    max_acc = 0
    max_acc_epoch = 0
    min_loss = 1e7
    min_loss_epoch = 0
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        logging.info('Epoch [{}/{}]'.format(epoch, config.TRAIN.END_EPOCH))
        train_step(train_loader, model, criterion, optimizer, device, total_step, logging=logging, config=config,
                   epo=epoch,
                   debug_steps=config.TRAIN.PRINT_FREQ,
                   scheduler=scheduler)
        if epoch % config.TRAIN.VALIDATION_EPOCH == 0 or epoch == config.TRAIN.END_EPOCH - 1:
            val_result = evaluation(val_loader, model, criterion, device, classNum=classNum, logging=logging,
                                    le=data_preprocess.classes)
            val_loss = val_result['loss']
            val_acc = val_result['acc']
            if val_acc > max_acc or val_loss < min_loss:
                if val_acc > max_acc:
                    max_acc = val_acc
                    max_acc_epoch = epoch
                if val_loss < min_loss:
                    min_loss = val_loss
                    min_loss_epoch = epoch
                model_path = os.path.join(save_dir,
                                          f"{config.MODEL.NAME}-Epoch-{epoch}-Loss-{val_loss}-Acc-{val_acc}.pth")
                torch.save(model.state_dict(), model_path)
                logging.info(f"Saved model {model_path}")
            logging.info(f"best val Acc: {max_acc} in epoch: {max_acc_epoch}")
            logging.info(f"Min val Loss: {min_loss} in epoch: {min_loss_epoch}")
        logging.info('--------------------------------------------')


if __name__ == '__main__':
    train()
