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
# from training.config import update_config, config
from libs.utils.yaml_config import init
from libs.dataset import get_data_preprocess
from libs.dataset import get_data_preprocess2
# from libs.pretext import get_data_preprocess
# from libs.pretext.cifar10 import DataPreprocessFlow4
from training.utils.early_stoppping import EarlyStopping


def train_function(cfg, step=1, dataset_part=''):
    config = cfg['master_model_params']
    # preprocess
    clusters = config.DATASET.NUM_CLASSES

    # Init save dir
    if step == 1:
        save_dir_root = cfg['general']['first_train_dir']
    elif step == 3:
        save_dir_root = cfg['general']['last_train_dir']
    save_dir = os.path.join(save_dir_root,
                            f'train_{clusters}_cluster')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Init logging
    logname = os.path.join(save_dir, 'log_train.txt')
    handlers = [logging.FileHandler(logname), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(
        handlers=handlers,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(cfg)
    logging.info(config)

    # Device configuration
    device = torch.device('cuda:{}'.format(config.GPUS))

    # data prepare
    DataPreprocess = get_data_preprocess(cfg)
    dp = DataPreprocess(cfg, step=step, dataset_part=dataset_part, shuffle_train=True)
    train_loader, val_loader = dp.train_loader, dp.val_loader

    # data_preprocess = DataPreprocess(config, cfg, step=step)
    # train_loader, val_loader = data_preprocess.train_loader, data_preprocess.val_loader

    # model prepare
    model = get_model(config)
    model = model.to(device)

    # criteria prepare
    # Loss and optimizer
    optimizer, scheduler = get_optimizer(config, model)

    criterion = nn.CrossEntropyLoss()

    # initialize the early_stopping object
    save_best_model = os.path.join(save_dir, f"{config.MODEL.NAME}-best.pth")
    early_stopping = EarlyStopping(patience=config.TRAIN.EarlyStopping, verbose=True, path=save_best_model)

    # train process
    total_step = len(train_loader)
    classNum = config.DATASET.NUM_CLASSES
    max_acc = 0
    max_acc_epoch = 0
    min_loss = 1e7
    min_loss_epoch = 0
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        logging.info('Epoch [{}/{}]'.format(epoch, config.TRAIN.END_EPOCH))
        train_step(train_loader, model, criterion, optimizer, device, total_step, logging=logging, config=config,
                   epo=epoch,
                   debug_steps=config.TRAIN.PRINT_FREQ,
                   scheduler=scheduler
                   )
        if epoch % config.TRAIN.VALIDATION_EPOCH == 0 or epoch == config.TRAIN.END_EPOCH - 1:
            val_result = evaluation(val_loader, model, criterion, device, classNum=classNum, logging=logging,
                                    le=dp.classes)
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
            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
            if val_loss == 0.0 or val_acc == 1.0:
                break
        logging.info('--------------------------------------------')
        if scheduler is not None:
            scheduler.step()
    return save_best_model


def train_function2(cfg, dataset_part='train'):
    # preprocess
    clusters = int(cfg['master_model_params'].DATASET.NUM_CLASSES / 2)

    # Init save dir
    save_dir_root = cfg['general']['first_train_dir']
    save_dir = os.path.join(save_dir_root, f'train_{clusters}_cluster')
    config = cfg['master_model_params']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Init logging
    logname = os.path.join(save_dir, 'log_train.txt')
    handlers = [logging.FileHandler(logname), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(
        handlers=handlers,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(cfg)
    logging.info(cfg['master_model_params'])

    # Device configuration
    device = torch.device('cuda:{}'.format(cfg['master_model_params'].GPUS))

    # data prepare
    DataPreprocess = get_data_preprocess2(cfg)
    dp = DataPreprocess(cfg, class_merging=True, shuffle_train=True, dataset_part='train')
    train_loader, val_loader = dp.train_loader, dp.val_loader

    # model prepare
    config.DATASET.NUM_CLASSES = len(dp.classes)
    model = get_model(cfg['master_model_params'])
    model = model.to(device)
    print(model)

    # criteria prepare
    # Loss and optimizer
    optimizer, scheduler = get_optimizer(cfg['master_model_params'], model)
    criterion = nn.CrossEntropyLoss()

    # initialize the early_stopping object
    save_best_model = os.path.join(save_dir, f"{config.MODEL.NAME}-best.pth")
    early_stopping = EarlyStopping(patience=config.TRAIN.EarlyStopping, verbose=True, path=save_best_model)

    # train process
    total_step = len(train_loader)
    classNum = config.DATASET.NUM_CLASSES
    max_acc = 0
    max_acc_epoch = 0
    min_loss = 1e7
    min_loss_epoch = 0
    smallest_loss_weight_path = ''
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        logging.info('Epoch [{}/{}]'.format(epoch, config.TRAIN.END_EPOCH))
        train_step(loader=train_loader, net=model, crit=criterion, optim=optimizer, dev=device, total_step=total_step,
                   logging=logging, config=config,
                   epo=epoch,
                   debug_steps=config.TRAIN.PRINT_FREQ,
                   scheduler=scheduler
                   )
        if epoch % config.TRAIN.VALIDATION_EPOCH == 0 or epoch == config.TRAIN.END_EPOCH - 1:
            val_result = evaluation(val_loader, model, criterion, device, classNum=classNum, logging=logging,
                                    le=dp.classes)
            val_loss = val_result['loss']
            val_acc = val_result['acc']
            if val_acc > max_acc or val_loss < min_loss:
                model_path = os.path.join(save_dir,
                                          f"{config.MODEL.NAME}-Epoch-{epoch}-Loss-{val_loss}-Acc-{val_acc}.pth")
                if val_acc > max_acc:
                    max_acc = val_acc
                    max_acc_epoch = epoch
                if val_loss < min_loss:
                    min_loss = val_loss
                    min_loss_epoch = epoch
                    smallest_loss_weight_path = model_path

                torch.save(model.state_dict(), model_path)
                logging.info(f"Saved model {model_path}")
            logging.info(f"best val Acc: {max_acc} in epoch: {max_acc_epoch}")
            logging.info(f"Min val Loss: {min_loss} in epoch: {min_loss_epoch}")

            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
            if val_loss == 0.0 or val_acc == 1.0:
                break
        logging.info('--------------------------------------------')
        if scheduler is not None:
            scheduler.step()
    return smallest_loss_weight_path


def train_function3(cfg, dataset_part='train'):
    # preprocess
    clusters = int(cfg['master_model_params'].DATASET.NUM_CLASSES)

    # Init save dir
    save_dir_root = cfg['general']['first_train_dir']
    save_dir = os.path.join(save_dir_root, f'train_{clusters}_cluster')
    config = cfg['master_model_params']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Init logging
    logname = os.path.join(save_dir, 'log_train.txt')
    handlers = [logging.FileHandler(logname), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(
        handlers=handlers,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(cfg)
    logging.info(cfg['master_model_params'])

    # Device configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['master_model_params'].GPUS)
    device = torch.device('cuda:{}'.format(0))

    # data prepare
    # DataPreprocess = DataPreprocessFlow4(args)
    DataPreprocess = get_data_preprocess2(cfg)
    dp = DataPreprocess(cfg, noise=cfg['1st_train_params']['noise'], add_noise=cfg['1st_train_params']['add_noise'],
                        shuffle_train=True,
                        dataset_part=dataset_part)
    train_loader, val_loader = dp.train_loader, dp.val_loader

    # model prepare
    config.DATASET.NUM_CLASSES = len(dp.classes)
    model = get_model(config)
    model = model.to(device)
    # print(model)

    # criteria prepare
    # Loss and optimizer
    optimizer, scheduler = get_optimizer(config, model)
    # print(optimizer)
    criterion = nn.CrossEntropyLoss()

    # initialize the early_stopping object
    save_best_model = os.path.join(save_dir, f"{config.MODEL.NAME}-best.pth")
    early_stopping = EarlyStopping(patience=25, verbose=True, path=save_best_model)

    # train process
    total_step = len(train_loader)
    classNum = config.DATASET.NUM_CLASSES
    max_acc = 0
    max_acc_epoch = 0
    min_loss = 1e7
    min_loss_epoch = 0
    smallest_loss_weight_path = ''
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        logging.info('Epoch [{}/{}]'.format(epoch, config.TRAIN.END_EPOCH))
        train_step(loader=train_loader, net=model, crit=criterion, optim=optimizer, dev=device, total_step=total_step,
                   logging=logging, config=config,
                   epo=epoch,
                   debug_steps=config.TRAIN.PRINT_FREQ,
                   scheduler=scheduler
                   )
        if epoch % config.TRAIN.VALIDATION_EPOCH == 0 or epoch == config.TRAIN.END_EPOCH - 1:
            val_result = evaluation(val_loader, model, criterion, device, classNum=classNum, logging=logging,
                                    le=dp.classes)
            val_loss = val_result['loss']
            val_acc = val_result['acc']
            if val_acc > max_acc or val_loss < min_loss:
                model_path = os.path.join(save_dir,
                                          f"{config.MODEL.NAME}-Epoch-{epoch}-Loss-{val_loss}-Acc-{val_acc}.pth")
                if val_acc > max_acc:
                    max_acc = val_acc
                    max_acc_epoch = epoch
                if val_loss < min_loss:
                    min_loss = val_loss
                    min_loss_epoch = epoch
                    smallest_loss_weight_path = model_path

                torch.save(model.state_dict(), model_path)
                logging.info(f"Saved model {model_path}")
            logging.info(f"best val Acc: {max_acc} in epoch: {max_acc_epoch}")
            logging.info(f"Min val Loss: {min_loss} in epoch: {min_loss_epoch}")

            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
            if val_loss == 0.0 or val_acc == 1.0:
                break
        logging.info('--------------------------------------------')
        if scheduler is not None:
            scheduler.step()
    return smallest_loss_weight_path
