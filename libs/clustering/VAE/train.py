import torch as t
from torch import nn

from torch.nn import functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import os.path as osp
import yaml
import argparse
from datetime import datetime

from data_loader import VAEDataset
from models.models import VAE, SWAE
from early_stoppping import EarlyStopping


# return clsloss


def get_optimizer(config, model):
    optimizer, scheduler = None, None
    if config['trainer_params']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['trainer_params']['LR'])
    elif config['trainer_params']['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['trainer_params']['LR'],
                                    momentum=config['trainer_params']['momentum'],
                                    weight_decay=config['trainer_params']['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    return optimizer, scheduler


def train(model, epoch, optimizer, train_loader, device, n_genes, scheduler=None):
    model.train()
    train_loss = 0.0
    for bix, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # pred = model(data)
        # loss = model.loss_function(pred, labels=labels)
        results = model(data, labels=labels)
        loss = model.loss_function(*results,
                                   # M_N=self.params['batch_size'] / self.num_train_imgs,
                                   # optimizer_idx=optimizer_idx,
                                   # batch_idx=batch_idx
                                   )
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if scheduler:
            scheduler.step()
        if bix % 10 == 0:
            btch = bix * int(data.shape[0])
            print(f"\rTrain Epoch : {epoch:10d} [{btch:10} / {len(train_loader.dataset)} ] | Loss : {loss.item()}"
                  , end='')


def evaluation(val_loader, model, dev):
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        for idx, batch in enumerate(val_loader):
            if idx >= len(val_loader):
                break
            data = batch[0].to(dev)
            labels = batch[1].to(dev)
            # pred = net(data)
            # loss = net.loss_function(pred, labels=labels)
            results = model(data, labels=labels)
            loss = model.loss_function(*results,
                                       # M_N=self.params['batch_size'] / self.num_train_imgs,
                                       # optimizer_idx=optimizer_idx,
                                       # batch_idx=batch_idx
                                       )

            running_loss += loss.item()
    return running_loss / len(val_loader)


def fit(config):
    save_dir = config['logging_params']['save_dir']
    outdir = (save_dir if save_dir else osp.dirname(save_dir))
    outdir = osp.join(outdir, ''.join(['results_']))
    if not osp.exists(outdir): os.makedirs(outdir)

    n_genes = config['model_params']['n_genes']
    dataset = VAEDataset(path_datain=config['exp_params']['train_data_path'])
    train_len = int(len(dataset) * .8)
    trainset, valset = torch.utils.data.random_split(dataset, [train_len, len(dataset) - train_len])

    train_loader = t.utils.data.DataLoader(trainset,
                                           batch_size=config['trainer_params']['batch_size'],
                                           shuffle=True)

    val_loader = t.utils.data.DataLoader(valset,
                                         batch_size=config['trainer_params']['batch_size'],
                                         shuffle=False)

    os.environ["CUDA_VISIBLE_DEVICES"] = config['trainer_params']['gpus']
    device = t.device('cuda:{}'.format(0))
    model = SWAE(n_genes, latent_dim=config['model_params']['latent_dim'],
                 hidden_dims=config['model_params']['hidden_dims']).to(device)
    scheduler = None
    # optimizer = optim.Adam(model.parameters())
    optimizer, scheduler = get_optimizer(config, model)
    save_best_model = os.path.join(save_dir, f"{config['model_params']['name']}-best.pth")
    early_stopping = EarlyStopping(patience=25, verbose=True, path=save_best_model)
    print('initiate training')
    nepochs = config['trainer_params']['max_epochs']
    min_loss = 1e7
    min_loss_epoch = 0
    try:
        for epoch in range(1, nepochs):
            train(model, epoch, optimizer, train_loader, device, n_genes, scheduler=scheduler)

            if epoch % config['trainer_params']['validate_epoch'] == 0 or epoch == nepochs - 1:
                val_loss = evaluation(val_loader, model, device)
                # val_loss = val_result['loss']
                # val_acc = val_result['acc']
                if val_loss < min_loss:

                    if val_loss < min_loss:
                        min_loss = val_loss
                        min_loss_epoch = epoch
                    model_path = os.path.join(save_dir,
                                              f"{config['model_params']['name']}-Epoch-{epoch}-Loss-{val_loss}.pth")

                # print(f"best val Acc: {max_acc} in epoch: {max_acc_epoch}")
                print(f"Min val Loss: {min_loss} in epoch: {min_loss_epoch}")
                early_stopping(val_loss, model)

                if early_stopping.early_stop:
                    torch.save(model.state_dict(), model_path)
                    print(f"Saved model {model_path}")
                    print("Early stopping")
                    break
            # print('--------------------------------------------')

    except KeyboardInterrupt:
        print('Early Interruption')
    return save_best_model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='experiments/simplevae.yaml')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    print(config)

    fit(config)
