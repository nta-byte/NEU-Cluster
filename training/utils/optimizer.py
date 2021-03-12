import torch


def get_optimizer(config, model):
    if config.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN.LR)
    elif config.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.TRAIN.LR)
    return optimizer
