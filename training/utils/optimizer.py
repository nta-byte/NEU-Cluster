import torch


def get_optimizer(config, model):
    scheduler = None
    if config.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN.LR)
    elif config.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.TRAIN.LR, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    return optimizer, scheduler
