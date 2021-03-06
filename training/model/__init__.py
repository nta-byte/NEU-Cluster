from .mobilenetv3 import MobileNetV3
from .resnet import resnet18
from torchvision import models, transforms
import torch.nn as nn
import torch


def get_model(config):
    pretrained = config.MODEL.PRETRAINED
    finetune = config.TRAIN.FINETUNE
    n_class = config.DATASET.NUM_CLASSES
    dropout = config.TRAIN.DROPOUT
    input_size = config.TRAIN.IMAGE_SIZE[0]
    mode = 'small'
    width_mult = 1.0
    if config.MODEL.NAME == 'mobilenetv3':
        model = MobileNetV3(n_class, input_size, dropout, mode, width_mult)
        # model = mobilenetv3(pretrained=pretrained, n_class=n_class, dropout=dropout, input_size=input_size)
    elif config.MODEL.NAME == 'resnet50':
        model = models.resnet.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(2048, n_class)
    elif config.MODEL.NAME == 'resnet18':
        model = resnet18(pretrained=pretrained)
        model.fc = nn.Linear(512, n_class)
    elif config.MODEL.NAME == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        model.classifier[6] = nn.Linear(4096, n_class)

    if finetune:
        pretrained_dict = torch.load(finetune,
                                     map_location=lambda storage, loc: storage)
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        print(len(pretrained_dict.keys()), len(model_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


# if __name__ == '__main__':

