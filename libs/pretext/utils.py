import numpy as np
import os
from pathlib import Path
import pickle
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import torch.utils.data as data
# import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import torchvision

from libs.dataset.preprocess import get_list_files
from libs.helper.classification_tools import CustomLabelEncoder
from training.utils.loader import onehot


class Dataset(torch.utils.data.Dataset):
    def __init__(self, listfile, transform=None):
        self.transform = transform
        self.imgList = listfile
        self.dataList = []

    def __getitem__(self, index):
        imgpath = self.imgList[index]
        target = 0

        img = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        target = np.array(target).astype(np.float32)

        return img, target

    def __len__(self):
        return len(self.imgList)

    def get_classNum(self):
        return len(self.dataList[0])


def load_images(paths, min_img_size=224):
    # The min size, as noted in the PyTorch pretrained models doc, is 224 px.
    transform_pipeline = transforms.Compose([
        # transforms.Resize(min_img_size, interpolation=Image.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    train_dataset = Dataset(paths, transform_pipeline)
    # Data loader
    loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                         batch_size=16,
                                         num_workers=10,
                                         shuffle=False)
    return loader


def extract_labels(f):
    return [x.stem.split('_')[0] for x in f]


def load_state(weight_path, net):
    # state_dict = torch.load(args.pretrained_path,
    #                         map_location=lambda storage, loc: storage)
    # # model.load_state_dict(torch.load(config.MODEL.PRETRAINED,
    # #                                  map_location=lambda storage, loc: storage))
    # model.load_state_dict(state_dict, strict=True)

    pretrained_dict = torch.load(weight_path,
                                 map_location=lambda storage, loc: storage)
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    return net


def get_model(args):
    if args.model == 'RESNET50':
        model = models.resnet.resnet50(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
    elif args.model == 'VGG16':
        vgg16_path = '/home/ntanh/.cache/torch/checkpoints/vgg16-397923af.pth'
        model = models.vgg16(pretrained=True)
        model.classifier = nn.Sequential(*(list(model.classifier.children())[:1]))
        # model = load_state(vgg16_path, model)
    if args.pretrained_path:
        print("load weight:", args.pretrained_path)
        model = load_state(args.pretrained_path, model)
    return model


def get_data_list(args):
    def _get_list_files(data_preprocess_path, use_histeq):
        if use_histeq:
            data_preprocess_path = os.path.join(data_preprocess_path, 'images_histeq_resize')
        else:
            data_preprocess_path = os.path.join(data_preprocess_path, 'images_resize')
        img_root = Path(data_preprocess_path)  # directory where images are stored.
        files = get_list_files(img_root)
        return files

    if args.dataset == 'mlcc':
        files = _get_list_files(os.path.join(args.data_preprocess_path, 'train', 'images_preprocessed'),
                                args.use_histeq)
        files += _get_list_files(os.path.join(args.data_preprocess_path, 'valid', 'images_preprocessed'),
                                 args.use_histeq)
        files += _get_list_files(os.path.join(args.data_preprocess_path, 'test', 'images_preprocessed'),
                                 args.use_histeq)
    else:
        files = _get_list_files(os.path.join(args.data_preprocess_path, 'images_preprocessed'),
                                args.use_histeq)
        # files = _get_list_files(args.data_preprocess_path, args.use_histeq)
    print(len(files))
    # ## Shuffle the filenames so they appear randomly in the dataset.
    rs = np.random.RandomState(seed=749976)
    rs.shuffle(files)

    labels = extract_labels(files)
    print('labels', labels)
    return files, labels
