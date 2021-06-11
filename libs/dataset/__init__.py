import os
import pickle
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
# from training.utils.loader import DataLoader
from training.utils.loader import Dataset, onehot, NEU_Dataset, MLCCDataset
from libs.pretext.mlcc import mlcc_data_preprocess
from libs.helper import classification_tools as ct
from libs.pretext.utils import get_data_list, load_images
# from libs.pretext.utils import Dataset as NEU_Dataset
from libs.helper.classification_tools import CustomLabelEncoder
import torch
import torchvision
from torch._utils import _accumulate
from torch import randperm
import numpy as np
from PIL import Image

from .cifar10 import DataPreprocess as cifar10
from .cifar10 import DataPreprocess2 as cifar102

# from .mlcc import DataPreprocess as mlcc
# from .neu_cls import DataPreprocess as neu_cls
from .stl10 import DataPreprocess as stl10
from .stl10 import DataPreprocess2 as stl102


def get_data_preprocess(cfg):
    if cfg['master_model_params'].DATASET.DATASET == 'mlcc':
        return mlcc
    elif cfg['master_model_params'].DATASET.DATASET == 'cifar10':
        return cifar10
    elif cfg['master_model_params'].DATASET.DATASET == 'stl10':
        return stl10
    elif cfg['master_model_params'].DATASET.DATASET == 'neu-cls':
        return neu_cls


def get_data_preprocess2(cfg):
    if cfg['master_model_params'].DATASET.DATASET == 'mlcc':
        return mlcc
    elif cfg['master_model_params'].DATASET.DATASET == 'cifar10':
        return cifar102
    elif cfg['master_model_params'].DATASET.DATASET == 'stl10':
        return stl102
    elif cfg['master_model_params'].DATASET.DATASET == 'neu-cls':
        return neu_cls


#
# class DataPreprocess:
#     def __init__(self, config, args, step=1):
#         self.args = args
#         self.config = config
#         if args.dataset == 'mlcc':
#             normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#             transform_train = transforms.Compose([
#                 transforms.Resize((config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1])),
#                 # transforms.RandomCrop(config.TRAIN.IMAGE_SIZE[0], padding=4),
#                 transforms.RandomHorizontalFlip(),
#                 # transforms.RandomVerticalFlip(),
#                 transforms.ToTensor(),
#                 normalize
#             ])
#             transform_test = transforms.Compose([
#                 transforms.Resize((config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1])),
#                 transforms.ToTensor(),
#                 normalize
#             ])
#             self.files, self.labels = get_data_list(self.args, shuffle=False)
#             self.le = CustomLabelEncoder()
#             # self.le.fit(self.labels)
#             self.le.mapper = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, 'etc': 9}
#             self.classes = list(self.le.mapper.values())
#             if args.cluster_dataset == 'train_test':
#                 self.train_files = self.files[:950]
#                 self.train_labels = self.labels[:950]
#                 self.test_files = self.files[950:]
#                 self.test_labels = self.labels[950:]
#
#                 trainset = MLCCDataset(imgList=self.train_files, dataList=self.train_labels, le=self.le,
#                                        transform=transform_train)
#                 testset = MLCCDataset(imgList=self.test_files, dataList=self.test_labels, le=self.le,
#                                       transform=transform_test)
#
#                 if step == 3:
#                     with open(config.DATASET.LE_PATH, 'rb') as f:
#                         new_le = pickle.load(f)
#                     with open(config.DATASET.TRAIN_LIST, 'rb') as f:
#                         train_label = pickle.load(f)
#                     with open(config.DATASET.VAL_LIST, 'rb') as f:
#                         test_label = pickle.load(f)
#                     trainset.dataList_transformed = new_le.transform(train_label.tolist())
#                     testset.dataList_transformed = new_le.transform(test_label)
#                     self.classes = list(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])).keys())
#                     print('new class name: ', self.classes)
#                     print('new mapper:', new_le.mapper)
#             elif args.cluster_dataset == 'train':
#
#                 self.train_files = self.files[:950]
#                 self.train_labels = self.labels[:950]
#
#                 dataset = MLCCDataset(imgList=self.train_files, dataList=self.train_labels, le=self.le,
#                                       transform=transform_train)
#
#                 if step == 3:
#                     with open(config.DATASET.LE_PATH, 'rb') as f:
#                         new_le = pickle.load(f)
#                     with open(config.DATASET.TRAIN_LIST, 'rb') as f:
#                         train_label = pickle.load(f)
#                     dataset.dataList_transformed = new_le.transform(train_label.tolist())
#                     self.classes = list(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])).keys())
#                     print('new class name: ', self.classes)
#                     print('new mapper:', new_le.mapper)
#
#                 train_len = int(len(dataset) * .8)
#                 trainset, testset = torch.utils.data.random_split(dataset, [train_len, len(dataset) - train_len])
#             elif args.cluster_dataset == 'test':
#                 self.test_files = self.files[950:]
#                 self.test_labels = self.labels[950:]
#                 dataset = MLCCDataset(imgList=self.test_files, dataList=self.test_labels, le=self.le,
#                                       transform=transform_train)
#
#                 if step == 3:
#                     with open(config.DATASET.LE_PATH, 'rb') as f:
#                         new_le = pickle.load(f)
#                     with open(config.DATASET.VAL_LIST, 'rb') as f:
#                         test_label = pickle.load(f)
#                     dataset.dataList_transformed = new_le.transform(test_label.tolist())
#                     self.classes = list(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])).keys())
#                     print('new class name: ', self.classes)
#                     print('new mapper:', new_le.mapper)
#                 train_len = int(len(dataset) * .5)
#                 trainset, testset = torch.utils.data.random_split(dataset,
#                                                                   [train_len, len(dataset) - train_len])
#                 trainset, testset = trainset.dataset, testset.dataset
#
#             # Data loader
#             self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.TRAIN.BATCH_SIZE,
#                                                             shuffle=True, num_workers=config.WORKERS)
#
#             self.val_loader = torch.utils.data.DataLoader(testset, batch_size=config.TEST.BATCH_SIZE,
#                                                           shuffle=False, num_workers=config.WORKERS)
#             #
#             # self.loader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size,
#             #                                           shuffle=False, num_workers=self.args.workers)
#
#         elif args.dataset == 'cifar10':
#             normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#             transform_train = transforms.Compose([
#                 transforms.Resize((config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1])),
#                 transforms.RandomCrop(config.TRAIN.IMAGE_SIZE[0], padding=4),
#                 transforms.RandomHorizontalFlip(),
#                 # transforms.RandomVerticalFlip(),
#                 transforms.ToTensor(),
#                 normalize
#             ])
#
#             transform_val = transforms.Compose([
#                 # transforms.Resize((config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1])),
#                 transforms.ToTensor(),  # 3*H*W, [0, 1]
#                 normalize])
#             self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#             print('cluster_dataset:', args.cluster_dataset)
#             if args.cluster_dataset == 'train_test':
#                 trainset = torchvision.datasets.CIFAR10(root=args.dataset_root, train=True,
#                                                         download=True, transform=transform_train,
#                                                         )
#                 testset = torchvision.datasets.CIFAR10(root=args.dataset_root, train=False,
#                                                        download=True, transform=transform_val,
#                                                        )
#                 if step == 3:
#                     with open(config.DATASET.LE_PATH, 'rb') as f:
#                         new_le = pickle.load(f)
#                     with open(config.DATASET.TRAIN_LIST, 'rb') as f:
#                         train_label = pickle.load(f)
#                     with open(config.DATASET.VAL_LIST, 'rb') as f:
#                         test_label = pickle.load(f)
#                     print(type(train_label), train_label)
#                     trainset.targets = new_le.transform(train_label.tolist())
#                     testset.targets = new_le.transform(test_label)
#                     print(new_le.mapper)
#                     print(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])))
#                     print(list(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])).keys()))
#                     self.classes = list(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])).keys())
#                     print('new class name: ', self.classes)
#                     print('new mapper:', new_le.mapper)
#             elif args.cluster_dataset == 'train':
#                 dtset = torchvision.datasets.CIFAR10(root=args.dataset_root, train=True,
#                                                      download=True, transform=transform_train,
#                                                      )
#                 if step == 3:
#                     with open(config.DATASET.LE_PATH, 'rb') as f:
#                         new_le = pickle.load(f)
#                     with open(config.DATASET.TRAIN_LIST, 'rb') as f:
#                         train_label = pickle.load(f)
#                     print(type(train_label), train_label)
#                     dtset.targets = new_le.transform(train_label.tolist())
#                     print(new_le.mapper)
#                     print(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])))
#                     print(list(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])).keys()))
#                     self.classes = list(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])).keys())
#                     print('new class name: ', self.classes)
#                     print('new mapper:', new_le.mapper)
#                 trainset, testset = torch.utils.data.random_split(dtset, [40000, 10000])
#             elif args.cluster_dataset == 'test':
#                 dtset = torchvision.datasets.CIFAR10(root=args.dataset_root, train=False,
#                                                      download=True, transform=transform_train,
#                                                      )
#                 if step == 3:
#                     with open(config.DATASET.LE_PATH, 'rb') as f:
#                         new_le = pickle.load(f)
#                     with open(config.DATASET.VAL_LIST, 'rb') as f:
#                         test_label = pickle.load(f)
#                     dtset.targets = new_le.transform(test_label.tolist())
#                     print(new_le.mapper)
#                     print(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])))
#                     print(list(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])).keys()))
#                     self.classes = list(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])).keys())
#                     print('new class name: ', self.classes)
#                     print('new mapper:', new_le.mapper)
#                 trainset, testset = torch.utils.data.random_split(dtset, [8000, 2000])
#                 # dataset_valid, dataset_test = torch.utils.data.random_split(dtset, [5000, 5000])
#                 trainset, testset = trainset.dataset, testset.dataset
#                 print('old mapper:', trainset.class_to_idx)
#
#             print('old class name: ', self.classes)
#             # print('old mapper:', trainset.class_to_idx)
#
#             # testset.targets = onehot(testset.targets)
#             # trainset.targets = onehot(trainset.targets)
#             print(config.TRAIN.BATCH_SIZE)
#             self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.TRAIN.BATCH_SIZE,
#                                                             shuffle=True, num_workers=config.WORKERS)
#
#             self.val_loader = torch.utils.data.DataLoader(testset, batch_size=config.TEST.BATCH_SIZE,
#                                                           shuffle=False, num_workers=config.WORKERS)
#         elif args.dataset == 'stl10':
#             normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#             transform_train = transforms.Compose([
#                 transforms.RandomCrop(config.TRAIN.IMAGE_SIZE[0], padding=4),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 normalize
#             ])
#
#             transform_val = transforms.Compose([
#                 transforms.Resize((config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1])),
#                 transforms.ToTensor(),  # 3*H*W, [0, 1]
#                 normalize])
#             self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#             print('cluster_dataset:', args.cluster_dataset)
#             if args.cluster_dataset == 'train_test':
#                 trainset = torchvision.datasets.STL10(root=args.dataset_root, split='train',
#                                                       download=True, transform=transform_train,
#                                                       # target_transform=onehot,
#                                                       )
#                 testset = torchvision.datasets.STL10(root=args.dataset_root, split='test',
#                                                      download=True, transform=transform_train,
#                                                      # target_transform=onehot,
#                                                      )
#                 if step == 3:
#                     with open(config.DATASET.LE_PATH, 'rb') as f:
#                         new_le = pickle.load(f)
#                     with open(config.DATASET.TRAIN_LIST, 'rb') as f:
#                         train_label = pickle.load(f)
#                     with open(config.DATASET.VAL_LIST, 'rb') as f:
#                         test_label = pickle.load(f)
#                     print(type(train_label), train_label)
#                     trainset.targets = new_le.transform(train_label.tolist())
#                     testset.targets = new_le.transform(test_label)
#                     print(new_le.mapper)
#                     print(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])))
#                     print(list(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])).keys()))
#                     self.classes = list(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])).keys())
#                     print('new class name: ', self.classes)
#                     print('new mapper:', new_le.mapper)
#             elif args.cluster_dataset == 'train':
#                 dtset = torchvision.datasets.STL10(root=args.dataset_root, split='train',
#                                                    download=True, transform=transform_train,
#                                                    )
#                 if step == 3:
#                     with open(config.DATASET.LE_PATH, 'rb') as f:
#                         new_le = pickle.load(f)
#                     with open(config.DATASET.TRAIN_LIST, 'rb') as f:
#                         train_label = pickle.load(f)
#                     print(type(train_label), train_label)
#                     dtset.targets = new_le.transform(train_label.tolist())
#                     print(new_le.mapper)
#                     print(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])))
#                     print(list(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])).keys()))
#                     self.classes = list(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])).keys())
#                     print('new class name: ', self.classes)
#                     print('new mapper:', new_le.mapper)
#
#                 train_len = int(len(dtset) * .8)
#                 trainset, testset = torch.utils.data.random_split(dtset, [train_len, len(dtset) - train_len])
#             elif args.cluster_dataset == 'test':
#                 dtset = torchvision.datasets.STL10(root=args.dataset_root, split='test',
#                                                    download=True, transform=transform_train,
#                                                    )
#                 if step == 3:
#                     with open(config.DATASET.LE_PATH, 'rb') as f:
#                         new_le = pickle.load(f)
#                     # with open(config.DATASET.TRAIN_LIST, 'rb') as f:
#                     #     train_label = pickle.load(f)
#                     with open(config.DATASET.VAL_LIST, 'rb') as f:
#                         test_label = pickle.load(f)
#                     # print(type(train_label), train_label)
#                     dtset.targets = new_le.transform(test_label.tolist())
#                     # testset.targets = new_le.transform(test_label)
#                     print(new_le.mapper)
#                     print(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])))
#                     print(list(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])).keys()))
#                     self.classes = list(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])).keys())
#                     print('new class name: ', self.classes)
#                     print('new mapper:', new_le.mapper)
#                 train_len = int(len(dtset) * .5)
#                 trainset, testset = torch.utils.data.random_split(dtset, [train_len, len(dtset) - train_len])
#                 # trainset, testset = torch.utils.data.random_split(dtset, [2000, 6000])
#                 # dataset_valid, dataset_test = torch.utils.data.random_split(dtset, [5000, 5000])
#                 trainset, testset = trainset.dataset, testset.dataset
#                 # print('old mapper:', trainset.class_to_idx)
#
#             print('old class name: ', self.classes)
#             # print('old mapper:', trainset.class_to_idx)
#             # print(testset.labels)
#             # testset.labels = onehot(testset.labels)
#             # print(testset.labels)
#             # trainset.labels = onehot(trainset.labels)
#             # print('trainset.labels', trainset.labels)
#             # print('onehot(trainset.labels)', onehot(trainset.labels))
#             print(config.TRAIN.BATCH_SIZE)
#             self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.TRAIN.BATCH_SIZE,
#                                                             shuffle=True, num_workers=config.WORKERS)
#
#             self.val_loader = torch.utils.data.DataLoader(testset, batch_size=config.TEST.BATCH_SIZE,
#                                                           shuffle=False, num_workers=config.WORKERS)
#         elif args.dataset == 'neu-cls':
#
#             normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#             transform_train = transforms.Compose([
#                 transforms.Resize((config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1])),
#                 # transforms.RandomCrop(config.TRAIN.IMAGE_SIZE[0], padding=4),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomVerticalFlip(),
#                 transforms.ToTensor(),
#                 normalize
#             ])
#             transform_test = transforms.Compose([
#                 transforms.Resize((config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1])),
#                 transforms.ToTensor(),
#                 normalize
#             ])
#             self.files, self.labels = get_data_list(self.args, shuffle=True)
#             self.le = CustomLabelEncoder()
#             self.le.fit(self.labels)
#             # self.le.mapper = {'Cr': 0, 'In': 1, 'PS': 2, 'Pa': 3, 'RS': 4, 'Sc': 5}
#             # self.classes = list(self.le.mapper.values())
#             self.classes = list(dict(sorted(self.le.mapper.items(), key=lambda item: item[1])).keys())
#
#             split_len = int(len(self.files) * .5)
#             if args.cluster_dataset == 'train_test':
#
#                 self.train_files = self.files[:split_len]
#                 self.train_labels = self.labels[:split_len]
#                 self.test_files = self.files[split_len:]
#                 self.test_labels = self.labels[split_len:]
#
#                 trainset = NEU_Dataset(imgList=self.train_files, dataList=self.train_labels, le=self.le,
#                                        transform=transform_train)
#                 testset = NEU_Dataset(imgList=self.test_files, dataList=self.test_labels, le=self.le,
#                                       transform=transform_test)
#
#                 if step == 3:
#                     with open(config.DATASET.LE_PATH, 'rb') as f:
#                         new_le = pickle.load(f)
#                     with open(config.DATASET.TRAIN_LIST, 'rb') as f:
#                         train_label = pickle.load(f)
#                     with open(config.DATASET.VAL_LIST, 'rb') as f:
#                         test_label = pickle.load(f)
#                     trainset.dataList_transformed = new_le.transform(train_label.tolist())
#                     testset.dataList_transformed = new_le.transform(test_label)
#                     self.classes = list(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])).keys())
#                     print('new class name: ', self.classes)
#                     print('new mapper:', new_le.mapper)
#             elif args.cluster_dataset == 'train':
#
#                 self.train_files = self.files[:split_len]
#                 self.train_labels = self.labels[:split_len]
#
#                 dataset = NEU_Dataset(imgList=self.train_files, dataList=self.train_labels, le=self.le,
#                                       transform=transform_train)
#
#                 if step == 3:
#                     with open(config.DATASET.LE_PATH, 'rb') as f:
#                         new_le = pickle.load(f)
#                     with open(config.DATASET.TRAIN_LIST, 'rb') as f:
#                         train_label = pickle.load(f)
#                     dataset.dataList_transformed = new_le.transform(train_label.tolist())
#                     self.classes = list(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])).keys())
#                     print('new class name: ', self.classes)
#                     print('new mapper:', new_le.mapper)
#
#                 train_len = int(len(dataset) * .8)
#                 trainset, testset = torch.utils.data.random_split(dataset, [train_len, len(dataset) - train_len])
#             elif args.cluster_dataset == 'test':
#                 self.test_files = self.files[split_len:]
#                 self.test_labels = self.labels[split_len:]
#                 dataset = NEU_Dataset(imgList=self.test_files, dataList=self.test_labels, le=self.le,
#                                       transform=transform_train)
#
#                 if step == 3:
#                     with open(config.DATASET.LE_PATH, 'rb') as f:
#                         new_le = pickle.load(f)
#                     with open(config.DATASET.VAL_LIST, 'rb') as f:
#                         test_label = pickle.load(f)
#                     dataset.dataList_transformed = new_le.transform(test_label.tolist())
#                     self.classes = list(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])).keys())
#                     print('new class name: ', self.classes)
#                     print('new mapper:', new_le.mapper)
#
#                 train_len = int(len(dataset) * .5)
#                 trainset, testset = torch.utils.data.random_split(dataset,
#                                                                   [train_len, len(dataset) - train_len])
#                 trainset, testset = trainset.dataset, testset.dataset
#             print('len trainset:', len(trainset), 'len testset:', len(testset))
#
#             # Data loader
#             self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.TRAIN.BATCH_SIZE,
#                                                             shuffle=True, num_workers=config.WORKERS)
#
#             self.val_loader = torch.utils.data.DataLoader(testset, batch_size=config.TEST.BATCH_SIZE,
#                                                           shuffle=False, num_workers=config.WORKERS)
#             #
#             # self.loader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size,
#             #                                           shuffle=False, num_workers=self.args.workers)
#

if __name__ == '__main__':
    a = np.array([1, 2, 3, 4, 5, 6])
    # a = np.reshape(a, 6)
    print(a)
    # a = np.reshape(a, 6, order='F')
    # print(a)

    a = np.reshape(a, (-1, 1))
    print(a.shape)
    # a = onehot_transform(a)
    print(a)
