import os
import pickle

from pathlib import Path
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
import random
from PIL import Image


class DataPreprocess:
    def __init__(self, cfg, step=1, dataset_part='train', shuffle_train=False):
        self.cfg = cfg
        self.config = self.cfg['master_model_params']
        self.dataset_part = dataset_part
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transform_train = transforms.Compose([
            transforms.Resize((self.config.TRAIN.IMAGE_SIZE[0], self.config.TRAIN.IMAGE_SIZE[1])),
            transforms.RandomCrop(self.config.TRAIN.IMAGE_SIZE[0], padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        transform_val = transforms.Compose([
            transforms.Resize((self.config.TEST.IMAGE_SIZE[0], self.config.TEST.IMAGE_SIZE[1])),
            transforms.ToTensor(),  # 3*H*W, [0, 1]
            normalize])
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        print('cluster_dataset:', self.dataset_part)
        if self.dataset_part == 'train_test':
            trainset = torchvision.datasets.STL10(root=self.config.DATASET.ROOT, split='train',
                                                  download=True, transform=transform_train,
                                                  # target_transform=onehot,
                                                  )
            testset = torchvision.datasets.STL10(root=self.config.DATASET.ROOT, split='test',
                                                 download=True, transform=transform_val,
                                                 # target_transform=onehot,
                                                 )
            if step == 3:
                with open(self.config.DATASET.LE_PATH, 'rb') as f:
                    new_le = pickle.load(f)
                with open(self.config.DATASET.TRAIN_LIST, 'rb') as f:
                    train_label = pickle.load(f)
                with open(self.config.DATASET.VAL_LIST, 'rb') as f:
                    test_label = pickle.load(f)
                print(type(train_label), train_label)
                trainset.targets = new_le.transform(train_label.tolist())
                testset.targets = new_le.transform(test_label)
                print(new_le.mapper)
                print(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])))
                print(list(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])).keys()))
                self.classes = list(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])).keys())
                print('new class name: ', self.classes)
                print('new mapper:', new_le.mapper)
        elif self.dataset_part == 'train':
            dtset = torchvision.datasets.STL10(root=self.config.DATASET.ROOT, split='train',
                                               download=True, transform=transform_train,
                                               )
            if step == 3:
                with open(self.config.DATASET.LE_PATH, 'rb') as f:
                    new_le = pickle.load(f)
                with open(self.config.DATASET.TRAIN_LIST, 'rb') as f:
                    train_label = pickle.load(f)
                print(type(train_label), train_label)
                dtset.targets = new_le.transform(train_label.tolist())
                print(new_le.mapper)
                print(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])))
                print(list(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])).keys()))
                self.classes = list(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])).keys())
                print('new class name: ', self.classes)
                print('new mapper:', new_le.mapper)

            train_len = int(len(dtset) * .8)
            trainset, testset = torch.utils.data.random_split(dtset, [train_len, len(dtset) - train_len])
        elif self.dataset_part == 'test':
            dtset = torchvision.datasets.STL10(root=self.config.DATASET.ROOT, split='test',
                                               download=True, transform=transform_train,
                                               )
            if step == 3:
                with open(self.config.DATASET.LE_PATH, 'rb') as f:
                    new_le = pickle.load(f)
                with open(self.config.DATASET.VAL_LIST, 'rb') as f:
                    test_label = pickle.load(f)
                dtset.targets = new_le.transform(test_label.tolist())
                print(new_le.mapper)
                print(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])))
                print(list(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])).keys()))
                self.classes = list(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])).keys())
                print('new class name: ', self.classes)
                print('new mapper:', new_le.mapper)
            train_len = int(len(dtset) * .5)
            trainset, testset = torch.utils.data.random_split(dtset, [train_len, len(dtset) - train_len])
            # dataset_valid, dataset_test = torch.utils.data.random_split(dtset, [5000, 5000])
            trainset, testset = trainset.dataset, testset.dataset
            # print('old mapper:', trainset.class_to_idx)

        print('old class name: ', self.classes)
        print(self.config.TRAIN.BATCH_SIZE)
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.config.TRAIN.BATCH_SIZE,
                                                        shuffle=shuffle_train, num_workers=self.config.WORKERS)

        self.val_loader = torch.utils.data.DataLoader(testset, batch_size=self.config.TEST.BATCH_SIZE,
                                                      shuffle=False, num_workers=self.config.WORKERS)


class DataPreprocess2:
    def __init__(self, cfg, class_merging=False, renew_merge=False, add_noise=.1, shuffle_train=False,
                 dataset_part='train'):
        self.cfg = cfg
        self.renew_merge = renew_merge
        self.dataset_part = dataset_part
        self.config = self.cfg['master_model_params']
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transform_train = transforms.Compose([
            transforms.Resize((self.config.TRAIN.IMAGE_SIZE[0], self.config.TRAIN.IMAGE_SIZE[1])),
            transforms.RandomCrop(self.config.TRAIN.IMAGE_SIZE[0], padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        transform_val = transforms.Compose([
            transforms.Resize((self.config.TRAIN.IMAGE_SIZE[0], self.config.TRAIN.IMAGE_SIZE[1])),
            transforms.ToTensor(),  # 3*H*W, [0, 1]
            normalize])
        trainset = torchvision.datasets.STL10(root=self.config.DATASET.ROOT, split='train',
                                              download=True, transform=transform_train,
                                              )
        testset = torchvision.datasets.STL10(root=self.config.DATASET.ROOT, split='test',
                                             download=True, transform=transform_train,
                                             )
        trainset.class_to_idx = {_class: i for i, _class in enumerate(trainset.classes)}
        testset.class_to_idx = {_class: i for i, _class in enumerate(testset.classes)}
        self.original_le = CustomLabelEncoder()
        self.original_le.mapper = trainset.class_to_idx

        self.org_trainlabels = self.original_le.inverse_transform(trainset.labels)
        self.org_testlabel = self.original_le.inverse_transform(testset.labels)
        if self.dataset_part == 'train':
            self.org_labels = self.org_trainlabels
        elif self.dataset_part == 'test':
            self.org_labels = self.org_testlabel
        else:
            self.org_labels = np.concatenate((self.org_trainlabels, self.org_testlabel))

        if class_merging:
            if self.renew_merge:
                self.label_transform = None
            else:
                if os.path.isfile(self.cfg['pretext_params']['label_transform_path']):
                    print(f"reload self.label_transform in {self.cfg['pretext_params']['label_transform_path']}")
                    with open(self.cfg['pretext_params']['label_transform_path'], 'rb') as f:
                        self.label_transform = pickle.load(f)
                else:
                    self.label_transform = None

            trainset, testset = self.random_class_merging(trainset, testset)

        if add_noise != 0:
            self.create_noise()

        self.new_le = CustomLabelEncoder()
        self.new_le.mapper = trainset.class_to_idx

        self.new_trainlabels = self.new_le.inverse_transform(trainset.labels)
        self.new_testlabel = self.new_le.inverse_transform(testset.labels)
        if self.dataset_part == 'train':
            self.new_labels = self.new_trainlabels
        elif self.dataset_part == 'test':
            self.new_labels = self.new_testlabel
        else:
            self.new_labels = np.concatenate((self.new_trainlabels, self.new_testlabel))
        print('extract data set org label', set(self.org_labels))
        print('extract data set new label', set(self.new_labels))

        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.config.TRAIN.BATCH_SIZE,
                                                        shuffle=shuffle_train, num_workers=self.config.WORKERS)

        self.val_loader = torch.utils.data.DataLoader(testset, batch_size=self.config.TEST.BATCH_SIZE,
                                                      shuffle=False, num_workers=self.config.WORKERS)
        self.classes = trainset.classes

    def random_class_merging(self, trainset=None, testset=None):
        class_to_idx = trainset.class_to_idx
        le = CustomLabelEncoder()
        le.mapper = trainset.class_to_idx
        trainlabels = le.inverse_transform(trainset.labels)
        testlabels = le.inverse_transform(testset.labels)
        print("start to merge classes")
        if self.label_transform is None:
            print("create label_transform")
            self.label_transform = {'converted': {}, 'new_classes': None, 'new_class_to_idx': {},
                                    'new_le': CustomLabelEncoder()}
            classes = list(class_to_idx.keys())
            random.shuffle(classes)
            mergepoint = int(len(classes) / 2)
            new_classes1 = classes[:mergepoint]
            new_classes2 = classes[mergepoint:]
            self.label_transform['new_classes'] = [x1 + "_" + x2 for (x1, x2) in zip(new_classes1, new_classes2)]
            new_classes_list = [[x1, x2] for (x1, x2) in zip(new_classes1, new_classes2)]
            # print(new_classes_list)
            for group_classes in new_classes_list:
                self.label_transform['converted'][group_classes[0]] = "_".join(group_classes)
                self.label_transform['converted'][group_classes[1]] = "_".join(group_classes)
            for i, x in enumerate(self.label_transform['new_classes']):
                self.label_transform['new_class_to_idx'][x] = i

            self.label_transform['new_le'].mapper = self.label_transform['new_class_to_idx']
            with open(Path(self.cfg['pretext_params']['label_transform_path']), 'wb') as f:
                pickle.dump(self.label_transform, f)
        # else:
        new_trainlabels = [''] * len(trainlabels)
        for iddd, x in enumerate(trainlabels):
            new_trainlabels[iddd] = self.label_transform['converted'][x]
        new_testlabels = [''] * len(testlabels)
        for iddd, x in enumerate(testlabels):
            new_testlabels[iddd] = self.label_transform['converted'][x]

        # new_le = CustomLabelEncoder()
        trainset.labels = self.label_transform['new_le'].transform(new_trainlabels)
        print('check1', set(new_trainlabels))
        trainset.class_to_idx = self.label_transform['new_class_to_idx']
        trainset.classes = self.label_transform['new_classes']
        testset.labels = self.label_transform['new_le'].transform(new_testlabels)
        testset.class_to_idx = self.label_transform['new_class_to_idx']
        testset.classes = self.label_transform['new_classes']
        # print("self.label_transform", self.label_transform)
        # print("self.args.label_transform_path", self.args.label_transform_path)
        return trainset, testset

    def create_noise(self):
        pass


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
