import torch.utils.data as data
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import pickle
from torchvision import models, transforms
import torch
import torchvision

from libs.helper.classification_tools import CustomLabelEncoder
import random


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
            trainset = torchvision.datasets.CIFAR10(root=self.config.DATASET.ROOT, train=True,
                                                    download=True, transform=transform_train,
                                                    )
            testset = torchvision.datasets.CIFAR10(root=self.config.DATASET.ROOT, train=False,
                                                   download=True, transform=transform_val,
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
            dtset = torchvision.datasets.CIFAR10(root=self.config.DATASET.ROOT, train=True,
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
            dtset = torchvision.datasets.CIFAR10(root=self.config.DATASET.ROOT, train=False,
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
            train_len = int(len(dtset) * .8)
            trainset, testset = torch.utils.data.random_split(dtset, [train_len, len(dtset) - train_len])
            # dataset_valid, dataset_test = torch.utils.data.random_split(dtset, [5000, 5000])
            trainset, testset = trainset.dataset, testset.dataset
            print('old mapper:', trainset.class_to_idx)

        print('old class name: ', self.classes)
        print(self.config.TRAIN.BATCH_SIZE)
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.config.TRAIN.BATCH_SIZE,
                                                        shuffle=shuffle_train, num_workers=self.config.WORKERS)

        self.val_loader = torch.utils.data.DataLoader(testset, batch_size=self.config.TEST.BATCH_SIZE,
                                                      shuffle=False, num_workers=self.config.WORKERS)


class DataPreprocess2:
    def __init__(self, cfg, noise='add_noise', renew_merge=False, add_noise=0., renew_noise=False, renew_drop=True,
                 shuffle_train=False,
                 dataset_part='train'):
        """
        :param cfg:
        :param noise: None|add_noise|class_merging|drop_class
        :param renew_merge:
        :param add_noise:
        :param renew_noise:
        :param shuffle_train:
        :param dataset_part:
        """
        self.cfg = cfg
        self.noise = noise
        self.addnoise = add_noise
        self.renew_noise = renew_noise
        self.renew_merge = renew_merge
        self.renew_drop = renew_drop
        self.dataset_part = dataset_part
        self.config = self.cfg['master_model_params']
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transform_train = transforms.Compose([
            transforms.Resize((self.config.TRAIN.IMAGE_SIZE[0], self.config.TRAIN.IMAGE_SIZE[1])),
            transforms.RandomCrop(self.config.TRAIN.IMAGE_SIZE[0], padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        transform_val = transforms.Compose([
            transforms.Resize((self.config.TRAIN.IMAGE_SIZE[0], self.config.TRAIN.IMAGE_SIZE[1])),
            transforms.ToTensor(),  # 3*H*W, [0, 1]
            normalize])
        trainset = torchvision.datasets.CIFAR10(root=self.config.DATASET.ROOT, train=True,
                                                download=True, transform=transform_train,
                                                )

        testset = torchvision.datasets.CIFAR10(root=self.config.DATASET.ROOT, train=False,
                                               download=True, transform=transform_val,
                                               )
        self.original_le = CustomLabelEncoder()
        self.original_le.mapper = trainset.class_to_idx

        self.org_trainlabels = self.original_le.inverse_transform(trainset.targets)
        self.org_testlabel = self.original_le.inverse_transform(testset.targets)
        if self.dataset_part == 'train':
            self.org_labels = self.org_trainlabels
        elif self.dataset_part == 'test':
            self.org_labels = self.org_testlabel
        else:
            self.org_labels = np.concatenate((self.org_trainlabels, self.org_testlabel))
        print(self.noise)
        # augment data
        if self.noise == 'class_merging':
            if self.renew_merge:
                self.label_transform = {}
            else:
                if os.path.isfile(self.cfg['pretext_params']['label_transform_path']):
                    print(f"reload self.label_transform in {self.cfg['pretext_params']['label_transform_path']}")
                    with open(self.cfg['pretext_params']['label_transform_path'], 'rb') as f:
                        self.label_transform = pickle.load(f)
                else:
                    self.label_transform = {}

            trainset, testset = self.random_merge_class(trainset, testset)

        elif self.noise == 'add_noise':
            if self.renew_noise:
                self.label_transform = {}
            else:
                if os.path.isfile(self.cfg['pretext_params']['label_transform_path']):
                    print(f"reload self.label_transform in {self.cfg['pretext_params']['label_transform_path']}")
                    with open(self.cfg['pretext_params']['label_transform_path'], 'rb') as f:
                        self.label_transform = pickle.load(f)
                else:
                    self.label_transform = {}

            trainset = self.create_noise(trainset)

        elif self.noise == 'drop_class':
            if self.renew_drop:
                self.label_transform = {}
            else:
                if os.path.isfile(self.cfg['pretext_params']['label_transform_path']):
                    print(f"reload self.label_transform in {self.cfg['pretext_params']['label_transform_path']}")
                    with open(self.cfg['pretext_params']['label_transform_path'], 'rb') as f:
                        self.label_transform = pickle.load(f)
                else:
                    self.label_transform = {}

            trainset = self.random_drop_class(trainset)

        self.new_le = CustomLabelEncoder()
        self.new_le.mapper = trainset.class_to_idx

        self.new_trainlabels = self.new_le.inverse_transform(trainset.targets)
        self.new_testlabel = self.new_le.inverse_transform(testset.targets)
        if self.dataset_part == 'train':
            self.new_labels = self.new_trainlabels
        elif self.dataset_part == 'test':
            self.new_labels = self.new_testlabel
        else:
            self.new_labels = np.concatenate((self.new_trainlabels, self.new_testlabel))
        print('extract data set org label', set(self.org_labels))
        print('extract data set new label', set(self.new_labels))

        if self.dataset_part == 'train':
            train_len = int(len(trainset) * .8)
            trainset, testset = torch.utils.data.random_split(trainset, [train_len, len(trainset) - train_len])
            trainset, testset = trainset.dataset, testset.dataset
        elif self.dataset_part == 'test':
            train_len = int(len(testset) * .8)
            trainset, testset = torch.utils.data.random_split(testset, [train_len, len(testset) - train_len])
            trainset, testset = trainset.dataset, testset.dataset

        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.config.TRAIN.BATCH_SIZE,
                                                        shuffle=shuffle_train, num_workers=self.config.WORKERS)

        self.val_loader = torch.utils.data.DataLoader(testset, batch_size=self.config.TEST.BATCH_SIZE,
                                                      shuffle=False, num_workers=self.config.WORKERS)
        self.classes = trainset.classes

    def random_merge_class(self, trainset=None, testset=None):
        class_to_idx = trainset.class_to_idx
        le = CustomLabelEncoder()
        le.mapper = class_to_idx
        trainlabels = le.inverse_transform(trainset.targets)
        testlabels = le.inverse_transform(testset.targets)
        print("start to merge classes")
        if self.label_transform == {}:
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
        trainset.targets = self.label_transform['new_le'].transform(new_trainlabels)
        print('check1', set(new_trainlabels))
        trainset.class_to_idx = self.label_transform['new_class_to_idx']
        trainset.classes = self.label_transform['new_classes']
        testset.targets = self.label_transform['new_le'].transform(new_testlabels)
        testset.class_to_idx = self.label_transform['new_class_to_idx']
        testset.classes = self.label_transform['new_classes']
        # print("self.label_transform", self.label_transform)
        # print("self.args.label_transform_path", self.args.label_transform_path)
        return trainset, testset

    def create_noise(self, trainset=None):
        if 'new_noise_label' not in self.label_transform:
            train_labels = np.asarray(trainset.targets)
            ix_size = int(self.addnoise * len(train_labels))
            ix = np.random.choice(len(train_labels), size=ix_size, replace=False)
            b = train_labels[ix]
            np.random.shuffle(b)
            train_labels[ix] = b

            # test_labels = np.asarray(testset.targets)
            # ix_size = int(self.addnoise * len(test_labels))
            # ix = np.random.choice(len(test_labels), size=ix_size, replace=False)
            # b = test_labels[ix]
            # np.random.shuffle(b)
            # test_labels[ix] = b

            new_noise_label = {'new_train_labels': train_labels}
            self.label_transform['new_noise_label'] = new_noise_label

            with open(Path(self.cfg['pretext_params']['label_transform_path']), 'wb') as f:
                pickle.dump(self.label_transform, f)
        else:
            # test_labels = self.label_transform['new_noise_label']['new_test_labels']
            train_labels = self.label_transform['new_noise_label']['new_train_labels']
        # testset.targets = test_labels
        trainset.targets = train_labels
        return trainset

    def random_drop_class(self, dataset):
        if 'new_noise_label' not in self.label_transform:
            train_labels = np.asarray(dataset.targets)
            # ix_size = int(self.addnoise * len(train_labels))
            original_class_name = list(dataset.class_to_idx.keys())
            original_class_id = np.array(list(dataset.class_to_idx.values()))
            dropping_class_name = random.choice(original_class_name)
            dropping_class_id = dataset.class_to_idx[dropping_class_name]
            print(original_class_name)
            print("dropped class:", dropping_class_name, " id:", dropping_class_id)
            ix = np.where(train_labels == dropping_class_id)
            # print(np.array(list(dataset.class_to_idx.values())))
            # print(set(ix))
            dropped_label = np.delete(original_class_id, [dropping_class_id])
            print(dropped_label)
            # ix = np.random.choice(len(train_labels), size=ix_size, replace=False)
            # b = train_labels[ix]
            # np.random.shuffle(b)
            train_labels[ix] = np.random.choice(dropped_label)
            print(set(train_labels))

            new_noise_label = {'new_train_labels': train_labels}
            self.label_transform['new_noise_label'] = new_noise_label

            with open(Path(self.cfg['pretext_params']['label_transform_path']), 'wb') as f:
                pickle.dump(self.label_transform, f)
        else:
            train_labels = self.label_transform['new_noise_label']['new_train_labels']
        dataset.targets = train_labels
        return dataset


class DataPreprocessFlow4:
    def __init__(self, cfg, noise='add_noise', add_noise=0, active_data='train', renew_noise=False, decrease_dim=False):
        self.noise = noise
        self.addnoise = add_noise
        self.cfg = cfg
        self.config = self.cfg['master_model_params']
        self.active_data = active_data
        self.renew_noise = renew_noise
        self.decrease_dim = decrease_dim
        print(f"dataset: {self.args.dataset}")
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
        self.org_trainset = torchvision.datasets.CIFAR10(root=self.config.DATASET.ROOT, train=True,
                                                         download=True, transform=transform_val,
                                                         )

        self.org_testset = torchvision.datasets.CIFAR10(root=self.config.DATASET.ROOT, train=False,
                                                        download=True, transform=transform_val,
                                                        )
        self.original_le = CustomLabelEncoder()
        self.original_le.mapper = self.org_trainset.class_to_idx

        self.org_trainlabels = self.original_le.inverse_transform(self.org_trainset.targets)
        self.org_testlabels = self.original_le.inverse_transform(self.org_testset.targets)

        if add_noise != 0:
            if self.renew_noise:
                self.label_transform = None
            else:
                if os.path.isfile(self.args.label_transform_path):
                    print(f'reload self.label_transform in {self.args.label_transform_path}')
                    with open(self.args.label_transform_path, 'rb') as f:
                        self.label_transform = pickle.load(f)
                else:
                    self.label_transform = None

            self.create_noise()

        # if class_merging:

        self.new_trainlabels = self.original_le.inverse_transform(self.org_trainset.targets)

        train_len = int(len(self.org_trainset) * .8)
        trainset, valset = torch.utils.data.random_split(self.org_trainset,
                                                         [train_len, len(self.org_trainset) - train_len])
        valset.dataset.transform = transform_val
        trainset.dataset.transform = transform_train

        self.org_trainloader = torch.utils.data.DataLoader(self.org_trainset, batch_size=self.config.TRAIN.BATCH_SIZE,
                                                           shuffle=False, num_workers=self.config.WORKERS)

        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.config.TRAIN.BATCH_SIZE,
                                                        shuffle=True, num_workers=self.config.WORKERS)
        self.val_loader = torch.utils.data.DataLoader(valset, batch_size=self.config.TEST.BATCH_SIZE,
                                                      shuffle=False, num_workers=self.config.WORKERS)
        self.test_loader = torch.utils.data.DataLoader(self.org_testset, batch_size=self.config.TEST.BATCH_SIZE,
                                                       shuffle=False, num_workers=self.config.WORKERS)
        self.classes = self.org_trainset.classes

    def create_noise(self):
        if self.label_transform is None:
            self.label_transform = self.org_trainset.targets
            self.label_transform = np.asarray(self.label_transform)
            ix_size = int(self.addnoise * len(self.label_transform))
            ix = np.random.choice(len(self.label_transform), size=ix_size, replace=False)
            b = self.label_transform[ix]
            np.random.shuffle(b)
            self.label_transform[ix] = b
            with open(Path(self.args.label_transform_path), 'wb') as f:
                pickle.dump(self.label_transform, f)
        self.org_trainset.targets = self.label_transform

    def drop_class(self):
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
