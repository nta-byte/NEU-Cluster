import os
import pickle
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
# from training.utils.loader import DataLoader
from training.utils.loader import Dataset, onehot
from libs.helper import classification_tools as ct
import torch
import torchvision
import numpy as np
from PIL import Image


class DataPreprocess:
    def __init__(self, config, args, step=1):
        if args.dataset == 'mlcc':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transform_train = transforms.Compose([
                transforms.Resize((config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1]),
                                  interpolation=Image.NEAREST),
                transforms.ToTensor(),
                normalize
            ])

            transform_val = transforms.Compose([
                transforms.Resize((config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1]),
                                  interpolation=Image.NEAREST),
                transforms.ToTensor(),  # 3*H*W, [0, 1]
                normalize])

            train_dataset = Dataset(
                label_file=config.DATASET.TRAIN_LIST,
                transform=transform_train, augment=None)

            le = train_dataset.le
            print(le.mapper)
            self.classes = list(le.mapper.values())

            val_dataset = Dataset(
                label_file=config.DATASET.VAL_LIST,
                transform=transform_val, augment=None, le=le)

            # Data loader
            self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                            batch_size=config.TRAIN.BATCH_SIZE,
                                                            num_workers=config.WORKERS,
                                                            shuffle=True)

            self.val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                          batch_size=config.TEST.BATCH_SIZE,
                                                          num_workers=config.WORKERS,
                                                          shuffle=False)

        elif args.dataset == 'cifar10':
            normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            transform_train = transforms.Compose([

                transforms.RandomCrop(config.TRAIN.IMAGE_SIZE[0], padding=4),
                transforms.RandomHorizontalFlip(),
                # transforms.Resize((config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1]),
                #                   interpolation=Image.NEAREST),
                transforms.ToTensor(),
                normalize
            ])

            transform_val = transforms.Compose([
                transforms.Resize((config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1]),
                                  interpolation=Image.NEAREST),
                transforms.ToTensor(),  # 3*H*W, [0, 1]
                normalize])
            # transform = transforms.Compose([
            #     transforms.Resize((config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1])),
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # ])

            trainset = torchvision.datasets.CIFAR10(root='/data4T/ntanh/data/', train=True,
                                                    download=False, transform=transform_train,
                                                    # target_transform=one_hot
                                                    )
            testset = torchvision.datasets.CIFAR10(root='/data4T/ntanh/data/', train=False,
                                                   download=False, transform=transform_val,
                                                   # target_transform=one_hot
                                                   )
            self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
            print('old class name: ', self.classes)
            print('old mapper:', trainset.class_to_idx)
            if step == 3:
                with open(config.DATASET.LE_PATH, 'rb') as f:
                    new_le = pickle.load(f)
                with open(config.DATASET.TRAIN_LIST, 'rb') as f:
                    train_list = pickle.load(f)
                with open(config.DATASET.VAL_LIST, 'rb') as f:
                    test_list = pickle.load(f)
                print(type(train_list), train_list)
                trainset.targets = new_le.transform(train_list.tolist())
                testset.targets = new_le.transform(test_list)
                print(new_le.mapper)
                print(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])))
                print(list(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])).keys()))
                self.classes = list(dict(sorted(new_le.mapper.items(), key=lambda item: item[1])).keys())
                print('new class name: ', self.classes)
                print('new mapper:', new_le.mapper)

            testset.targets = onehot(testset.targets)
            trainset.targets = onehot(trainset.targets)
            print(config.TRAIN.BATCH_SIZE)
            self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.TRAIN.BATCH_SIZE,
                                                            shuffle=True, num_workers=config.WORKERS)

            self.val_loader = torch.utils.data.DataLoader(testset, batch_size=config.TEST.BATCH_SIZE,
                                                          shuffle=False, num_workers=config.WORKERS)
        elif args.dataset == 'neu-cls':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transform_train = transforms.Compose([
                transforms.Resize((config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1]),
                                  interpolation=Image.NEAREST),
                transforms.ToTensor(),
                normalize
            ])

            transform_val = transforms.Compose([
                transforms.Resize((config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1]),
                                  interpolation=Image.NEAREST),
                transforms.ToTensor(),  # 3*H*W, [0, 1]
                normalize])
            if isinstance(config.DATASET.TRAIN_LIST, str):
                train_dataset = Dataset(
                    label_file=config.DATASET.TRAIN_LIST,
                    transform=transform_train, augment=None)
                le = train_dataset.le
                print(le.mapper)
                self.classes = list(le.mapper.values())

                val_dataset = Dataset(
                    label_file=config.DATASET.VAL_LIST,
                    transform=transform_val, augment=None, le=le)
            elif isinstance(config.DATASET.TRAIN_LIST, list):
                train_dataset = Dataset(
                    label_list=config.DATASET.TRAIN_LIST,
                    transform=transform_train, augment=None)
                le = train_dataset.le
                print(le.mapper)
                self.classes = list(le.mapper.values())

                val_dataset = Dataset(
                    label_list=config.DATASET.VAL_LIST,
                    transform=transform_val, augment=None, le=le)

            # Data loader
            self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                            batch_size=config.TRAIN.BATCH_SIZE,
                                                            num_workers=config.WORKERS,
                                                            shuffle=True)

            self.val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                          batch_size=config.TEST.BATCH_SIZE,
                                                          num_workers=config.WORKERS,
                                                          shuffle=False)
        # dataiter = iter(self.val_loader)
        # images, labels = dataiter.next()
        # print(labels)


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
