import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
# from training.utils.loader import DataLoader
from training.utils.loader import Dataset, onehot
# from training.utils.loader import onehot
import torch
import torchvision
import numpy as np
from PIL import Image


class DataPreprocess:
    def __init__(self, config, args):
        if args.dataset == 'mlcc':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transform_train = transforms.Compose([
                # transforms.ColorJitter(brightness=(0.2, 2),
                #                        contrast=(0.3, 2),
                #                        saturation=(0.2, 2),
                #                        hue=(-0.3, 0.3)),
                transforms.Resize((config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1]),
                                  interpolation=Image.NEAREST),
                transforms.ToTensor(),
                normalize
            ])

            transform_val = transforms.Compose([
                transforms.Resize((config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1]),
                                  interpolation=Image.NEAREST),
                # transforms.ColorJitter(brightness=(0.2, 2),
                #                        contrast=(0.3, 2),
                #                        saturation=(0.2, 2),
                #                        hue=(-0.3, 0.3)),
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

            transform = transforms.Compose([
                transforms.Resize((config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            # target_transform =

            # print(b)

            trainset = torchvision.datasets.CIFAR10(root='/data4T/ntanh/data/', train=True,
                                                    download=False, transform=transform,
                                                    # target_transform=one_hot
                                                    )
            trainset.targets = onehot(trainset.targets)
            print(config.TRAIN.BATCH_SIZE)
            self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.TRAIN.BATCH_SIZE,
                                                            shuffle=True, num_workers=config.WORKERS)
            self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

            testset = torchvision.datasets.CIFAR10(root='/data4T/ntanh/data/', train=False,
                                                   download=False, transform=transform,
                                                   # target_transform=one_hot
                                                   )
            testset.targets = onehot(testset.targets)
            self.val_loader = torch.utils.data.DataLoader(testset, batch_size=config.TEST.BATCH_SIZE,
                                                          shuffle=False, num_workers=config.WORKERS)
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
