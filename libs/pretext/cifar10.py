import numpy as np
import os
import time
from tqdm import tqdm
from pathlib import Path
import pickle
import random
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
from libs.pretext.utils import get_data_list, load_images


class DataPreprocess:
    def __init__(self, argus, class_merging=False):
        self.args = argus
        print(f"dataset: {self.args.dataset}")
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        self.transform = transforms.Compose([
            transforms.Resize((self.args.image_size, self.args.image_size)),
            transforms.ToTensor(),
            normalize
        ])
        trainset = torchvision.datasets.CIFAR10(root=self.args.dataset_root, train=True,
                                                download=True, transform=self.transform,
                                                )

        testset = torchvision.datasets.CIFAR10(root=self.args.dataset_root, train=False,
                                               download=True, transform=self.transform,
                                               )
        if class_merging:
            trainset, testset = self.random_class_merging(trainset, testset)
        print('class_to_idx', trainset.class_to_idx)
        # print(trainset.classes)
        self.le = CustomLabelEncoder()
        self.le.mapper = trainset.class_to_idx
        # self.le.fit(self.labels)
        self.trainlabels = self.le.inverse_transform(trainset.targets)
        self.testlabel = self.le.inverse_transform(testset.targets)
        if self.args.cluster_dataset == 'train':
            self.labels = self.trainlabels
        elif self.args.cluster_dataset == 'test':
            self.labels = self.testlabel
        else:
            self.labels = np.concatenate((self.trainlabels, self.testlabel))
        # trainset.targets = onehot(trainset.targets)
        # testset.targets = onehot(testset.targets)

        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.args.batch_size,
                                                        shuffle=False, num_workers=self.args.workers)
        self.val_loader = torch.utils.data.DataLoader(testset, batch_size=self.args.batch_size,
                                                      shuffle=False, num_workers=self.args.workers)
        self.classes = trainset.classes

    def random_class_merging(self, trainset=None, testset=None):
        # print(trainset.class_to_idx)
        # print(trainset.targets)
        class_to_idx = trainset.class_to_idx
        le = CustomLabelEncoder()
        le.mapper = trainset.class_to_idx
        trainlabels = le.inverse_transform(trainset.targets)
        testlabels = le.inverse_transform(testset.targets)
        # print(trainlabels)
        classes = list(class_to_idx.keys())
        random.shuffle(classes)
        mergepoint = int(len(classes) / 2)
        # print(mergepoint)
        new_classes1 = classes[:mergepoint]
        new_classes2 = classes[mergepoint:]
        new_classes = [x1 + "_" + x2 for (x1, x2) in zip(new_classes1, new_classes2)]
        new_classes_list = [[x1, x2] for (x1, x2) in zip(new_classes1, new_classes2)]
        # print('new_classes_list', new_classes_list)
        # print(new_classes2)
        # print('new_classes', new_classes)

        new_trainlabels = [''] * len(trainlabels)
        for iddd, x in enumerate(trainlabels):
            for idxxxx, new_x in enumerate(new_classes_list):
                if x in new_x:
                    new_trainlabels[iddd] = new_classes[idxxxx]

        new_testlabels = [''] * len(testlabels)
        for iddd, x in enumerate(testlabels):
            for idxxxx, new_x in enumerate(new_classes_list):
                if x in new_x:
                    new_testlabels[iddd] = new_classes[idxxxx]
        # print('set(trainlabels)', set(trainlabels))
        # print('set(new_trainlabels)', set(new_trainlabels))
        new_class_to_idx = {}
        for i, x in enumerate(new_classes):
            new_class_to_idx[x] = i
        # print(new_class_to_idx)
        new_le = CustomLabelEncoder()
        new_le.mapper = new_class_to_idx
        trainset.targets = new_le.transform(new_trainlabels)
        trainset.class_to_idx = new_class_to_idx
        trainset.classes = new_classes
        testset.targets = new_le.transform(new_testlabels)
        testset.class_to_idx = new_class_to_idx
        testset.classes = new_classes
        return trainset, testset

    def infer(self, net, dev):
        self.output = []
        net.eval()
        net = net.to(dev)
        if self.args.cluster_dataset == 'train' or self.args.cluster_dataset == 'train_test':
            for images, labels in self.train_loader:
                images = images.to(dev)
                with torch.no_grad():
                    out = net(images)
                    out = out.cpu().detach().numpy()
                    self.output.append(out)
        if self.args.cluster_dataset == 'test' or self.args.cluster_dataset == 'train_test':
            for images, labels in self.val_loader:
                images = images.to(dev)
                with torch.no_grad():
                    out = net(images)
                    out = out.cpu().detach().numpy()
                    self.output.append(out)
        self.output = np.concatenate(self.output, axis=0)
        if len(self.output.shape) > 2:
            self.output = self.output.reshape((self.output.shape[0], self.output.shape[1]))

    def evaluate(self, net, dev):
        net.eval()
        net = net.to(dev)
        with torch.no_grad():
            correct = 0
            total = 0
            pbar = tqdm(total=len(self.val_loader), desc='eval model')
            for idx, batch in enumerate(self.val_loader):
                if idx >= len(self.val_loader):
                    break
                images = batch[0].to(dev)
                targets = batch[1].to(dev)
                predicted = net(images)
                predicted = predicted.argmax(1)
                targets = targets.argmax(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                pbar.update(1)

        pbar.close()
        acc = 100. * correct / total
        print('total', total, 'correct', correct)
        print('accuracy:', round(acc, 3))

    def save_output(self):
        results = {
            # 'filename': self.files,
            'features': self.output,
            'labels': self.labels,
            'le': self.le,
            'layer_name': 'fc1'
        }

        feature_dir = Path(self.args.fc1_dir).parent

        os.makedirs(feature_dir, exist_ok=True)
        with open(Path(self.args.fc1_path), 'wb') as f:
            pickle.dump(results, f)

        print(self.output.shape)
