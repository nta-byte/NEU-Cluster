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
    def __init__(self, argus, class_merging=False, renew_merge=False):
        self.args = argus
        self.renew_merge = renew_merge
        print(f"dataset: {self.args.dataset}")
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        self.transform = transforms.Compose([
            transforms.Resize((self.args.image_size, self.args.image_size)),
            transforms.ToTensor(),
            normalize
        ])
        trainset = torchvision.datasets.STL10(root=self.args.dataset_root, split='train',
                                              download=True, transform=self.transform,
                                              )
        testset = torchvision.datasets.STL10(root=self.args.dataset_root, split='test',
                                             download=True, transform=self.transform,
                                             )
        trainset.class_to_idx = {_class: i for i, _class in enumerate(trainset.classes)}
        testset.class_to_idx = {_class: i for i, _class in enumerate(testset.classes)}
        self.original_le = CustomLabelEncoder()
        self.original_le.mapper = trainset.class_to_idx

        self.org_trainlabels = self.original_le.inverse_transform(trainset.labels)
        self.org_testlabel = self.original_le.inverse_transform(testset.labels)
        if self.args.cluster_dataset == 'train':
            self.org_labels = self.org_trainlabels
        elif self.args.cluster_dataset == 'test':
            self.org_labels = self.org_testlabel
        else:
            self.org_labels = np.concatenate((self.org_trainlabels, self.org_testlabel))

        if class_merging:
            if self.renew_merge:
                self.label_transform = None
            else:
                if os.path.isfile(self.args.label_transform_path):
                    print(f'reload self.label_transform in {self.args.label_transform_path}')
                    with open(self.args.label_transform_path, 'rb') as f:
                        self.label_transform = pickle.load(f)
                else:
                    self.label_transform = None

            trainset, testset = self.random_class_merging(trainset, testset)

        self.new_le = CustomLabelEncoder()
        self.new_le.mapper = trainset.class_to_idx
        # print('class_to_idx', self.original_le.mapper)
        # print(trainset.classes)

        # self.le.fit(self.labels)
        self.new_trainlabels = self.new_le.inverse_transform(trainset.labels)
        self.new_testlabel = self.new_le.inverse_transform(testset.labels)
        if self.args.cluster_dataset == 'train':
            self.new_labels = self.new_trainlabels
        elif self.args.cluster_dataset == 'test':
            self.new_labels = self.new_testlabel
        else:
            self.new_labels = np.concatenate((self.new_trainlabels, self.new_testlabel))
        print('extract data set org label', set(self.org_labels))
        print('extract data set new label', set(self.new_labels))

        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.args.batch_size,
                                                        shuffle=False, num_workers=self.args.workers)
        self.val_loader = torch.utils.data.DataLoader(testset, batch_size=self.args.batch_size,
                                                      shuffle=False, num_workers=self.args.workers)
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
            with open(Path(self.args.label_transform_path), 'wb') as f:
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
            'org_labels': self.org_labels,
            'new_labels': self.new_labels,
            'original_le': self.original_le,
            'new_le': self.new_le,
            'layer_name': 'fc1'
        }

        feature_dir = Path(self.args.fc1_dir).parent

        os.makedirs(feature_dir, exist_ok=True)
        with open(Path(self.args.fc1_path), 'wb') as f:
            pickle.dump(results, f)

        # print(self.output.shape)

    def save_pretext_for_vae(self):
        results = {
            # 'filename': self.files,
            'features': self.output,
            'org_labels': self.org_labels,
            'new_labels': self.new_labels,
            'original_le': self.original_le,
            'new_le': self.new_le,
            'layer_name': 'fc1'
        }

        feature_dir = Path(self.args.fc1_dir).parent

        os.makedirs(feature_dir, exist_ok=True)
        with open(Path(self.args.fc1_path_vae), 'wb') as f:
            pickle.dump(results, f)

        # print(self.output.shape)
