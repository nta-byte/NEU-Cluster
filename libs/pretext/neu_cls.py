import numpy as np
import os
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

from libs.dataset.preprocess import get_list_files, data_preprocess
from libs.helper.classification_tools import CustomLabelEncoder
from training.utils.loader import onehot
from libs.pretext.utils import get_data_list, load_images, Dataset
from training.utils.loader import NEU_Dataset


def neu_data_preprocess(args):
    list_images = get_list_files(args.dataset_root)
    print(len(list_images))
    data_preprocess(list_images, 'neu-cls', args.data_preprocess_path)


class DataPreprocess:
    def __init__(self, argus, class_merging=False, renew_merge=False):
        self.args = argus
        self.renew_merge = renew_merge
        print(f"dataset: {self.args.dataset}")
        print("[DataPreprocess]data_preprocess_path: ", os.path.exists(self.args.data_preprocess_path))
        if not os.path.exists(self.args.data_preprocess_path):
            neu_data_preprocess(self.args)
        self.files, self.org_labels = get_data_list(self.args)

        self.original_le = CustomLabelEncoder()
        self.original_le.mapper = {'Cr': 0, 'In': 1, 'PS': 2, 'Pa': 3, 'RS': 4, 'Sc': 5}
        # self.original_le.fit(self.org_labels)
        # print(self.original_le.mapper)
        self.class_to_idx = self.original_le.mapper
        self.classes = list(self.original_le.mapper.keys())
        split_len = int(len(self.files) * .5)
        if self.args.cluster_dataset == 'train_test':
            pass
        elif self.args.cluster_dataset == 'train':
            self.files = self.files[:split_len]
            self.org_labels = self.org_labels[:split_len]
        elif self.args.cluster_dataset == 'test':
            self.files = self.files[split_len:]
            self.org_labels = self.org_labels[split_len:]

        print('label encodings: {}'.format(self.original_le.mapper))
        transform_pipeline = transforms.Compose([
            transforms.Resize((self.args.image_size, self.args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

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

            self.random_class_merging()
            dataset = NEU_Dataset(imgList=self.files, dataList=self.new_labels, le=self.label_transform['new_le'],
                                  transform=transform_pipeline)
            self.classes = self.label_transform['new_classes']
            print('extract data set new label', set(self.new_labels))
        else:
            dataset = NEU_Dataset(imgList=self.files, dataList=self.org_labels, le=self.original_le,
                                  transform=transform_pipeline)
            self.new_labels = self.org_labels
            self.new_le = self.original_le
        # if class_merging:
        #     self.random_class_merging()
        # else:

        # self.classes = dataset.classes
        # self.le.mapper = self.class_to_idx
        # print(dataset.labels, self.le.mapper)
        train_len = int(len(dataset) * .8)
        trainset, testset = torch.utils.data.random_split(dataset, [train_len, len(dataset) - train_len])
        # Data loader
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.args.batch_size,
                                                        shuffle=True, num_workers=self.args.workers)

        self.val_loader = torch.utils.data.DataLoader(testset, batch_size=self.args.batch_size,
                                                      shuffle=False, num_workers=self.args.workers)

        self.loader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size,
                                                  shuffle=False, num_workers=self.args.workers)

    def random_class_merging(self, trainset=None):
        class_to_idx = self.original_le.mapper
        labels = self.org_labels
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
        new_labels = [''] * len(labels)
        for iddd, x in enumerate(labels):
            new_labels[iddd] = self.label_transform['converted'][x]

        self.new_labels = new_labels
        print(self.label_transform['new_classes'])
        self.new_le = self.label_transform['new_le']
        # classes = list(class_to_idx.keys())
        # random.shuffle(classes)
        # mergepoint = int(len(classes) / 2)
        # new_classes1 = classes[:mergepoint]
        # new_classes2 = classes[mergepoint:]
        # new_classes = [x1 + "_" + x2 for (x1, x2) in zip(new_classes1, new_classes2)]
        # new_classes_list = [[x1, x2] for (x1, x2) in zip(new_classes1, new_classes2)]
        #
        # new_trainlabels = [''] * len(trainlabels)
        # for iddd, x in enumerate(trainlabels):
        #     for idxxxx, new_x in enumerate(new_classes_list):
        #         if x in new_x:
        #             new_trainlabels[iddd] = new_classes[idxxxx]
        #
        # new_class_to_idx = {}
        # for i, x in enumerate(new_classes):
        #     new_class_to_idx[x] = i
        # # print(new_class_to_idx)
        # new_le = CustomLabelEncoder()
        # new_le.mapper = new_class_to_idx
        # self.labels = new_trainlabels
        # self.class_to_idx = new_class_to_idx
        # self.classes = new_classes
        # self.le = new_le
        # return trainset

    def infer(self, net, dev):
        self.output = []
        net.eval()
        net = net.to(dev)
        for images, labels in self.loader:
            images = images.to(dev)
            with torch.no_grad():
                out = net(images)
                out = out.cpu().detach().numpy()
                self.output.append(out)
        self.output = np.concatenate(self.output, axis=0)
        if len(self.output.shape) > 2:
            self.output = self.output.reshape((self.output.shape[0], self.output.shape[1]))

    def save_output(self):
        results = {
            'filename': self.files,
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
