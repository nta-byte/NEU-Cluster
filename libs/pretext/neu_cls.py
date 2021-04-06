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

from libs.dataset.preprocess import get_list_files
from libs.helper.classification_tools import CustomLabelEncoder
from training.utils.loader import onehot
from libs.pretext.utils import get_data_list, load_images, Dataset
from training.utils.loader import NEU_Dataset


class DataPreprocess:
    def __init__(self, argus, class_merging=False):
        self.args = argus
        print(f"dataset: {self.args.dataset}")
        self.files, self.labels = get_data_list(self.args)
        # files = sorted(files)  # returns a list of all of the images in the directory, sorted by filename.
        # print(files)
        print('first 10 files: {}'.format(self.files[:10]))
        print('first 10 labels: {}'.format(self.labels[:10]))

        self.le = CustomLabelEncoder()
        self.le.fit(self.labels)
        print(self.le.mapper)

        labels_int = self.le.transform(self.labels[:10])
        labels_str = self.le.inverse_transform(labels_int)

        print('label encodings: {}'.format(self.le.mapper))
        print('first 10 integer labels: {}'.format(labels_int))
        print('first 10 string labels: {}'.format(labels_str))
        transform_pipeline = transforms.Compose([
            # transforms.Resize(min_img_size, interpolation=Image.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        # dataset = NEU_Dataset(imgList=self.files, dataList=self.labels, le=self.le, transform=transform_pipeline)
        if class_merging:
            self.random_class_merging()
        # else:
        dataset = NEU_Dataset(imgList=self.files, dataList=self.labels, le=self.le, transform=transform_pipeline)

        # self.classes = dataset.classes
        self.le.mapper = self.class_to_idx
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

        dataiter = iter(self.val_loader)
        images, labels = dataiter.next()
        print(labels)

    def random_class_merging(self, trainset=None):
        class_to_idx = self.le.mapper
        # le = self.le
        # le.mapper = trainset.class_to_idx
        trainlabels = self.labels
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

        new_class_to_idx = {}
        for i, x in enumerate(new_classes):
            new_class_to_idx[x] = i
        # print(new_class_to_idx)
        new_le = CustomLabelEncoder()
        new_le.mapper = new_class_to_idx
        # self.labels = new_le.transform(new_trainlabels)
        self.labels = new_trainlabels
        self.class_to_idx = new_class_to_idx
        self.classes = new_classes
        self.le = new_le
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
            'labels': self.labels,
            'le': self.le,
            'layer_name': 'fc1'
        }

        feature_dir = Path(self.args.fc1_dir).parent

        os.makedirs(feature_dir, exist_ok=True)
        with open(Path(self.args.fc1_path), 'wb') as f:
            pickle.dump(results, f)

        print(self.output.shape)
