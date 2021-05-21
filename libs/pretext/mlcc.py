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
from libs.dataset.preprocess import get_list_files, data_preprocess
from libs.helper.classification_tools import CustomLabelEncoder
from training.utils.loader import onehot
from libs.pretext.utils import get_data_list, load_images
from training.utils.loader import MLCCDataset


def mlcc_data_preprocess(args):
    print("[mlcc_data_preprocess]args: ", args)
    train_file_path = os.path.join(args.dataset_root, "train")
    list_images = get_list_files(train_file_path)
    print(len(list_images))
    data_preprocess(list_images, 'train', output_dir=args.data_preprocess_path)

    valid_file_path = os.path.join(args.dataset_root, "valid")
    list_images = get_list_files(valid_file_path)
    print(len(list_images))
    data_preprocess(list_images, 'valid', output_dir=args.data_preprocess_path)

    test_file_path = os.path.join(args.dataset_root, "test")
    list_images = get_list_files(test_file_path)
    print(len(list_images))
    data_preprocess(list_images, 'test', output_dir=args.data_preprocess_path)


class DataPreprocess:
    def __init__(self, argus, class_merging=False, renew_merge=False):
        self.args = argus
        self.renew_merge = renew_merge
        print(f"dataset: {self.args.dataset}")
        print("[DataPreprocess]data_preprocess_path: ", os.path.exists(self.args.data_preprocess_path))
        if not os.path.exists(self.args.data_preprocess_path):
            mlcc_data_preprocess(self.args)
        self.original_le = CustomLabelEncoder()
        self.original_le.mapper = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, 'etc': 9}
        # self.class_to_idx = self.le.mapper
        self.files, self.org_labels = get_data_list(self.args, shuffle=False)
        print(len(self.files))
        if self.args.cluster_dataset == 'train_test':
            pass
        elif self.args.cluster_dataset == 'train':
            self.files = self.files[:950]
            self.org_labels = self.org_labels[:950]
        elif self.args.cluster_dataset == 'test':
            self.files = self.files[950:]
            self.org_labels = self.org_labels[950:]
        transform_pipeline = transforms.Compose([
            # transforms.Resize(min_img_size, interpolation=Image.NEAREST),
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
            dataset = MLCCDataset(imgList=self.files, dataList=self.new_labels, le=self.label_transform['new_le'],
                                  transform=transform_pipeline)
            self.classes = self.label_transform['new_classes']
        else:
            dataset = MLCCDataset(imgList=self.files, dataList=self.org_labels, le=self.original_le,
                                  transform=transform_pipeline)
            self.classes = self.label_transform['org_classes']

        print('extract data set org label', set(self.org_labels))
        print('extract data set new label', set(self.new_labels))

        # self.le.mapper = self.class_to_idx
        train_len = int(len(dataset) * .8)
        trainset, testset = torch.utils.data.random_split(dataset, [train_len, len(dataset) - train_len])

        # # Data loader
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
        # self.class_to_idx = self.label_transform['new_class_to_idx']
        # self.classes = self.label_transform['new_classes']
        self.new_le = self.label_transform['new_le']

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
        # print(self.output.shape)

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
        # print(len(self.labels))
        # print(len(self.files))

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
