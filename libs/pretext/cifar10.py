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
from libs.dataset.cifar10 import DataPreprocess2


# from training.utils.loader import onehot
# from libs.pretext.utils import get_data_list, load_images
# # from cluster_run import decrease_dim
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.metrics import *
# from hdbscan import HDBSCAN
# import umap


#
# def decrease_dim(args, fc1, data=None):
#     print(f'decrease_dim by {args.reduce_dimension}')
#     if args.reduce_dimension == 'umap':
#
#         x = umap.UMAP(
#             n_neighbors=200,
#             min_dist=0.0,
#             n_components=2,
#             # random_state=42,
#             metric='correlation',
#             init='random',
#         ).fit_transform(fc1,
#                         # y=y_gt
#                         )
#     elif args.reduce_dimension == 'pca':
#         pca = PCA(n_components=args.pca_component, svd_solver='full', whiten=True)
#         pca_nw = PCA(n_components=args.pca_component, svd_solver='full', whiten=False)
#         x = pca.fit_transform(fc1)
#         x_nw = pca_nw.fit_transform(fc1)
#         if not args.pca_whitten:
#             x = x_nw
#     # elif args.reduce_dimension == 'vae':
#     #     with open(args.vae_cfg, 'r') as file:
#     #         try:
#     #             vaeconfig = yaml.safe_load(file)
#     #         except yaml.YAMLError as exc:
#     #             print(exc)
#     #
#     #     print("VAE dim")
#     #     vaeconfig['exp_params']['train_data_path'] = args.fc1_path_vae
#     #     vaeconfig['logging_params']['save_dir'] = os.path.join(args.save_dir, vaeconfig['logging_params']['save_dir'])
#     #     # print(vaeconfig)
#     #     # print('0', vaeconfig['model_params']['hidden_dims'])
#     #     vaeconfig['infer']['weight_path'] = fit(vaeconfig)
#     #     # print('1', vaeconfig['infer']['weight_path'])
#     #     x = vae_reduce_dimension(vaeconfig, data)
#     elif args.reduce_dimension == 'none':
#         x = fc1
#     return x


class PretextCreater:
    def __init__(self, cfg, class_merging=False, renew_merge=False, add_noise=.1, shuffle_train=False,
                 dataset_part='train'):
        self.cfg = cfg
        self.renew_merge = renew_merge
        self.dataset_part = dataset_part
        self.config = self.cfg['master_model_params']
        self.dp = DataPreprocess2(self.cfg, class_merging=class_merging, renew_merge=renew_merge, add_noise=add_noise,
                                  shuffle_train=shuffle_train, dataset_part=dataset_part)
        self.train_loader = self.dp.train_loader
        self.val_loader = self.dp.val_loader
        self.org_labels = self.dp.org_labels
        self.new_labels = self.dp.new_labels
        self.original_le = self.dp.original_le
        self.new_le = self.dp.new_le

    def infer(self, net, dev):
        self.output = []
        net.eval()
        net = net.to(dev)
        if self.dataset_part == 'train' or self.dataset_part == 'train_test':
            for images, labels in self.train_loader:
                images = images.to(dev)
                with torch.no_grad():
                    out = net(images)
                    out = out.cpu().detach().numpy()
                    self.output.append(out)
        if self.dataset_part == 'test' or self.dataset_part == 'train_test':
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

        feature_dir = Path(self.cfg['pretext_params']['fc1_dir']).parent

        os.makedirs(feature_dir, exist_ok=True)
        with open(Path(self.cfg['pretext_params']['fc1_path']), 'wb') as f:
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

        feature_dir = Path(self.cfg['pretext_params']['fc1_dir']).parent

        os.makedirs(feature_dir, exist_ok=True)
        with open(Path(self.cfg['pretext_params']['reduce_dimension_params']['vae_params']['exp_params'][
                           'train_data_path']), 'wb') as f:
            pickle.dump(results, f)

# class DataPreprocess:
#     def __init__(self, cfg, class_merging=False, renew_merge=False, add_noise=.1, shuffle_train=False,
#                  dataset_part='train'):
#         self.cfg = cfg
#         self.renew_merge = renew_merge
#         self.dataset_part = dataset_part
#         self.config = self.cfg['master_model_params']
#         normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#         transform_train = transforms.Compose([
#             transforms.Resize((self.config.TRAIN.IMAGE_SIZE[0], self.config.TRAIN.IMAGE_SIZE[1])),
#             transforms.RandomCrop(self.config.TRAIN.IMAGE_SIZE[0], padding=4),
#             transforms.RandomHorizontalFlip(),
#             # transforms.RandomVerticalFlip(),
#             transforms.ToTensor(),
#             normalize
#         ])
#         transform_val = transforms.Compose([
#             transforms.Resize((self.config.TRAIN.IMAGE_SIZE[0], self.config.TRAIN.IMAGE_SIZE[1])),
#             transforms.ToTensor(),  # 3*H*W, [0, 1]
#             normalize])
#         trainset = torchvision.datasets.CIFAR10(root=self.config.DATASET.ROOT, train=True,
#                                                 download=True, transform=transform_train,
#                                                 )
#
#         testset = torchvision.datasets.CIFAR10(root=self.config.DATASET.ROOT, train=False,
#                                                download=True, transform=transform_val,
#                                                )
#         self.original_le = CustomLabelEncoder()
#         self.original_le.mapper = trainset.class_to_idx
#
#         self.org_trainlabels = self.original_le.inverse_transform(trainset.targets)
#         self.org_testlabel = self.original_le.inverse_transform(testset.targets)
#         if self.dataset_part == 'train':
#             self.org_labels = self.org_trainlabels
#         elif self.dataset_part == 'test':
#             self.org_labels = self.org_testlabel
#         else:
#             self.org_labels = np.concatenate((self.org_trainlabels, self.org_testlabel))
#
#         if class_merging:
#             if self.renew_merge:
#                 self.label_transform = None
#             else:
#                 if os.path.isfile(self.cfg['pretext_params']['label_transform_path']):
#                     print(f"reload self.label_transform in {self.cfg['pretext_params']['label_transform_path']}")
#                     with open(self.cfg['pretext_params']['label_transform_path'], 'rb') as f:
#                         self.label_transform = pickle.load(f)
#                 else:
#                     self.label_transform = None
#
#             trainset, testset = self.random_class_merging(trainset, testset)
#
#         if add_noise != 0:
#             self.create_noise()
#
#         self.new_le = CustomLabelEncoder()
#         self.new_le.mapper = trainset.class_to_idx
#
#         self.new_trainlabels = self.new_le.inverse_transform(trainset.targets)
#         self.new_testlabel = self.new_le.inverse_transform(testset.targets)
#         if self.dataset_part == 'train':
#             self.new_labels = self.new_trainlabels
#         elif self.dataset_part == 'test':
#             self.new_labels = self.new_testlabel
#         else:
#             self.new_labels = np.concatenate((self.new_trainlabels, self.new_testlabel))
#         print('extract data set org label', set(self.org_labels))
#         print('extract data set new label', set(self.new_labels))
#
#         self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.config.TRAIN.BATCH_SIZE,
#                                                         shuffle=shuffle_train, num_workers=self.config.WORKERS)
#
#         self.val_loader = torch.utils.data.DataLoader(testset, batch_size=self.config.TEST.BATCH_SIZE,
#                                                       shuffle=False, num_workers=self.config.WORKERS)
#         self.classes = trainset.classes
#
#     def random_class_merging(self, trainset=None, testset=None):
#         class_to_idx = trainset.class_to_idx
#         le = CustomLabelEncoder()
#         le.mapper = class_to_idx
#         trainlabels = le.inverse_transform(trainset.targets)
#         testlabels = le.inverse_transform(testset.targets)
#         print("start to merge classes")
#         if self.label_transform is None:
#             print("create label_transform")
#             self.label_transform = {'converted': {}, 'new_classes': None, 'new_class_to_idx': {},
#                                     'new_le': CustomLabelEncoder()}
#             classes = list(class_to_idx.keys())
#             random.shuffle(classes)
#             mergepoint = int(len(classes) / 2)
#             new_classes1 = classes[:mergepoint]
#             new_classes2 = classes[mergepoint:]
#             self.label_transform['new_classes'] = [x1 + "_" + x2 for (x1, x2) in zip(new_classes1, new_classes2)]
#             new_classes_list = [[x1, x2] for (x1, x2) in zip(new_classes1, new_classes2)]
#             # print(new_classes_list)
#             for group_classes in new_classes_list:
#                 self.label_transform['converted'][group_classes[0]] = "_".join(group_classes)
#                 self.label_transform['converted'][group_classes[1]] = "_".join(group_classes)
#             for i, x in enumerate(self.label_transform['new_classes']):
#                 self.label_transform['new_class_to_idx'][x] = i
#
#             self.label_transform['new_le'].mapper = self.label_transform['new_class_to_idx']
#             with open(Path(self.cfg['pretext_params']['label_transform_path']), 'wb') as f:
#                 pickle.dump(self.label_transform, f)
#         # else:
#         new_trainlabels = [''] * len(trainlabels)
#         for iddd, x in enumerate(trainlabels):
#             new_trainlabels[iddd] = self.label_transform['converted'][x]
#         new_testlabels = [''] * len(testlabels)
#         for iddd, x in enumerate(testlabels):
#             new_testlabels[iddd] = self.label_transform['converted'][x]
#
#         # new_le = CustomLabelEncoder()
#         trainset.targets = self.label_transform['new_le'].transform(new_trainlabels)
#         print('check1', set(new_trainlabels))
#         trainset.class_to_idx = self.label_transform['new_class_to_idx']
#         trainset.classes = self.label_transform['new_classes']
#         testset.targets = self.label_transform['new_le'].transform(new_testlabels)
#         testset.class_to_idx = self.label_transform['new_class_to_idx']
#         testset.classes = self.label_transform['new_classes']
#         # print("self.label_transform", self.label_transform)
#         # print("self.args.label_transform_path", self.args.label_transform_path)
#         return trainset, testset
#
#     def create_noise(self):
#         pass
#
#     def infer(self, net, dev):
#         self.output = []
#         net.eval()
#         net = net.to(dev)
#         if self.dataset_part == 'train' or self.dataset_part == 'train_test':
#             for images, labels in self.train_loader:
#                 images = images.to(dev)
#                 with torch.no_grad():
#                     out = net(images)
#                     out = out.cpu().detach().numpy()
#                     self.output.append(out)
#         if self.dataset_part == 'test' or self.dataset_part == 'train_test':
#             for images, labels in self.val_loader:
#                 images = images.to(dev)
#                 with torch.no_grad():
#                     out = net(images)
#                     out = out.cpu().detach().numpy()
#                     self.output.append(out)
#         self.output = np.concatenate(self.output, axis=0)
#         if len(self.output.shape) > 2:
#             self.output = self.output.reshape((self.output.shape[0], self.output.shape[1]))
#
#     def evaluate(self, net, dev):
#         net.eval()
#         net = net.to(dev)
#         with torch.no_grad():
#             correct = 0
#             total = 0
#             pbar = tqdm(total=len(self.val_loader), desc='eval model')
#             for idx, batch in enumerate(self.val_loader):
#                 if idx >= len(self.val_loader):
#                     break
#                 images = batch[0].to(dev)
#                 targets = batch[1].to(dev)
#                 predicted = net(images)
#                 predicted = predicted.argmax(1)
#                 targets = targets.argmax(1)
#                 total += targets.size(0)
#                 correct += predicted.eq(targets).sum().item()
#                 pbar.update(1)
#
#         pbar.close()
#         acc = 100. * correct / total
#         print('total', total, 'correct', correct)
#         print('accuracy:', round(acc, 3))
#
#     def save_output(self):
#         results = {
#             # 'filename': self.files,
#             'features': self.output,
#             'org_labels': self.org_labels,
#             'new_labels': self.new_labels,
#             'original_le': self.original_le,
#             'new_le': self.new_le,
#             'layer_name': 'fc1'
#         }
#
#         feature_dir = Path(self.cfg['pretext_params']['fc1_dir']).parent
#
#         os.makedirs(feature_dir, exist_ok=True)
#         with open(Path(self.cfg['pretext_params']['fc1_path']), 'wb') as f:
#             pickle.dump(results, f)
#
#         # print(self.output.shape)
#
#     def save_pretext_for_vae(self):
#         results = {
#             # 'filename': self.files,
#             'features': self.output,
#             'org_labels': self.org_labels,
#             'new_labels': self.new_labels,
#             'original_le': self.original_le,
#             'new_le': self.new_le,
#             'layer_name': 'fc1'
#         }
#
#         feature_dir = Path(self.cfg['pretext_params']['fc1_dir']).parent
#
#         os.makedirs(feature_dir, exist_ok=True)
#         with open(Path(
#                 self.cfg['pretext_params']['reduce_dimension_params']['vae_params']['exp_params']['train_data_path']),
#                 'wb') as f:
#             pickle.dump(results, f)

# print(self.output.shape)
#
#
# class DataPreprocessFlow4:
#     def __init__(self, argus, config, add_noise=0, active_data='train', renew_noise=False, decrease_dim=False):
#         self.noise = add_noise
#         self.args = argus
#         self.active_data = active_data
#         self.renew_noise = renew_noise
#         self.decrease_dim = decrease_dim
#         # rs = np.random.RandomState(seed=self.args.seed)
#         # self.renew_merge = renew_merge
#         print(f"dataset: {self.args.dataset}")
#         normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#         transform_train = transforms.Compose([
#             transforms.Resize((config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1])),
#             transforms.RandomCrop(config.TRAIN.IMAGE_SIZE[0], padding=4),
#             transforms.RandomHorizontalFlip(),
#             # transforms.RandomVerticalFlip(),
#             transforms.ToTensor(),
#             normalize
#         ])
#
#         transform_val = transforms.Compose([
#             transforms.Resize((config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1])),
#             transforms.ToTensor(),  # 3*H*W, [0, 1]
#             normalize])
#         self.org_trainset = torchvision.datasets.CIFAR10(root=self.args.dataset_root, train=True,
#                                                          download=True, transform=transform_val,
#                                                          )
#
#         self.org_testset = torchvision.datasets.CIFAR10(root=self.args.dataset_root, train=False,
#                                                         download=True, transform=transform_val,
#                                                         )
#         self.original_le = CustomLabelEncoder()
#         self.original_le.mapper = self.org_trainset.class_to_idx
#
#         self.org_trainlabels = self.original_le.inverse_transform(self.org_trainset.targets)
#         self.org_testlabels = self.original_le.inverse_transform(self.org_testset.targets)
#
#         if add_noise != 0:
#             if self.renew_noise:
#                 self.label_transform = None
#             else:
#                 if os.path.isfile(self.args.label_transform_path):
#                     print(f'reload self.label_transform in {self.args.label_transform_path}')
#                     with open(self.args.label_transform_path, 'rb') as f:
#                         self.label_transform = pickle.load(f)
#                 else:
#                     self.label_transform = None
#
#             self.create_noise()
#
#         # if class_merging:
#
#         self.new_trainlabels = self.original_le.inverse_transform(self.org_trainset.targets)
#
#         train_len = int(len(self.org_trainset) * .8)
#         trainset, valset = torch.utils.data.random_split(self.org_trainset,
#                                                          [train_len, len(self.org_trainset) - train_len])
#         valset.dataset.transform = transform_val
#         trainset.dataset.transform = transform_train
#
#         self.org_trainloader = torch.utils.data.DataLoader(self.org_trainset, batch_size=config.TRAIN.BATCH_SIZE,
#                                                            shuffle=False, num_workers=config.WORKERS)
#
#         self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.TRAIN.BATCH_SIZE,
#                                                         shuffle=True, num_workers=config.WORKERS)
#         self.val_loader = torch.utils.data.DataLoader(valset, batch_size=config.TEST.BATCH_SIZE,
#                                                       shuffle=False, num_workers=config.WORKERS)
#         self.test_loader = torch.utils.data.DataLoader(self.org_testset, batch_size=config.TEST.BATCH_SIZE,
#                                                        shuffle=False, num_workers=config.WORKERS)
#         self.classes = self.org_trainset.classes
#
#     def create_noise(self):
#         # self.new_trainlabel = self.org_trainlabels
#         if self.label_transform is None:
#             self.label_transform = self.org_trainset.targets
#             self.label_transform = np.asarray(self.label_transform)
#             ix_size = int(self.noise * len(self.label_transform))
#             # print(ix_size)
#             ix = np.random.choice(len(self.label_transform), size=ix_size, replace=False)
#             # print(self.label_transform)
#             # print(ix)
#             b = self.label_transform[ix]
#             # print(b)
#             np.random.shuffle(b)
#             self.label_transform[ix] = b
#             with open(Path(self.args.label_transform_path), 'wb') as f:
#                 pickle.dump(self.label_transform, f)
#         self.org_trainset.targets = self.label_transform
#         # pass
#
#     def drop_class(self):
#         pass
#
#     def infer(self, net, dev):
#         self.output = []
#         net.eval()
#         net = net.to(dev)
#         if self.active_data == 'train':
#             for images, labels in self.org_trainloader:
#                 images = images.to(dev)
#                 with torch.no_grad():
#                     out = net(images)
#                     out = out.cpu().detach().numpy()
#                     self.output.append(out)
#         if self.active_data == 'test':
#             for images, labels in self.test_loader:
#                 images = images.to(dev)
#                 with torch.no_grad():
#                     out = net(images)
#                     out = out.cpu().detach().numpy()
#                     self.output.append(out)
#         self.output = np.concatenate(self.output, axis=0)
#         if len(self.output.shape) > 2:
#             self.output = self.output.reshape((self.output.shape[0], self.output.shape[1]))
#
#     def save_output(self):
#         if self.decrease_dim:
#             self.output = decrease_dim(self.args, self.output)
#         if self.active_data == 'train':
#             results = {
#                 'features': self.output,
#                 'org_trainlabels': self.org_trainlabels,
#                 'new_trainlabels': self.new_trainlabels,
#                 'original_le': self.original_le,
#                 'layer_name': 'fc1'
#             }
#         elif self.active_data == 'test':
#             results = {
#                 'features': self.output,
#                 'org_testlabels': self.org_testlabels,
#                 'original_le': self.original_le,
#                 'layer_name': 'fc1'
#             }
#         fc1_file_name = f'{self.args.model}_fc1_{self.active_data}_features_std.pickle'
#         fc1_path = os.path.join(self.args.fc1_dir, fc1_file_name)
#         feature_dir = Path(self.args.fc1_dir).parent
#
#         os.makedirs(feature_dir, exist_ok=True)
#         with open(Path(fc1_path), 'wb') as f:
#             pickle.dump(results, f)
#
#         # print(self.output.shape)
#
#     def save_pretext_for_vae(self):
#         results = {
#             # 'filename': self.files,
#             'features': self.output,
#             'org_trainlabels': self.org_trainlabels,
#             'new_trainlabels': self.new_trainlabels,
#             'original_le': self.original_le,
#             'layer_name': 'fc1'
#         }
#
#         feature_dir = Path(self.args.fc1_dir).parent
#
#         os.makedirs(feature_dir, exist_ok=True)
#         with open(Path(self.args.fc1_path_vae), 'wb') as f:
#             pickle.dump(results, f)

# print(self.output.shape)
