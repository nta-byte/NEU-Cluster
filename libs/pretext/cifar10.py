import numpy as np
import os
from pathlib import Path
import pickle
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
    def __init__(self, argus):
        self.args = argus
        print(f"dataset: {self.args.dataset}")
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize((self.args.image_size, self.args.image_size)),
            transforms.ToTensor(),
            normalize
        ])
        # target_transform =

        # print(b)

        trainset = torchvision.datasets.CIFAR10(root='/data4T/ntanh/data/', train=False,
                                                download=False, transform=transform,
                                                # target_transform=one_hot
                                                )

        self.loader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                                  shuffle=False, num_workers=5)

        print(trainset.class_to_idx)
        self.le = CustomLabelEncoder()
        self.le.mapper = trainset.class_to_idx
        # self.le.fit(self.labels)
        self.labels = self.le.inverse_transform(trainset.targets)
        # trainset.targets = one_hot(trainset.targets)
        # print(config.TRAIN.BATCH_SIZE)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        with open(self.args.le_path, 'wb') as f:
            pickle.dump(self.le, f)

        testset = torchvision.datasets.CIFAR10(root='/data4T/ntanh/data/', train=False,
                                               download=False, transform=transform,
                                               # target_transform=one_hot
                                               )
        testset.targets = onehot(testset.targets)
        self.val_loader = torch.utils.data.DataLoader(testset, batch_size=32,
                                                      shuffle=False, num_workers=5)

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
            # 'filename': self.files,
            'features': self.output,
            'labels': self.labels,
            'layer_name': 'fc1'
        }

        feature_dir = Path(self.args.fc1_dir).parent

        os.makedirs(feature_dir, exist_ok=True)
        with open(Path(self.args.fc1_path), 'wb') as f:
            pickle.dump(results, f)

        print(self.output.shape)
