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
        self.loader = load_images(self.files, argus.image_size)

        with open(self.args.le_path, 'wb') as f:
            pickle.dump(self.le, f)

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
            'layer_name': 'fc1'
        }

        feature_dir = Path(self.args.fc1_dir).parent

        os.makedirs(feature_dir, exist_ok=True)
        with open(Path(self.args.fc1_path), 'wb') as f:
            pickle.dump(results, f)

        print(self.output.shape)
