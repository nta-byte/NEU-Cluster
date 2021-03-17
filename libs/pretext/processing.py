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


def infer(loader, net, dev):
    output = []
    net.eval()
    net = net.to(dev)
    for images, labels in loader:
        images = images.to(dev)
        with torch.no_grad():
            out = net(images)
            out = out.cpu().detach().numpy()
            output.append(out)
    output = np.concatenate(output, axis=0)
    if len(output.shape) > 2:
        output = output.reshape((output.shape[0], output.shape[1]))
    return output
