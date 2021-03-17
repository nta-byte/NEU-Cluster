import numpy as np
import torch.nn.functional as F
import torch


def onehot_transform(dataList_transformed):
    dataList_transformed = np.reshape(dataList_transformed, (-1, 1))
    target_transform = np.zeros((dataList_transformed.size, dataList_transformed.max() + 1))
    target_transform[np.arange(dataList_transformed.size), dataList_transformed] = 1
    return target_transform

def one_hot(array):
    unique, inverse = np.unique(array, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]
    return onehot

a = np.array([1, 2, 3, 4, 5, 6])
# a = np.reshape(a, 6)
print(a)
# a = np.reshape(a, 6, order='F')
# print(a)

# a = np.reshape(a, (-1, 1))
print(a)
print(one_hot(a))
x_data = torch.tensor(a)
a = F.one_hot(x_data)
print(a)
