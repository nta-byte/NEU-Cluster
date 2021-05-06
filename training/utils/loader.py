import os
import cv2
import torch
# from torch.utils.data import Dataset
from torchvision.transforms.functional import pad
import numbers
from pathlib import Path
from PIL import Image
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
from sklearn import preprocessing
from libs.dataset.preprocess import get_list_files
from libs.helper.classification_tools import CustomLabelEncoder


def onehot(array):
    unique, inverse = np.unique(array, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]
    return onehot


class ImgAugTransform:
    def __init__(self):
        sometimes = lambda aug: iaa.Sometimes(0.3, aug)

        self.aug = iaa.Sequential(
            iaa.SomeOf(
                (1, 5), [
                    # blur
                    sometimes(iaa.OneOf([
                        iaa.GaussianBlur(sigma=(0, 1.0)),
                        iaa.MotionBlur(k=3),
                        iaa.AdditiveGaussianNoise(scale=(0, .1 * 255)),
                        iaa.AdditiveLaplaceNoise(scale=(0, .1 * 255))
                    ])),
                    # color
                    sometimes(
                        iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)),
                    sometimes(iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6),
                                                  per_channel=True)),
                    sometimes(iaa.Invert(0.25, per_channel=0.5)),
                    sometimes(iaa.Solarize(0.5, threshold=(32, 128))),
                    sometimes(iaa.Dropout2d(p=0.5)),
                    sometimes(iaa.Multiply((0.5, 1.5), per_channel=0.5)),
                    sometimes(iaa.Add((-40, 40), per_channel=0.5)),

                    sometimes(iaa.JpegCompression(compression=(5, 80))),

                    # distort
                    sometimes(iaa.Crop(percent=(0.01, 0.05), sample_independently=True)),
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.01))),
                    sometimes(iaa.Affine(scale=(0.7, 1.3), translate_percent=(-0.1, 0.1),
                                         #                            rotate=(-5, 5), shear=(-5, 5),
                                         order=[0, 1], cval=(0, 255),
                                         mode=ia.ALL)),
                    sometimes(iaa.ElasticTransformation(alpha=50, sigma=20)),

                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.02))),
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.01))),
                    sometimes(iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                         iaa.CoarseDropout(p=(0, 0.1),
                                                           size_percent=(0.02, 0.25))])),
                ],
                random_order=True),
            random_order=True)

    def __call__(self, img_):
        if isinstance(img_, Image.Image):
            img_ = np.array(img_)
        img_ = self.aug.augment_image(img_)
        img_ = Image.fromarray(img_)
        return img_


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir=None, transform=None, augment=None, le=None, label_file=None, label_list=None):
        self.transform = transform
        self.augment = augment
        self.imgList = []
        self.dataList = []
        if label_file:
            lf = open(label_file, 'r')
            for line in lf:
                sep = line.rstrip('\n').split(',')
                img_path = sep[0]
                if os.path.exists(img_path):
                    ldata = sep[1]
                    # ldata = list(map(int, ldata))
                    self.imgList.append(img_path)
                    self.dataList.append(ldata)
        elif label_list:
            for line in label_list:
                sep = line
                img_path = sep[0]
                if os.path.exists(img_path):
                    ldata = sep[1]
                    self.imgList.append(img_path)
                    self.dataList.append(ldata)
        else:
            lf = get_list_files(Path(data_dir))
            for line in lf:
                # sep = line.rstrip('\n').split(',')
                # img_path = sep[0]
                # name = inpath.parent.name + '_' + name
                if os.path.exists(line):
                    ldata = line.parent.name
                    # print(name)
                    # ldata = name.stem.split('_')[0]
                    # ldata = list(map(int, ldata))
                    self.imgList.append(str(line))
                    self.dataList.append(ldata)
        print(self.dataList)
        if le is None:
            self.le = CustomLabelEncoder()
            self.le.fit(self.dataList)
        else:
            self.le = le
        self.dataList_transformed = self.le.transform(self.dataList)
        # print(list(self.le.classes_))
        # print(self.dataList_transformed)
        b = np.zeros((self.dataList_transformed.size, self.dataList_transformed.max() + 1))
        b[np.arange(self.dataList_transformed.size), self.dataList_transformed] = 1
        # print(b)
        self.dataList_transformed = b

    def __getitem__(self, index):
        imgpath = self.imgList[index]
        target = self.dataList_transformed[index]

        img = Image.open(imgpath).convert('RGB')
        if self.augment is not None:
            img = self.augment(np.array(img))
            # img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        # print(img.shape)

        target = np.argmax(np.array(target).astype(np.long))
        return img, target

    def __len__(self):
        return len(self.imgList)

    def get_classNum(self):
        return len(self.dataList[0])


class NEU_Dataset(torch.utils.data.Dataset):
    def __init__(self, imgList=None, dataList=None, transform=None, augment=None, le=None, label_file=None,
                 label_list=None):
        self.transform = transform
        self.augment = augment
        self.imgList = imgList
        self.dataList = dataList

        # print(self.dataList)
        if le is None:
            self.le = CustomLabelEncoder()
            self.le.fit(self.dataList)
        else:
            self.le = le
        self.dataList_transformed = self.le.transform(self.dataList)
        # print(list(self.le.classes_))
        # print(self.dataList_transformed)
        # b = np.zeros((self.dataList_transformed.size, self.dataList_transformed.max() + 1))
        # b[np.arange(self.dataList_transformed.size), self.dataList_transformed] = 1
        # print(self.dataList_transformed)
        # self.dataList_transformed = b

    def __getitem__(self, index):
        imgpath = self.imgList[index]
        target = self.dataList_transformed[index]

        img = Image.open(imgpath).convert('RGB')
        if self.augment is not None:
            img = self.augment(np.array(img))
            # img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        # print(img.shape)

        target = np.array(target).astype(np.long)

        return img, target

    def __len__(self):
        return len(self.imgList)

    def get_classNum(self):
        return len(self.dataList[0])


class MLCCDataset(torch.utils.data.Dataset):
    def __init__(self, imgList=None, dataList=None, transform=None, augment=None, le=None, label_file=None,
                 label_list=None):
        self.transform = transform
        self.augment = augment
        self.imgList = imgList
        self.dataList = dataList

        # print(self.dataList)
        if le is None:
            self.le = CustomLabelEncoder()
            self.le.fit(self.dataList)
        else:
            self.le = le
        self.dataList_transformed = self.le.transform(self.dataList)

    def __getitem__(self, index):
        imgpath = self.imgList[index]
        target = self.dataList_transformed[index]

        img = Image.open(imgpath).convert('RGB')
        if self.augment is not None:
            img = self.augment(np.array(img))
            # img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        # print(img.shape)

        target = np.array(target).astype(np.long)

        return img, target

    def __len__(self):
        return len(self.imgList)

    def get_classNum(self):
        return len(self.dataList[0])


def get_img_paths(dir_, extensions=('.jpg', '.png', '.jpeg', '.PNG', '.JPG', '.JPEG')):
    img_paths = []
    if type(dir_) is list:
        for d in dir_:
            for root, dirs, files in os.walk(d):
                for file in files:
                    for e in extensions:
                        if file.endswith(e):
                            p = os.path.join(root, file)
                            img_paths.append(p)
    else:
        for root, dirs, files in os.walk(dir_):
            for file in files:
                for e in extensions:
                    if file.endswith(e):
                        p = os.path.join(root, file)
                        img_paths.append(p)
    return img_paths


class NewPad(object):
    def __init__(self, t_size=(64, 192), fill=(255, 255, 255), padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode
        self.t_size = t_size

    def __call__(self, img):
        # def __call__(self, img, t_size):
        target_h, target_w = self.t_size
        # h, w, c = img.shape
        w, h = img.size

        im_scale = h / w
        target_scale = target_h / target_w
        # print(im_scale, target_scale)
        if im_scale < target_scale:
            # keep w, add padding h
            new_w = int(round(target_h / im_scale))
            # new_w =
            out_im = img.resize((new_w, target_h))
            # out_im = img
        else:
            # keep h, add padding w
            new_w = h / target_scale
            _pad = (new_w - w) / 2
            _pad = int(round(_pad))
            padding = (_pad, 0, _pad, 0)  # left, top, right and bottom
            # padding = (0, _pad, 0, _pad)  # left, top, right and bottom
            out_im = pad(img, padding, self.fill, self.padding_mode)
            out_im = out_im.resize((self.t_size[1], self.t_size[0]))
        # print(img.size, out_im.size)
        return out_im

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'. \
            format(self.fill, self.padding_mode)


def resize_image(im, size, padding=True, border=cv2.BORDER_CONSTANT, color=[0, 0, 255]):
    # image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    target_w, target_h = size
    h, w, c = im.shape

    im_scale = h / w
    target_scale = target_h / target_w

    if im_scale < target_scale:
        # keep w, add padding h
        new_h = w * target_scale
        pad = (new_h - h) / 2
        pad = int(pad)
        constant = cv2.copyMakeBorder(im, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=color)
    elif im_scale > target_scale:
        # keep h, add padding w
        new_w = h / target_scale
        pad = (new_w - w) / 2
        pad = int(pad)
        constant = cv2.copyMakeBorder(im, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=color)

    image = cv2.resize(constant, size, interpolation=cv2.INTER_LINEAR)
    return image


if __name__ == '__main__':
    im_dir = '/home/ntanh/ntanh/MC_OCR/text_generator/output/train_1/'
    list_img = get_img_paths(im_dir)
    out_img_dir = '/home/ntanh/ntanh/MC_OCR/text_generator/output/train_'
    # im_path = '/home/ntanh/ntanh/MC_OCR/text_generator/output/test_/23.png'
    # img = Image.open(im_path)
    a = NewPad()
    # im = a(img)
    # im.save('/home/ntanh/ntanh/MC_OCR/text_generator/output/a.png')
    for im_path in list_img:
        img = Image.open(im_path)
        im = a(img)
        im.save(os.path.join(out_img_dir, os.path.basename(im_path)))
