import numpy as np
import os
from pathlib import Path
import pickle
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import torch
from libs.helper.classification_tools import CustomLabelEncoder
from libs.utils.yaml_config import init
from libs.dataset.preprocess import get_list_files

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Dataset(torch.utils.data.Dataset):
    def __init__(self, listfile, transform=None):
        self.transform = transform
        self.imgList = listfile
        self.dataList = []

    def __getitem__(self, index):
        imgpath = self.imgList[index]
        target = 0

        img = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        target = np.array(target).astype(np.float32)

        return img, target

    def __len__(self):
        return len(self.imgList)

    def get_classNum(self):
        return len(self.dataList[0])


def load_images(paths, min_img_size=224):
    # The min size, as noted in the PyTorch pretrained models doc, is 224 px.
    transform_pipeline = transforms.Compose([
        # transforms.Resize(min_img_size, interpolation=Image.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    train_dataset = Dataset(paths, transform_pipeline)
    # Data loader
    loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                         batch_size=16,
                                         num_workers=10,
                                         shuffle=False)
    return loader


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


def extract_labels(f):
    return [x.stem.split('_')[0] for x in f]


def load_state(weight_path, net):
    pretrained_dict = torch.load(weight_path)
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    return net


def get_model(args):
    if args.model == 'RESNET50':
        model = models.resnet.resnet50(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
    elif args.model == 'VGG16':
        vgg16_path = '/home/ntanh/.cache/torch/checkpoints/vgg16-397923af.pth'
        model = models.vgg16(pretrained=True)
        model.classifier = nn.Sequential(*(list(model.classifier.children())[:1]))
        model = load_state(vgg16_path, model)
    return model


def extract_feature(args, logging):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.use_histeq:
        args.data_preprocess_path = os.path.join(args.data_preprocess_path, 'images_histeq_resize')
    else:
        args.data_preprocess_path = os.path.join(args.data_preprocess_path, 'images_resize')
    img_root = Path(args.data_preprocess_path)  # directory where images are stored.
    files = get_list_files(img_root)  # returns a list of all of the images in the directory, sorted by filename.
    files = sorted(files)  # returns a list of all of the images in the directory, sorted by filename.
    print(files)
    # ## Shuffle the filenames so they appear randomly in the dataset.
    rs = np.random.RandomState(seed=749976)
    rs.shuffle(files)

    labels = extract_labels(files)
    print('first 10 labels: {}'.format(labels[:10]))

    le = CustomLabelEncoder()
    le.fit(labels, sorter=lambda x: x.upper())

    labels_int = le.transform(labels[:10])
    labels_str = le.inverse_transform(labels_int)

    # save the label encoder so it can be used throughout the rest of this study
    with open(args.le_path, 'wb') as f:
        pickle.dump(le, f)

    print('label encodings: {}'.format(le.mapper))
    print('first 10 integer labels: {}'.format(labels_int))
    print('first 10 string labels: {}'.format(labels_str))
    loader = load_images(files, args.image_size)
    print(len(loader))

    model = get_model(args)

    print(model)
    fc1 = infer(loader, model, device)
    print(fc1.shape)
    # save results
    results = {'filename': files,
               'features': fc1,
               'labels': labels,
               'layer_name': 'fc1'
               }

    feature_dir = Path(args.fc1_path).parent

    os.makedirs(feature_dir, exist_ok=True)
    with open(Path(args.fc1_path), 'wb') as f:
        pickle.dump(results, f)

    print(fc1.shape)


if __name__ == '__main__':
    args, logging = init("config/config.yaml")
    extract_feature(args, logging)
