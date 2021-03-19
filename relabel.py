import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import csv
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import *

from kneed import KneeLocator

from libs.helper import classification_tools as ct
from libs.helper.visualize import visual
from libs.utils.yaml_config import init
from create_pretext_pytorch import extract_feature


def write_csv(file_name, data_out):
    with open(file_name, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for line in data_out:
            date = line['file_path']
            value = line['label']
            writer.writerow([date, value])


def relabel(kmeans, labels_unmatched_, y_gt, le):
    y_pred_ = ct.label_matcher(labels_unmatched_, y_gt)
    cluster_mapper = {}
    for p in np.unique(y_pred_):
        y_clusters = kmeans.labels_[y_pred_ == p]
        for idx, value in enumerate(np.unique(y_clusters)):
            cluster_mapper['{}-{}'.format(le.inverse_transform([p])[0], idx)] = value
    new_le = ct.CustomLabelEncoder()
    new_le.update_mapper(cluster_mapper)
    y_pred_2_label = new_le.inverse_transform(labels_unmatched_)
    out_dict_train = []
    out_dict_test = []
    out_dict_valid = []
    for i, l in enumerate(files):
        if 'train' in str(files[i]):
            out_dict_train.append({'file_path': files[i], 'label': y_pred_2_label[i]})
        elif 'test' in str(files[i]):
            out_dict_test.append({'file_path': files[i], 'label': y_pred_2_label[i]})
        elif 'valid' in str(files[i]):
            out_dict_valid.append({'file_path': files[i], 'label': y_pred_2_label[i]})

    write_csv(os.path.join(args.relabel_dir, str(k) + '_train.txt'), out_dict_train)
    write_csv(os.path.join(args.relabel_dir, str(k) + '_test.txt'), out_dict_test)
    write_csv(os.path.join(args.relabel_dir, str(k) + '_valid.txt'), out_dict_valid)


def main():
    args, logging = init("experiments/mlcc/resnet50.yaml")

    with open(args.fc1_path, 'rb') as f:
        data = pickle.load(f)
    with open(args.le_path, 'rb') as f:
        le = pickle.load(f)

    files = data['filename']  # file paths to each image
    fc1 = data['features']  # array containing fc1 features for each file
    labels = data['labels']  # string labels for each image
    print('first 10 files: {}'.format(files[:10]))
    print('first 10 labels: {}'.format(labels[:10]))

    y_gt = le.transform(labels)  # integer labels for each image
    pca = PCA(n_components=args.pca_component, svd_solver='full', whiten=True)
    pca_nw = PCA(n_components=args.pca_component, svd_solver='full', whiten=False)
    x = pca.fit_transform(fc1)
    x_nw = pca_nw.fit_transform(fc1)
    if not args.pca_whitten:
        x = x_nw

    k_values = np.arange(args.k_min, args.k_max)
    acc_k = np.zeros(k_values.shape)
    rs = np.random.RandomState(seed=987654321)
    kmeans_total = {}
    for i, (k, state) in enumerate(zip(k_values, rs.randint(2 ** 32, size=len(k_values)))):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=args.kmeans_n_init, random_state=state)

        kmeans_total[i] = kmeans
        labels_unmatched_ = kmeans.fit_predict(x)
        y_pred_ = ct.label_matcher(labels_unmatched_, y_gt)
        cluster_mapper = {}
        for p in np.unique(y_pred_):
            y_clusters = kmeans.labels_[y_pred_ == p]
            for idx, value in enumerate(np.unique(y_clusters)):
                cluster_mapper['{}-{}'.format(le.inverse_transform([p])[0], idx)] = value
        new_le = ct.CustomLabelEncoder()
        new_le.update_mapper(cluster_mapper)
        y_pred_2_label = new_le.inverse_transform(labels_unmatched_)
        out_dict_train = []
        out_dict_test = []
        out_dict_valid = []
        for i, l in enumerate(files):
            if 'train' in str(files[i]):
                out_dict_train.append({'file_path': files[i], 'label': y_pred_2_label[i]})
            elif 'test' in str(files[i]):
                out_dict_test.append({'file_path': files[i], 'label': y_pred_2_label[i]})
            elif 'valid' in str(files[i]):
                out_dict_valid.append({'file_path': files[i], 'label': y_pred_2_label[i]})

        write_csv(os.path.join(args.relabel_dir, str(k) + '_train.txt'), out_dict_train)
        write_csv(os.path.join(args.relabel_dir, str(k) + '_test.txt'), out_dict_test)
        write_csv(os.path.join(args.relabel_dir, str(k) + '_valid.txt'), out_dict_valid)


if __name__ == '__main__':
    main()
