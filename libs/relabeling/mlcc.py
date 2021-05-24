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


class Relabel:
    def __init__(self, args, data):
        self.args = args
        self.files = data['filename']  # file paths to each image
        self.fc1 = data['features']  # array containing fc1 features for each file
        # self.labels = data['labels']  # string labels for each image
        self.labels = data['new_labels']  # string labels for each image
        # self.labels = data['labels']
        self.le = data['new_le']
        # self.labels = data['labels']
        # self.le = data['le']
        self.y_gt = self.le.transform(self.labels)  # integer labels for each image

    def load_state(self):
        with open(self.args.kmeans_k_cache_path, 'rb') as f:
            results_ = pickle.load(f)
            self.kmeans_total = results_['kmean']
            self.k_values = results_['k_values']
            self.acc_k = results_['acc_k']
            self.dict_cluster_labels = results_['cluster_labels']
            self.dict_adjusted_rand = results_['dict_adjusted_rand']
            self.dict_adjusted_mutual_info = results_['dict_adjusted_mutual_info']
            self.dict_normalized_mutual_info = results_['dict_normalized_mutual_info']
            self.dict_fow_avg = results_['dict_fow_avg']

    def process_relabel(self):
        for i, k in enumerate(self.k_values):
            labels_unmatched_ = self.dict_cluster_labels[k]
            kmeans = self.kmeans_total[i]
            y_pred_ = ct.label_matcher(labels_unmatched_, self.y_gt)
            cluster_mapper = {}
            for p in np.unique(y_pred_):
                y_clusters = kmeans.labels_[y_pred_ == p]
                for idx, value in enumerate(np.unique(y_clusters)):
                    cluster_mapper['{}-{}'.format(self.le.inverse_transform([p])[0], idx)] = value
            new_le = ct.CustomLabelEncoder()
            new_le.update_mapper(cluster_mapper)
            y_pred_2_label = new_le.inverse_transform(labels_unmatched_)
            out_dict_train = []
            for i, l in enumerate(self.files):
                out_dict_train.append({'file_path': self.files[i], 'label': y_pred_2_label[i]})
            if self.args.cluster_dataset == 'train':
                out_train = y_pred_2_label
                with open(os.path.join(self.args.relabel_dir, str(k) + '_train.pkl'), 'wb') as f:
                    pickle.dump(out_train, f)
                with open(os.path.join(self.args.relabel_dir, str(k) + '_new_le.pkl'), 'wb') as f:
                    pickle.dump(new_le, f)

                # write_csv(os.path.join(self.args.relabel_dir, str(k) + '_train.txt'), out_dict_train)

            if self.args.cluster_dataset == 'test':
                # out_train = y_pred_2_label
                out_test = y_pred_2_label
                with open(os.path.join(self.args.relabel_dir, str(k) + '_test.pkl'), 'wb') as f:
                    pickle.dump(out_test, f)
                with open(os.path.join(self.args.relabel_dir, str(k) + '_new_le.pkl'), 'wb') as f:
                    pickle.dump(new_le, f)

                # write_csv(os.path.join(self.args.relabel_dir, str(k) + '_test.txt'), out_dict_train)

            elif self.args.cluster_dataset == 'train_test':
                train_num = 950
                out_train = y_pred_2_label[:train_num]
                out_test = y_pred_2_label[train_num:]
                with open(os.path.join(self.args.relabel_dir, str(k) + '_train.pkl'), 'wb') as f:
                    pickle.dump(out_train, f)
                with open(os.path.join(self.args.relabel_dir, str(k) + '_test.pkl'), 'wb') as f:
                    pickle.dump(out_test, f)
                with open(os.path.join(self.args.relabel_dir, str(k) + '_new_le.pkl'), 'wb') as f:
                    pickle.dump(new_le, f)

                # write_csv(os.path.join(self.args.relabel_dir, str(k) + '_train.txt'), out_dict_train[:train_num])
                # write_csv(os.path.join(self.args.relabel_dir, str(k) + '_test.txt'), out_dict_train[train_num:])

    def save_output(self):
        pass


if __name__ == '__main__':
    pass
