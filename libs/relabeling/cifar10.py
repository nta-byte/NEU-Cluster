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
    def __init__(self, cfg, data, list_optimals=[]):
        self.cfg = cfg
        self.list_optimals = list_optimals
        # self.files = data['filename']  # file paths to each image
        self.fc1 = data['features']  # array containing fc1 features for each file
        self.labels = data['new_labels']  # string labels for each image
        # self.labels = data['labels']
        self.le = data['new_le']
        print('new_le', self.le.mapper)
        print('set data', set(self.labels))
        self.y_gt = self.le.transform(self.labels)  # integer labels for each image

    def load_state(self):
        with open(self.cfg['clustering_params']['kmean']['cache_path'], 'rb') as f:
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
            # if k in self.list_optimals:
            print(f"cluster: {k} "
                  f"accuracy: {self.acc_k[i]} "
                  f"fow:{self.dict_fow_avg[k]} "
                  f"AMI:{self.dict_adjusted_mutual_info[k]} "
                  f"NMI:{self.dict_normalized_mutual_info[k]} "
                  f"AR:{self.dict_adjusted_rand[k]}")
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
            # print(new_le.mapper)
            y_pred_2_label = new_le.inverse_transform(labels_unmatched_)
            if self.cfg['relabel_params']['dataset_part'] == 'train':
                out_train = y_pred_2_label
                # out_test = y_pred_2_label[50000:]
                with open(os.path.join(self.cfg['relabel_params']['relabel_dir'], str(k) + '_train.pkl'), 'wb') as f:
                    pickle.dump(out_train, f)
                with open(os.path.join(self.cfg['relabel_params']['relabel_dir'], str(k) + '_new_le.pkl'), 'wb') as f:
                    pickle.dump(new_le, f)

            if self.cfg['relabel_params']['dataset_part'] == 'test':
                # out_train = y_pred_2_label
                out_test = y_pred_2_label
                with open(os.path.join(self.cfg['relabel_params']['relabel_dir'], str(k) + '_test.pkl'), 'wb') as f:
                    pickle.dump(out_test, f)
                with open(os.path.join(self.cfg['relabel_params']['relabel_dir'], str(k) + '_new_le.pkl'), 'wb') as f:
                    pickle.dump(new_le, f)

            elif self.cfg['relabel_params']['dataset_part'] == 'train_test':
                out_train = y_pred_2_label[:50000]
                out_test = y_pred_2_label[50000:]
                with open(os.path.join(self.cfg['relabel_params']['relabel_dir'], str(k) + '_train.pkl'), 'wb') as f:
                    pickle.dump(out_train, f)
                with open(os.path.join(self.cfg['relabel_params']['relabel_dir'], str(k) + '_test.pkl'), 'wb') as f:
                    pickle.dump(out_test, f)
                with open(os.path.join(self.cfg['relabel_params']['relabel_dir'], str(k) + '_new_le.pkl'), 'wb') as f:
                    pickle.dump(new_le, f)

    def save_output(self):
        pass


# from cluster_run import decrease_dim


class Relabel_flow4:
    def __init__(self, args, data, list_optimals=[], active_data='train'):
        self.args = args
        # self.
        self.list_optimals = list_optimals
        if active_data == 'train':
            self.labels = data['new_trainlabels']  # string labels for each image
        else:
            self.labels = data['org_testlabels']
        self.fc1 = data['features']  # array containing fc1 features for each file
        # self.labels = data['new_labels']
        self.le = data['original_le']
        # print('new_le', self.le.mapper)
        # print('set data', set(self.labels))
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
            if k in self.list_optimals:
                print(f"cluster: {k} "
                      f"accuracy: {self.acc_k[i]} "
                      f"fow:{self.dict_fow_avg[k]} "
                      f"AMI:{self.dict_adjusted_mutual_info[k]} "
                      f"NMI:{self.dict_normalized_mutual_info[k]} "
                      f"AR:{self.dict_adjusted_rand[k]}")
                # labels_unmatched_ = self.dict_cluster_labels[k]
                kmeans = self.kmeans_total[i]
                labels_unmatched_ = kmeans.predict(self.fc1)
                print(k)
                print(set(kmeans.labels_), kmeans.labels_)
                print(set(labels_unmatched_), labels_unmatched_)
                y_pred_ = ct.label_matcher(labels_unmatched_, self.y_gt)

                print(set(y_pred_), y_pred_)

                cluster_mapper = {}
                for p in np.unique(kmeans.labels_):
                    y_clusters = labels_unmatched_[y_pred_ == p]
                    print(set(y_clusters))
                    for idx, value in enumerate(np.unique(y_clusters)):
                        cluster_mapper['{}-{}'.format(self.le.inverse_transform([p])[0], idx)] = value
                print(cluster_mapper)
                new_le = ct.CustomLabelEncoder()
                new_le.update_mapper(cluster_mapper)
                y_pred_2_label = new_le.inverse_transform(labels_unmatched_)
                out_train = y_pred_2_label
                with open(os.path.join(self.args.relabel_dir, str(k) + '_train.pkl'), 'wb') as f:
                    pickle.dump(out_train, f)
                with open(os.path.join(self.args.relabel_dir, str(k) + '_new_le.pkl'), 'wb') as f:
                    pickle.dump(new_le, f)

    def save_output(self):
        pass


if __name__ == '__main__':
    pass
