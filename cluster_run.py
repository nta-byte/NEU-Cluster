from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import pickle
import yaml
import os
import logging
import argparse
from pathlib import Path
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import *

from kneed import KneeLocator
import skimage.io
from scipy.spatial.distance import cdist
import sys

from helper import classification_tools as ct
from helper import visualize as vis
from utils.yaml_config import init


def visual(y_pred, kmeans, le, x_nw, save_name, cluster):
    tsne = TSNE(n_components=2, random_state=12214)
    x_nw_tsne = tsne.fit_transform(x_nw)
    cluster_mapper = {}
    for p in np.unique(y_pred):
        y_clusters = kmeans.labels_[y_pred == p]
        for idx, value in enumerate(np.unique(y_clusters)):
            cluster_mapper[value] = '{}-{}'.format(le.inverse_transform([p])[0], idx)
    palette = np.concatenate((sns.color_palette('pastel', cluster), sns.color_palette('dark', cluster)), axis=0)
    hue = [cluster_mapper[x] for x in kmeans.labels_]
    hue_order = sorted(cluster_mapper.values(), key=lambda x: x.upper())

    fig, ax = plt.subplots(dpi=300, figsize=(5, 5))
    sns.scatterplot(x_nw_tsne[:, 0], x_nw_tsne[:, 1], hue=hue, hue_order=hue_order,
                    palette=dict(zip(hue_order, palette)), ax=ax)
    ax.legend(loc='upper center', bbox_to_anchor=(1, 1))
    plt.savefig(save_name)
    # plt.show()


def get_optimal(k_values, y, direction="decreasing", curve="convex", metric=''):
    x = k_values
    optimal = KneeLocator(x=x, y=y, curve=curve,
                          direction=direction)
    print(f"{metric}: The optimal number of clusters: {optimal.knee} with an inertia of {optimal.knee_y}")


def main():
    args, logging = init("config/config.yaml")

    with open(args.fc1_path, 'rb') as f:
        data = pickle.load(f)

    with open(args.le_path, 'rb') as f:
        le = pickle.load(f)

    files = data['filename']  # file paths to each image
    fc1 = data['features']  # array containing fc1 features for each file
    labels = data['labels']  # string labels for each image
    y_gt = le.transform(labels)  # integer labels for each image
    # print(fc1.shape)
    pca = PCA(n_components=args.pca_component, svd_solver='full', whiten=True)
    pca_nw = PCA(n_components=args.pca_component, svd_solver='full', whiten=False)
    x = pca.fit_transform(fc1)
    x_nw = pca_nw.fit_transform(fc1)
    if not args.pca_whitten:
        x = x_nw
    list_davies_avg = []
    list_completeness = []
    list_silhouette = []
    if not args.use_cache or not os.path.isfile(args.kmeans_k_cache_path):
        k_values = np.arange(5, 15)
        acc_k = np.zeros(k_values.shape)
        rs = np.random.RandomState(seed=987654321)
        kmeans_total = {}
        for i, (k, state) in enumerate(zip(k_values, rs.randint(2 ** 32, size=len(k_values)))):
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=args.kmeans_n_init, random_state=state)
            # kmeans.fit(x)
            kmeans_total[i] = kmeans
            cluster_labels = kmeans.fit_predict(x)
            labels_unmatched_ = kmeans.labels_
            y_pred_ = ct.label_matcher(labels_unmatched_, y_gt)
            acc = (y_pred_ == y_gt).sum() / len(y_gt)
            acc_k[i] = round(acc, 3)

            silhouette_avg = silhouette_score(X=x, labels=cluster_labels, metric='euclidean')
            silhouette_avg = round(float(silhouette_avg), 3)
            list_silhouette.append(silhouette_avg)

            calinski_avg = calinski_harabasz_score(X=x, labels=cluster_labels)
            calinski_avg = round(float(calinski_avg), 3)

            fow_avg = fowlkes_mallows_score(y_gt, cluster_labels)
            fow_avg = round(float(fow_avg), 3)

            davies_avg = davies_bouldin_score(X=x, labels=cluster_labels)
            davies_avg = round(float(davies_avg), 3)
            list_davies_avg.append(davies_avg)

            adjusted_mutual_info = adjusted_mutual_info_score(y_gt, cluster_labels)
            adjusted_mutual_info = round(float(adjusted_mutual_info), 3)

            # completeness = completeness_score(y_gt, cluster_labels)
            # completeness = round(float(completeness), 3)
            # list_completeness.append(completeness)

            # consensus = consensus_score(y_gt, cluster_labels)
            # consensus = round(float(consensus), 3)
            # list_consensus.append(consensus)

            adjusted_rand = adjusted_rand_score(y_gt, cluster_labels)
            adjusted_rand = round(float(adjusted_rand), 3)

            logging.info(
                "cluster: {} accuracy: {} silhouette: {} calinski:{} davies:{} fow:{} AMI:{} AR:{}".format(
                    k, acc_k[i], silhouette_avg, calinski_avg, davies_avg, fow_avg, adjusted_mutual_info,
                    adjusted_rand,
                    # completeness,
                    # consensus,

                ))
            sav_figure_name = "{}_{}_.jpg".format(k, acc_k[i])
            sav_figure_name = os.path.join(args.save_dir, sav_figure_name)
            visual(y_pred_, kmeans, le, x_nw, sav_figure_name, k)

        with open(args.kmeans_k_cache_path, 'wb') as f:
            pickle.dump({
                'kmean': kmeans_total,
                'k_values': k_values,
                'acc_k': acc_k},
                f)
    else:
        with open(args.kmeans_k_cache_path, 'rb') as f:
            results_ = pickle.load(f)
            k_values = results_['k_values']
            acc_k = results_['acc_k']
    get_optimal(k_values, list_davies_avg, metric='davies')
    # get_optimal(k_values, list_silhouette, metric='silhouette', curve='concave', direction="increasing")


if __name__ == '__main__':
    main()
