import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import yaml
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import *
from hdbscan import HDBSCAN
import umap
from kneed import KneeLocator

from libs.helper import classification_tools as ct
from libs.helper.visualize import visual
from libs.utils.yaml_config import init
from create_pretext_pytorch import extract_feature
# from libs.clustering.VAE.data_loader import VAEDataset
# from libs.clustering.VAE.models.models import VAE
# from libs.pretext.utils import load_state
from libs.clustering.VAE.train import fit
from libs.clustering.VAE.cluster import vae_reduce_dimension


def get_optimal(logging, k_values, y, direction="decreasing", curve="convex", metric=''):
    x = k_values
    optimal = KneeLocator(x=x, y=y, curve=curve, direction=direction)
    logging.info(f"{metric}: The optimal number of clusters: {optimal.knee} with an inertia of {optimal.knee_y}")


def print_optimal(logging, dict_data, metric='', args=None):
    dict_data = dict(sorted(dict_data.items(), key=lambda item: item[1]))
    optimal_cluster = [*dict_data][-1]
    logging.info(
        f"{metric}: The optimal number of clusters: {optimal_cluster} with an inertia of {dict_data[optimal_cluster]}")
    sav_figure_name = "{}_.jpg".format(metric)
    if args is not None:
        sav_figure_name = os.path.join(args.save_dir, sav_figure_name)
    sub_visual(dict_data, metric, save_name=sav_figure_name)
    return optimal_cluster


def sub_visual(dict_in, title='', save_name=''):
    # dict_in = dict(sorted(dict_in.keys(), key=lambda item: item[1]))
    k_values, acc_k = [], []
    for key in sorted(dict_in):
        k_values.append(key)
        acc_k.append(dict_in[key])
    fig, ax = plt.subplots(dpi=200, figsize=(5, 4))
    ax.plot(k_values, acc_k, '-bo')
    # ax[1].plot(K, [np.median(x) for x in silhouettes], ':o')
    ax.set_xticks(range(int(min(k_values)), int(max(k_values)) + 1, 1))
    ax.set_xlabel('number of clusters')
    ax.set_ylabel(title)
    if not save_name:
        save_name = title + '.jpg'
    plt.savefig(save_name)


def decrease_dim(args, fc1, data=None):
    print(f'decrease_dim by {args.reduce_dimension}')
    if args.reduce_dimension == 'umap':

        x = umap.UMAP(
            n_neighbors=200,
            min_dist=0.0,
            n_components=2,
            # random_state=42,
            metric='correlation',
            init='random',
        ).fit_transform(fc1,
                        # y=y_gt
                        )
    elif args.reduce_dimension == 'pca':
        pca = PCA(n_components=args.pca_component, svd_solver='full', whiten=True)
        pca_nw = PCA(n_components=args.pca_component, svd_solver='full', whiten=False)
        x = pca.fit_transform(fc1)
        x_nw = pca_nw.fit_transform(fc1)
        if not args.pca_whitten:
            x = x_nw
    elif args.reduce_dimension == 'vae':
        with open(args.vae_cfg, 'r') as file:
            try:
                vaeconfig = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)

        # print(vaeconfig)
        vaeconfig['exp_params']['train_data_path'] = args.fc1_path_vae
        vaeconfig['logging_params']['save_dir'] = os.path.join(args.save_dir, vaeconfig['logging_params']['save_dir'])
        # print(vaeconfig)
        # print('0', vaeconfig['model_params']['hidden_dims'])
        vaeconfig['infer']['weight_path'] = fit(vaeconfig)
        # print('1', vaeconfig['infer']['weight_path'])
        x = vae_reduce_dimension(vaeconfig, data)
    elif args.reduce_dimension == 'none':
        x = fc1
    return x


def clustering(args, logging, data, org_eval=True):
    fc1 = data['features']  # array containing fc1 features for each file
    labels = data['org_labels'] if org_eval else data['new_labels']  # string labels for each image
    le = data['original_le'] if org_eval else data['new_le']
    dict_fow_avg = {}
    dict_adjusted_mutual_info = {}
    dict_normalized_mutual_info = {}
    dict_adjusted_rand = {}
    dict_silhouette = {}
    dict_cluster_labels = {}
    if not args.use_cache or not os.path.isfile(args.kmeans_k_cache_path):
        y_gt = le.transform(labels)  # integer labels for each image
        x = decrease_dim(args, fc1, data)
        print("done decrease dimension", x.shape)
        k_values = np.arange(3, 17)
        acc_k = np.zeros(k_values.shape)
        rs = np.random.RandomState(seed=987654321)
        kmeans_total = {}
        for i, (k, state) in enumerate(zip(k_values, rs.randint(2 ** 32, size=len(k_values)))):
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=args.kmeans_n_init, random_state=state)

            cluster_labels = kmeans.fit_predict(x)
            dict_cluster_labels[k] = cluster_labels
            kmeans_total[i] = kmeans
            labels_unmatched_ = kmeans.labels_
            y_pred_ = ct.label_matcher(labels_unmatched_, y_gt)
            acc = (y_pred_ == y_gt).sum() / len(y_gt)
            acc_k[i] = round(acc, 3)
            fow_avg = fowlkes_mallows_score(y_gt, cluster_labels)
            fow_avg = round(float(fow_avg), 3)
            dict_fow_avg[k] = fow_avg

            adjusted_mutual_info = adjusted_mutual_info_score(cluster_labels, y_gt)
            adjusted_mutual_info = round(float(adjusted_mutual_info), 3)
            dict_adjusted_mutual_info[k] = adjusted_mutual_info

            normalized_mutual_info = normalized_mutual_info_score(cluster_labels, y_gt)
            normalized_mutual_info = round(float(normalized_mutual_info), 3)
            dict_normalized_mutual_info[k] = normalized_mutual_info

            adjusted_rand = adjusted_rand_score(cluster_labels, y_gt)
            adjusted_rand = round(float(adjusted_rand), 3)
            dict_adjusted_rand[k] = adjusted_rand

            silhouette = silhouette_score(x, cluster_labels)
            silhouette = round(float(silhouette), 3)
            dict_silhouette[k] = silhouette

            sav_figure_name = "{}_{}_.jpg".format(k, acc_k[i])
            sav_figure_name = os.path.join(args.save_img_1st_step_dir, sav_figure_name)
            visual(y_pred_, kmeans, le, x, sav_figure_name, k)

            print(
                "cluster: {} accuracy: {} fow:{} AMI:{} NMI:{} AR:{}, Silhouette:{}".format(
                    k,
                    acc_k[i],
                    dict_fow_avg[k],
                    dict_adjusted_mutual_info[k],
                    dict_normalized_mutual_info[k],
                    dict_adjusted_rand[k],
                    dict_silhouette[k],
                ))
        optimal_cluster_list = []
        optimal_cluster_list.append(print_optimal(logging, dict_fow_avg, metric='fowlkes_mallows_score', args=args))
        optimal_cluster_list.append(
            print_optimal(logging, dict_adjusted_mutual_info, metric='adjusted_mutual_info_score', args=args))
        optimal_cluster_list.append(
            print_optimal(logging, dict_normalized_mutual_info, metric='normalized_mutual_info_score', args=args))
        optimal_cluster_list.append(print_optimal(logging, dict_adjusted_rand, metric='adjusted_rand_score', args=args))
        # optimal_cluster_list.append(print_optimal(logging, dict_silhouette, metric='silhouette_score', args=args))

        with open(args.kmeans_k_cache_path, 'wb') as f:
            pickle.dump({
                'kmean': kmeans_total,
                'k_values': k_values,
                'acc_k': acc_k,
                'cluster_labels': dict_cluster_labels,
                'dict_adjusted_rand': dict_adjusted_rand,
                'dict_adjusted_mutual_info': dict_adjusted_mutual_info,
                'dict_normalized_mutual_info': dict_normalized_mutual_info,
                'dict_fow_avg': dict_fow_avg,
                'dict_silhouette': dict_silhouette,
                'optimal_cluster_list': optimal_cluster_list
            }, f)
    else:
        with open(args.kmeans_k_cache_path, 'rb') as f:
            results_ = pickle.load(f)
            k_values = results_['k_values']
            acc_k = results_['acc_k']
            dict_adjusted_rand = results_['dict_adjusted_rand']
            dict_adjusted_mutual_info = results_['dict_adjusted_mutual_info']
            dict_normalized_mutual_info = results_['dict_normalized_mutual_info']
            dict_fow_avg = results_['dict_fow_avg']
            dict_silhouette = results_['dict_silhouette']
            optimal_cluster_list = results_['optimal_cluster_list']

    for i, k in enumerate(k_values):
        logging.info(
            "cluster: {} accuracy: {} fow:{} AMI:{} NMI:{} AR:{} Silhouette:{}".format(
                k, acc_k[i],
                dict_fow_avg[k],
                dict_adjusted_mutual_info[k],
                dict_normalized_mutual_info[k],
                dict_adjusted_rand[k],
                dict_silhouette[k]
            ))

    logging.info(f"optimal number of clusters {optimal_cluster_list}")
    return optimal_cluster_list


def main():
    args, logging = init("experiments/neu-cls/flow1_resnet50.yaml")

    if os.path.exists(args.fc1_dir) and os.path.exists(args.le_path):
        print()
    else:
        extract_feature(args, logging)
    with open(args.fc1_path, 'rb') as f:
        data = pickle.load(f)
    with open(args.le_path, 'rb') as f:
        le = pickle.load(f)

    # files = data['filename']  # file paths to each image
    fc1 = data['features']  # array containing fc1 features for each file
    labels = data['labels']  # string labels for each image

    list_davies_avg = []
    list_silhouette = []
    dict_calinski_avg = {}
    dict_fow_avg = {}
    dict_adjusted_mutual_info = {}
    dict_normalized_mutual_info = {}
    dict_adjusted_rand = {}
    if not args.use_cache or not os.path.isfile(args.kmeans_k_cache_path):

        y_gt = le.transform(labels)  # integer labels for each image
        pca = PCA(n_components=args.pca_component, svd_solver='full', whiten=True)
        pca_nw = PCA(n_components=args.pca_component, svd_solver='full', whiten=False)
        x = pca.fit_transform(fc1)
        x_nw = pca_nw.fit_transform(fc1)
        if not args.pca_whitten:
            x = x_nw

        k_values = np.arange(3, 17)
        acc_k = np.zeros(k_values.shape)
        rs = np.random.RandomState(seed=987654321)
        kmeans_total = {}
        for i, (k, state) in enumerate(zip(k_values, rs.randint(2 ** 32, size=len(k_values)))):
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=args.kmeans_n_init, random_state=state)

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
            dict_calinski_avg[k] = calinski_avg

            fow_avg = fowlkes_mallows_score(y_gt, cluster_labels)
            fow_avg = round(float(fow_avg), 3)
            dict_fow_avg[k] = fow_avg

            davies_avg = davies_bouldin_score(X=x, labels=cluster_labels)
            davies_avg = round(float(davies_avg), 3)
            list_davies_avg.append(davies_avg)

            adjusted_mutual_info = adjusted_mutual_info_score(y_gt, cluster_labels)
            adjusted_mutual_info = round(float(adjusted_mutual_info), 3)
            dict_adjusted_mutual_info[k] = adjusted_mutual_info

            normalized_mutual_info = normalized_mutual_info_score(y_gt, cluster_labels)
            normalized_mutual_info = round(float(normalized_mutual_info), 3)
            dict_normalized_mutual_info[k] = normalized_mutual_info

            adjusted_rand = adjusted_rand_score(y_gt, cluster_labels)
            adjusted_rand = round(float(adjusted_rand), 3)
            dict_adjusted_rand[k] = adjusted_rand

            sav_figure_name = "{}_{}_.jpg".format(k, acc_k[i])
            sav_figure_name = os.path.join(args.save_img_1st_step_dir, sav_figure_name)
            visual(y_pred_, kmeans, le, x, sav_figure_name, k)
            print(
                "cluster: {} accuracy: {} silhouette: {} calinski:{} davies:{} fow:{} AMI:{} NMI:{} AR:{}".format(
                    k, acc_k[i], list_silhouette[i], dict_calinski_avg[k], list_davies_avg[i], dict_fow_avg[k],
                    dict_adjusted_mutual_info[k],
                    dict_normalized_mutual_info[k],
                    dict_adjusted_rand[k],
                ))

        with open(args.kmeans_k_cache_path, 'wb') as f:
            pickle.dump({
                'kmean': kmeans_total,
                'k_values': k_values,
                'acc_k': acc_k,
                'list_silhouette': list_silhouette,
                'dict_adjusted_rand': dict_adjusted_rand,
                'dict_adjusted_mutual_info': dict_adjusted_mutual_info,
                'dict_normalized_mutual_info': dict_normalized_mutual_info,
                'list_davies_avg': list_davies_avg,
                'dict_fow_avg': dict_fow_avg,
                'dict_calinski_avg': dict_calinski_avg
            }, f)
    else:
        with open(args.kmeans_k_cache_path, 'rb') as f:
            results_ = pickle.load(f)
            k_values = results_['k_values']
            acc_k = results_['acc_k']
            list_silhouette = results_['list_silhouette']
            dict_calinski_avg = results_['dict_calinski_avg']
            dict_adjusted_rand = results_['dict_adjusted_rand']
            dict_adjusted_mutual_info = results_['dict_adjusted_mutual_info']
            dict_normalized_mutual_info = results_['dict_normalized_mutual_info']
            list_davies_avg = results_['list_davies_avg']
            dict_fow_avg = results_['dict_fow_avg']

    for i, k in enumerate(k_values):
        logging.info(
            "cluster: {} accuracy: {} silhouette: {} calinski:{} davies:{} fow:{} AMI:{} NMI:{} AR:{}".format(
                k, acc_k[i], list_silhouette[i], dict_calinski_avg[k], list_davies_avg[i], dict_fow_avg[k],
                dict_adjusted_mutual_info[k],
                dict_normalized_mutual_info[k],
                dict_adjusted_rand[k],
            ))

    get_optimal(logging, k_values, list_davies_avg, metric='davies_bouldin_score')
    print_optimal(logging, dict_calinski_avg, metric='calinski_harabasz_score', args=args)
    print_optimal(logging, dict_fow_avg, metric='fowlkes_mallows_score', args=args)
    print_optimal(logging, dict_adjusted_mutual_info, metric='adjusted_mutual_info_score', args=args)
    print_optimal(logging, dict_normalized_mutual_info, metric='normalized_mutual_info_score', args=args)
    print_optimal(logging, dict_adjusted_rand, metric='adjusted_rand_score', args=args)


def umap_clustering():
    args, logging = init("experiments/cifar10/flow1_resnet18.yaml")
    args.cluster_dataset = 'test'

    # extract_feature(args, logging)
    with open(args.fc1_path, 'rb') as f:
        data = pickle.load(f)

    fc1 = data['features']  # array containing fc1 features for each file
    cluster_labels = data['labels']  # string labels for each image
    le = data['le']

    if not args.use_cache or not os.path.isfile(args.kmeans_k_cache_path):
        y_gt = le.transform(cluster_labels)  # integer labels for each image
        embedding = umap.UMAP(
            n_neighbors=200,
            min_dist=0.0,
            n_components=2,
            # random_state=42,
            metric='correlation',
            init='random',
        ).fit_transform(fc1,
                        # y=y_gt
                        )

        print(embedding.shape)
        plt.scatter(embedding[:, 0], embedding[:, 1], c=y_gt,
                    s=0.1, cmap='Spectral')
        plt.savefig('a.png')

        cluster_labels = HDBSCAN(
            min_samples=100,
            min_cluster_size=50,
        ).fit_predict(embedding)
        # cluster_labels = clusterer.fit_predict(x)
        # cluster_labels = cluster_labels

        # print(set(cluster_labels))
        y_pred_ = ct.label_matcher(cluster_labels, y_gt)

        acc = (y_pred_ == y_gt).sum() / len(y_gt)
        print(round(acc, 3), set(cluster_labels))

        unique, counts = np.unique(cluster_labels, return_counts=True)
        fr = np.asarray((unique, counts)).T
        print(fr)

        plt.scatter(embedding[:, 0], embedding[:, 1], c=y_pred_,
                    s=0.1, cmap='Spectral')
        plt.savefig('a_predict.png')

        k_values = np.arange(3, 17)
        acc_k = np.zeros(k_values.shape)
        rs = np.random.RandomState(seed=987654321)
        kmeans_total = {}
        dict_fow_avg = {}
        dict_adjusted_mutual_info = {}
        dict_normalized_mutual_info = {}
        dict_adjusted_rand = {}
        dict_cluster_labels = {}
        for i, (k, state) in enumerate(zip(k_values, rs.randint(2 ** 32, size=len(k_values)))):
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=args.kmeans_n_init, random_state=state)

            cluster_labels = kmeans.fit_predict(embedding)
            # dict_cluster_labels[k] = cluster_labels
            kmeans_total[i] = kmeans
            labels_unmatched_ = kmeans.labels_
            y_pred_ = ct.label_matcher(labels_unmatched_, y_gt)
            acc = (y_pred_ == y_gt).sum() / len(y_gt)
            acc_k[i] = round(acc, 3)
            fow_avg = fowlkes_mallows_score(y_gt, cluster_labels)
            fow_avg = round(float(fow_avg), 3)
            dict_fow_avg[k] = fow_avg

            adjusted_mutual_info = adjusted_mutual_info_score(y_gt, cluster_labels)
            adjusted_mutual_info = round(float(adjusted_mutual_info), 3)
            dict_adjusted_mutual_info[k] = adjusted_mutual_info

            normalized_mutual_info = normalized_mutual_info_score(y_gt, cluster_labels)
            normalized_mutual_info = round(float(normalized_mutual_info), 3)
            dict_normalized_mutual_info[k] = normalized_mutual_info

            adjusted_rand = adjusted_rand_score(y_gt, cluster_labels)
            adjusted_rand = round(float(adjusted_rand), 3)
            dict_adjusted_rand[k] = adjusted_rand

            sav_figure_name = "{}_{}_.jpg".format(k, acc_k[i])
            sav_figure_name = os.path.join(args.save_img_1st_step_dir, sav_figure_name)
            visual(y_pred_, kmeans, le, embedding, sav_figure_name, k)
            print(
                "cluster: {} accuracy: {} fow:{} AMI:{} NMI:{} AR:{}".format(
                    k,
                    acc_k[i],
                    dict_fow_avg[k],
                    dict_adjusted_mutual_info[k],
                    dict_normalized_mutual_info[k],
                    dict_adjusted_rand[k],
                ))


if __name__ == '__main__':
    umap_clustering()
