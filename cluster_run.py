import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import *

from kneed import KneeLocator

from libs.helper import classification_tools as ct
from libs.helper.visualize import visual
from libs.utils.yaml_config import init
from create_pretext_pytorch import extract_feature


def get_optimal(logging, k_values, y, direction="decreasing", curve="convex", metric=''):
    x = k_values
    optimal = KneeLocator(x=x, y=y, curve=curve, direction=direction)
    print(f"{metric}: The optimal number of clusters: {optimal.knee} with an inertia of {optimal.knee_y}")


def print_optimal(logging, dict_data, metric='',args=None):
    dict_data = dict(sorted(dict_data.items(), key=lambda item: item[1]))
    optimal_cluster = [*dict_data][-1]
    print(
        f"{metric}: The optimal number of clusters: {optimal_cluster} with an inertia of {dict_data[optimal_cluster]}")
    sav_figure_name = "{}_.jpg".format(metric)
    if args is not None:
        sav_figure_name = os.path.join(args.save_dir, sav_figure_name)
    sub_visual(dict_data, metric, save_name=sav_figure_name)


def sub_visual(dict_in, title='', save_name=''):
    # dict_in = dict(sorted(dict_in.keys(), key=lambda item: item[1]))
    k_values, acc_k = [], []
    for key in sorted(dict_in):
        k_values.append(key)
        acc_k.append(dict_in[key])
    fig, ax = plt.subplots(dpi=200, figsize=(5, 4))
    ax.plot(k_values, acc_k, '-bo')
    # ax[1].plot(K, [np.median(x) for x in silhouettes], ':o')
    ax.set_xticks(range(int(min(k_values)), int(max(k_values))+1, 1))
    ax.set_xlabel('number of clusters')
    ax.set_ylabel(title)
    if not save_name:
        save_name = title + '.jpg'
    plt.savefig(save_name)


def main():
    args, logging = init("config/config.yaml")

    if os.path.exists(args.fc1_dir) and os.path.exists(args.le_path):
        print()
    else:
        extract_feature(args, logging)
    with open(args.fc1_dir, 'rb') as f:
        data = pickle.load(f)
    with open(args.le_path, 'rb') as f:
        le = pickle.load(f)

    files = data['filename']  # file paths to each image
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
            sav_figure_name = os.path.join(args.save_dir, sav_figure_name)
            visual(y_pred_, kmeans, le, x_nw, sav_figure_name, k)
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


if __name__ == '__main__':
    main()
