import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from sklearn.cluster import KMeans
# from h2o4gpu.solvers import KMeans
# from libKMCUDA import kmeans_cuda, knn_cuda
# from fast_pytorch_kmeans import KMeans
# from kmeans_pytorch import kmeans, kmeans_predict

from sklearn.metrics import *
from kneed import KneeLocator
from libs.helper import classification_tools as ct
from libs.helper.visualize import visual, visual_
# from libs.clustering.VAE.data_loader import VAEDataset
from libs.clustering.dimension import decrease_dim, decrease_dim_for_visual
from libs.clustering.kmeans import run_kmeans3


# from libs.clustering.VAE.models.models import VAE
# from libs.pretext.utils import load_state


def get_optimal(logging, k_values, y, direction="decreasing", curve="convex", metric=''):
    x = k_values
    optimal = KneeLocator(x=x, y=y, curve=curve, direction=direction)
    logging.info(f"{metric}: The optimal number of clusters: {optimal.knee} with an inertia of {optimal.knee_y}")


def print_optimal(logging, dict_data, metric='', cfg=None):
    dict_data = dict(sorted(dict_data.items(), key=lambda item: item[1]))
    optimal_cluster = [*dict_data][-1]
    logging.info(
        f"{metric}: The optimal number of clusters: {optimal_cluster} with an inertia of {dict_data[optimal_cluster]}")
    sav_figure_name = "{}_.jpg".format(metric)
    if cfg is not None:
        sav_figure_name = os.path.join(cfg['general']['save_dir'], sav_figure_name)
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


def clustering(cfg, logging, data, org_eval=True, kmeans_step=1):
    fc1 = data['features']  # array containing fc1 features for each file
    labels = data['org_labels'] if org_eval else data['new_labels']  # string labels for each image
    le = data['original_le'] if org_eval else data['new_le']
    dict_fow_avg = {}
    dict_adjusted_mutual_info = {}
    dict_normalized_mutual_info = {}
    dict_adjusted_rand = {}
    dict_silhouette = {}
    dict_cluster_labels = {}
    c_params = cfg['clustering_params']
    if not c_params['kmean']['use_cache'] or not os.path.isfile(
            c_params['kmean']['cache_path']):
        y_gt = le.transform(labels)  # integer labels for each image
        x = decrease_dim(cfg, fc1, data)
        if x.shape[1] != 2:
            x_visual = decrease_dim_for_visual(fc1)
        else:
            x_visual = x
        # x_visual = decrease_dim_for_visual(fc1)
        print("done decrease dimension", x.shape)
        k_values = np.arange(c_params['kmean']['k_min'], c_params['kmean']['k_max'])
        acc_k = np.zeros(k_values.shape)
        rs = np.random.RandomState(seed=c_params['seed'])
        kmeans_total = {}
        for i, (k, state) in enumerate(zip(k_values, rs.randint(2 ** 32, size=len(k_values)))):

            if kmeans_step == 1:
                kmeans = KMeans(n_clusters=k, init='k-means++', n_init=c_params['kmean']['n_init'],
                                random_state=c_params['seed'])

                cluster_labels = kmeans.fit_predict(x)
            elif kmeans_step == 2:
                k_mean_first_step = KMeans(n_clusters=k * 2, init='k-means++', n_init=c_params['kmean']['n_init'],
                                           verbose=0, random_state=c_params['seed'])
                cluster_labels_ = k_mean_first_step.fit_predict(X=x)
                cluster_centers = k_mean_first_step.cluster_centers_

                center_groups = list()
                for i1 in range(len(cluster_centers)):
                    cluster_i = [j for j in cluster_labels_ if j == i1]
                    center_groups.append((i1, len(cluster_i)))
                center_groups = sorted(center_groups, key=lambda tup: tup[1])

                keep_cluster = [i2[0] for i2 in center_groups[k:]]
                cluster_centers = cluster_centers[keep_cluster]

                kmeans = KMeans(n_clusters=k, init=cluster_centers, verbose=0, random_state=c_params['seed'],
                                n_init=c_params['kmean']['n_init'])
                cluster_labels = kmeans.fit_predict(X=x)
            dict_cluster_labels[k] = cluster_labels
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

            silhouette = silhouette_score(x, cluster_labels)
            silhouette = round(float(silhouette), 3)
            dict_silhouette[k] = silhouette
            # dict_silhouette[k] = 1

            sav_figure_name = "{}_{}_.jpg".format(k, acc_k[i])
            sav_figure_name = os.path.join(cfg['general']['save_cluster_visualization'], sav_figure_name)
            visual(y_pred_, kmeans, le, x_visual, sav_figure_name, k)

            logging.info(
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
        optimal_cluster_list.append(print_optimal(logging, dict_fow_avg, metric='fowlkes_mallows_score', cfg=cfg))
        optimal_cluster_list.append(
            print_optimal(logging, dict_adjusted_mutual_info, metric='adjusted_mutual_info_score', cfg=cfg))
        optimal_cluster_list.append(
            print_optimal(logging, dict_normalized_mutual_info, metric='normalized_mutual_info_score', cfg=cfg))
        optimal_cluster_list.append(print_optimal(logging, dict_adjusted_rand, metric='adjusted_rand_score', cfg=cfg))
        # optimal_cluster_list.append(print_optimal(logging, dict_silhouette, metric='silhouette_score', args=args))

        with open(cfg['clustering_params']['kmean']['cache_path'], 'wb') as f:
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
        with open(cfg['clustering_params']['kmean']['cache_path'], 'rb') as f:
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


def gpu_clustering(cfg, logging, data, org_eval=True, kmeans_step=1):
    fc1 = data['features']  # array containing fc1 features for each file
    labels = data['org_labels'] if org_eval else data['new_labels']  # string labels for each image
    le = data['original_le'] if org_eval else data['new_le']
    dict_fow_avg = {}
    dict_adjusted_mutual_info = {}
    dict_normalized_mutual_info = {}
    dict_adjusted_rand = {}
    dict_silhouette = {}
    dict_cluster_labels = {}
    c_params = cfg['clustering_params']
    if not c_params['kmean']['use_cache'] or not os.path.isfile(
            c_params['kmean']['cache_path']):
        y_gt = le.transform(labels)  # integer labels for each image
        x = decrease_dim(cfg, fc1, data)
        # print('type x:', type(x))
        x = np.ascontiguousarray(x)
        # print('type x:', type(x))
        # x = torch.tensor(x)
        if x.shape[1] != 2:
            x_visual = decrease_dim_for_visual(fc1)
        else:
            x_visual = x
        print("done decrease dimension", x.shape)
        k_values = np.arange(c_params['kmean']['k_min'], c_params['kmean']['k_max'])
        acc_k = np.zeros(k_values.shape)
        rs = np.random.RandomState(seed=c_params['seed'])
        kmeans_total = {}

        for i, (k, state) in enumerate(zip(k_values, rs.randint(2 ** 32, size=len(k_values)))):
            cluster_labels = run_kmeans3(
                X=x, nmb_clusters=int(k),
                n_init=10,
                seed=c_params['seed']
            )
            dict_cluster_labels[k] = cluster_labels
            kmeans_total[i] = cluster_labels
            y_pred_ = ct.label_matcher(cluster_labels, y_gt)
            acc = (y_pred_ == y_gt).sum() / len(y_gt)
            acc_k[i] = round(acc, 3)
            fow_avg = fowlkes_mallows_score(y_gt, cluster_labels)
            fow_avg = round(float(fow_avg), 3)
            dict_fow_avg[k] = fow_avg
            # print('done fow cal')
            adjusted_mutual_info = adjusted_mutual_info_score(y_gt, cluster_labels)
            adjusted_mutual_info = round(float(adjusted_mutual_info), 3)
            dict_adjusted_mutual_info[k] = adjusted_mutual_info
            # print('done ami cal')
            normalized_mutual_info = normalized_mutual_info_score(y_gt, cluster_labels)
            normalized_mutual_info = round(float(normalized_mutual_info), 3)
            dict_normalized_mutual_info[k] = normalized_mutual_info
            # print('done nmi cal')
            adjusted_rand = adjusted_rand_score(y_gt, cluster_labels)
            adjusted_rand = round(float(adjusted_rand), 3)
            dict_adjusted_rand[k] = adjusted_rand
            # print('done ar cal')
            silhouette = silhouette_score(x, cluster_labels)
            silhouette = round(float(silhouette), 3)
            dict_silhouette[k] = silhouette
            # dict_silhouette[k] = 1
            # print('done sil cal')
            sav_figure_name = "{}_{}_.jpg".format(k, acc_k[i])
            sav_figure_name = os.path.join(cfg['general']['save_cluster_visualization'], sav_figure_name)
            visual_(y_pred_, cluster_labels, le, x_visual, sav_figure_name, k)

            logging.info(
                "cluster: {} accuracy: {:.3f} fow:{:.3f} AMI:{:.3f} NMI:{:.3f} AR:{:.3f} Silhouette:{:.3f}".format(
                    k,
                    acc_k[i],
                    dict_fow_avg[k],
                    dict_adjusted_mutual_info[k],
                    dict_normalized_mutual_info[k],
                    dict_adjusted_rand[k],
                    dict_silhouette[k],
                ))
        optimal_cluster_list = []
        optimal_cluster_list.append(print_optimal(logging, dict_fow_avg, metric='FM ', cfg=cfg))
        optimal_cluster_list.append(
            print_optimal(logging, dict_adjusted_mutual_info, metric='AMI', cfg=cfg))
        optimal_cluster_list.append(
            print_optimal(logging, dict_normalized_mutual_info, metric='NMI', cfg=cfg))
        optimal_cluster_list.append(print_optimal(logging, dict_adjusted_rand, metric='AR ', cfg=cfg))
        optimal_cluster_list.append(print_optimal(logging, dict_silhouette, metric='SC ', cfg=cfg))

        with open(cfg['clustering_params']['kmean']['cache_path'], 'wb') as f:
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
        with open(cfg['clustering_params']['kmean']['cache_path'], 'rb') as f:
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
                "cluster: {} accuracy: {:.3f} fow:{:.3f} AMI:{:.3f} NMI:{:.3f} AR:{:.3f} Silhouette:{:.3f}".format(
                    k, float(acc_k[i]),
                    dict_fow_avg[k],
                    dict_adjusted_mutual_info[k],
                    dict_normalized_mutual_info[k],
                    dict_adjusted_rand[k],
                    dict_silhouette[k]
                ))

    logging.info(f"optimal number of clusters {optimal_cluster_list}")
    return optimal_cluster_list


def clustering_flow4(args, logging, data, org_eval=True, active_data='train'):
    fc1 = data['features']  # array containing fc1 features for each file
    if active_data == 'train':
        labels = data['org_trainlabels'] if org_eval else data['new_trainlabels']  # string labels for each image
    else:
        labels = data['org_testlabels']
    le = data['original_le']
    print(fc1.shape, len(labels))
    dict_fow_avg = {}
    dict_adjusted_mutual_info = {}
    dict_normalized_mutual_info = {}
    dict_adjusted_rand = {}
    dict_silhouette = {}
    dict_cluster_labels = {}
    c_params = cfg['clustering_params']
    # todo: continue modify parameters
    if not c_params['kmean']['use_cache'] or not os.path.isfile(
            c_params['kmean']['cache_path']):
        y_gt = le.transform(labels)  # integer labels for each image
        x = decrease_dim(args, fc1, data)
        print("done decrease dimension", x.shape)
        k_values = np.arange(args.k_min, args.k_max)
        acc_k = np.zeros(k_values.shape)
        rs = np.random.RandomState(seed=args.seed)
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

            adjusted_mutual_info = adjusted_mutual_info_score(y_gt, cluster_labels)
            adjusted_mutual_info = round(float(adjusted_mutual_info), 3)
            dict_adjusted_mutual_info[k] = adjusted_mutual_info

            normalized_mutual_info = normalized_mutual_info_score(y_gt, cluster_labels)
            normalized_mutual_info = round(float(normalized_mutual_info), 3)
            dict_normalized_mutual_info[k] = normalized_mutual_info

            adjusted_rand = adjusted_rand_score(y_gt, cluster_labels)
            adjusted_rand = round(float(adjusted_rand), 3)
            dict_adjusted_rand[k] = adjusted_rand

            # silhouette = silhouette_score(x, cluster_labels)
            # silhouette = round(float(silhouette), 3)
            # dict_silhouette[k] = silhouette
            dict_silhouette[k] = 1

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
        optimal_cluster_list.append(print_optimal(logging, dict_fow_avg, metric='fowlkes_mallows_score', cfg=args))
        optimal_cluster_list.append(
            print_optimal(logging, dict_adjusted_mutual_info, metric='adjusted_mutual_info_score', cfg=args))
        optimal_cluster_list.append(
            print_optimal(logging, dict_normalized_mutual_info, metric='normalized_mutual_info_score', cfg=args))
        optimal_cluster_list.append(print_optimal(logging, dict_adjusted_rand, metric='adjusted_rand_score', cfg=args))
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

# if __name__ == '__main__':
#     umap_clustering()
