import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import yaml
import os
from sklearn.cluster import KMeans
# from h2o4gpu.solvers import KMeans
# from kmeans_pytorch import kmeans, kmeans_predict
from sklearn.decomposition import PCA
from sklearn.metrics import *
from hdbscan import HDBSCAN
import umap
import faiss
from kneed import KneeLocator
from sklearn.manifold import TSNE
from libs.helper import classification_tools as ct
from libs.helper.visualize import visual, visual_gpu, visual_
from libs.utils.yaml_config import init
from create_pretext_pytorch import extract_feature
# from libs.clustering.VAE.data_loader import VAEDataset
# from libs.clustering.VAE.models.models import VAE
# from libs.pretext.utils import load_state
from libs.clustering.VAE.train import fit
from libs.clustering.VAE.cluster import vae_reduce_dimension


class FaissKMeans:
    def __init__(self, n_clusters=8, n_init=10, max_iter=300, seed=1234):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.seed = seed

    def fit(self, X):
        self.kmeans = faiss.Kmeans(d=X.shape[1],
                                   gpu=True,
                                   k=self.n_clusters,
                                   niter=self.max_iter,
                                   nredo=self.n_init,
                                   seed=self.seed)
        self.kmeans.train(X.astype(np.float32))
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]

    def predict(self, X):
        _, I = self.kmeans.index.search(X.astype(np.float32), 1)

        return np.array([int(n[0]) for n in I])
        # return self.kmeans.index.search(X.astype(np.float32), 1)[1]


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


def run_kmeans(x, nmb_clusters, verbose=False, nredo=50, max_centroid_points=1000000, seed=1234):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape
    clus2 = faiss.Kmeans(d, nmb_clusters, gpu=True, niter=300, seed=seed, nredo=nredo,
                         max_points_per_centroid=max_centroid_points,
                         )
    clus2.train(x)
    _, I = clus2.index.search(x, 1)
    del clus2
    return np.array([int(n[0]) for n in I])


def run_kmeans3(x, nmb_clusters, verbose=False, nredo=50, max_centroid_points=1000000, seed=1234):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape
    clus2 = FaissKMeans(n_clusters=nmb_clusters, n_init=nredo, max_iter=300, seed=seed)
    clus2.fit(x)
    # _, I = clus2.index.search(x, 1)
    # del clus2
    return clus2.predict(x)


def run_kmeans2(x, nmb_clusters, verbose=False, nredo=50, max_centroid_points=1000000, seed=1234):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = seed

    clus.nredo = nredo
    clus.max_points_per_centroid = max_centroid_points
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    # losses = faiss.vector_to_array(clus.obj)
    stats = clus.iteration_stats
    # print(stats)
    losses = np.array([stats.at(i).obj for i in range(stats.size())])
    # if verbose:
    #     print('k-means loss evolution: {0}'.format(losses))
    # print(I)
    del clus, res
    return np.array([int(n[0]) for n in I])


def decrease_dim(cfg, fc1, data=None):
    print(f"decrease_dim by {cfg['reduce_dimension_params']['type']}")
    if cfg['reduce_dimension_params']['type'] == 'umap':
        # x = umap.parametric_umap.ParametricUMAP(
        #     n_neighbors=100,
        #     # spread=.75,
        #     min_dist=0.0,
        #     n_components=2,
        #     random_state=cfg['clustering_params']['seed'],
        #     metric='correlation',
        #     init='random',
        # ).fit_transform(fc1,
        #                 # y=y_gt
        #                 )
        x = umap.UMAP(
            n_neighbors=100,
            # spread=.75,
            min_dist=0.0,
            n_components=2,
            random_state=cfg['clustering_params']['seed'],
            metric='correlation',
            init='random',
        ).fit_transform(fc1,
                        # y=y_gt
                        )
    elif cfg['reduce_dimension_params']['type'] == 'pca':
        n_components = cfg['reduce_dimension_params']['pca_params']['dims']
        whitten = cfg['reduce_dimension_params']['pca_params']['whitten']
        pca = PCA(n_components=n_components, svd_solver='full', whiten=whitten,
                  random_state=cfg['clustering_params']['seed'])
        # pca_nw = PCA(n_components=n_components, svd_solver='full', whiten=False)
        x = pca.fit_transform(fc1)
        # x_nw = pca_nw.fit_transform(fc1)
        # if not whitten:
        #     x = x_nw
    elif cfg['reduce_dimension_params']['type'] == 'vae':
        vaeconfig = cfg['reduce_dimension_params']['vae_params']
        # with open(cfg.vae_cfg, 'r') as file:
        #     try:
        #         vaeconfig = yaml.safe_load(file)
        #     except yaml.YAMLError as exc:
        #         print(exc)

        print("VAE dim")
        vaeconfig['exp_params']['train_data_path'] = cfg.fc1_path_vae
        vaeconfig['logging_params']['save_dir'] = os.path.join(cfg.save_dir, vaeconfig['logging_params']['save_dir'])
        # print(vaeconfig)
        # print('0', vaeconfig['model_params']['hidden_dims'])
        vaeconfig['infer']['weight_path'] = fit(vaeconfig)
        # print('1', vaeconfig['infer']['weight_path'])
        x = vae_reduce_dimension(vaeconfig, data)
    elif cfg['reduce_dimension_params']['type'] == 'none':
        x = fc1

    return x


def decrease_dim_for_visual(fc1):
    _, s1 = fc1.shape
    if s1 != 2:
        tsne = TSNE(n_components=2, random_state=12214)
        fc1 = tsne.fit_transform(fc1)

    return fc1


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
        x = torch.tensor(x)
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
            _, cluster_centers = kmeans(
                X=x, num_clusters=int(k), distance='soft_dtw', device=torch.device('cuda:0'),
                iter_limit=1000,
                tqdm_flag=False,
                tol=1e-10,
                # random_state=c_params['seed']
            )
            # print(cluster_centers)
            _, cluster_centers = kmeans(
                X=x, num_clusters=int(k), distance='soft_dtw', device=torch.device('cuda:0'),
                iter_limit=1000,
                cluster_centers=cluster_centers,
                tqdm_flag=False,
                tol=1e-10,
                # random_state=c_params['seed']
            )
            cluster_labels = kmeans_predict(
                x, cluster_centers, 'soft_dtw', device=torch.device('cuda:0')
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
    print_optimal(logging, dict_calinski_avg, metric='calinski_harabasz_score', cfg=args)
    print_optimal(logging, dict_fow_avg, metric='fowlkes_mallows_score', cfg=args)
    print_optimal(logging, dict_adjusted_mutual_info, metric='adjusted_mutual_info_score', cfg=args)
    print_optimal(logging, dict_normalized_mutual_info, metric='normalized_mutual_info_score', cfg=args)
    print_optimal(logging, dict_adjusted_rand, metric='adjusted_rand_score', cfg=args)


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
