import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import gpumap
from libs.clustering.VAE.train import fit
from libs.clustering.VAE.cluster import vae_reduce_dimension


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
        # x = gpumap.GPUMAP(
        #     n_neighbors=100,
        #     # spread=.75,
        #     min_dist=0.0,
        #     n_components=2,
        #     random_state=cfg['clustering_params']['seed'],
        #     metric='correlation',
        #     init='random',).fit_transform(fc1)
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
