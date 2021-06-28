import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import faiss
from sklearn.manifold import TSNE
import seaborn as sns


def pano_plot(x, y, paths, patch_size=(3, 3), ax0=None):
    """
    Graphs y vs x with images on plot instead of points.

    Generates 'panoramic' image plots which are useful for visualizing how images 
    separate in feature space for clustering and classification challenges.
    
    Parameters
    ---------------
    x, y: ndarray
        n-element arrays of x and y coordinates for plot
        
    paths: list of strings or path objects
        n-element list of paths to images to be displaied at each point
        
    patch_size: tuple(int, int)
        size of the image patches displayed at each point
        
    ax0: None or matplotlib axis object
        if None, a new figure and axis will be created and the visualization will be displayed.
        if an axis is supplied, the panoramic visualization will be plotted on the axis in place.
        
    Returns
    ----------
    None
    
    """
    if ax0 is None:
        fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    else:
        ax = ax0
    px, py = patch_size
    ax.scatter(x, y, color=(0, 0, 0, 0))
    for xi, yi, pi in zip(x, y, paths):
        im = skimage.io.imread(pi)
        ax.imshow(im, extent=(xi - px, xi + px, yi - py, yi + py), cmap='gray')

    if ax0 is None:
        plt.show()


def pretty_cm(cm, labelnames, cscale=0.6, ax0=None, fs=6, cmap='cool'):
    """
    Generates a pretty-formated confusion matrix for convenient visualization.
    
    The true labels are displayed on the rows, and the predicted labels are displayed on the columns.
    
    Parameters
    ----------
    cm: ndarray 
        nxn array containing the data of the confusion matrix.
    
    labelnames: list(string)
        list of class names in order on which they appear in the confusion matrix. For example, the first
        element should contain the class corresponding to the first row and column of *cm*.

    cscale: float
        parameter that adjusts the color intensity. Allows color to be present for confusion matrices with few mistakes,
        and controlling the intensity for ones with many misclassifications.
    
    ax0: None or matplotlib axis object
        if None, a new figure and axis will be created and the visualization will be displayed.
        if an axis is supplied, the confusion matrix will be plotted on the axis in place.

    fs: int
        font size for text on confusion matrix.
        
    cmap: str
        matplotlib colormap to use
    
    Returns
    ---------
    None
    
    """

    acc = cm.trace() / cm.sum()
    if ax0 is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 2.5), dpi=300)
        fig.set_facecolor('w')
    else:
        ax = ax0

    n = len(labelnames)
    ax.imshow(np.power(cm, cscale), cmap=cmap, extent=(0, n, 0, n))
    labelticks = np.arange(n) + 0.5

    ax.set_xticks(labelticks, minor=True)
    ax.set_yticks(labelticks, minor=True)
    ax.set_xticklabels(['' for i in range(n)], minor=False, fontsize=fs)
    ax.set_yticklabels(['' for i in range(n)], minor=False, fontsize=fs)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels=labelnames, minor=True, fontsize=fs)
    ax.set_yticklabels(labels=reversed(labelnames), minor=True, fontsize=fs)

    ax.set_xlabel('Predicted Labels', fontsize=fs)
    ax.set_ylabel('Actual Labels', fontsize=fs)
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j + 0.5, n - i - 0.5, '{:^5}'.format(z), ha='center', va='center', fontsize=fs,
                bbox=dict(boxstyle='round', facecolor='w', edgecolor='0.3'))
    ax.grid(which='major', color=np.ones(3) * 0.33, linewidth=1)

    if ax0 is None:
        ax.set_title('Accuracy: {:.3f}'.format(cm.trace() / cm.sum()), fontsize=fs + 2)
        plt.show()
        return
    else:
        return ax


def visual(y_pred, kmeans, le, x_nw, save_name, cluster):
    # tsne = TSNE(n_components=2, random_state=12214)
    # x_nw_tsne = tsne.fit_transform(x_nw)
    _, s1 = x_nw.shape
    if s1 != 2:
        tsne = TSNE(n_components=2, random_state=12214)
        x_nw = tsne.fit_transform(x_nw)
    cluster_mapper = {}
    for p in np.unique(y_pred):
        y_clusters = kmeans.labels_[y_pred == p]
        for idx, value in enumerate(np.unique(y_clusters)):
            cluster_mapper[value] = '{}-{}'.format(le.inverse_transform([p])[0], idx)
    palette = np.concatenate((sns.color_palette('pastel', cluster), sns.color_palette('dark', cluster)), axis=0)
    hue = [cluster_mapper[x] for x in kmeans.labels_]
    hue_order = sorted(cluster_mapper.values(), key=lambda x: x.upper())

    fig, ax = plt.subplots(dpi=300, figsize=(5, 5))
    sns.scatterplot(x_nw[:, 0], x_nw[:, 1], hue=hue, hue_order=hue_order,
                    palette=dict(zip(hue_order, palette)), ax=ax)
    ax.legend(loc='upper center', bbox_to_anchor=(1, 1))
    plt.savefig(save_name)
    # plt.show()

# def visual(y_pred, labels_unmatched, le, x_nw, save_name, cluster):
#     # from MulticoreTSNE import MulticoreTSNE as TSNE
#     if not isinstance(y_pred, np.ndarray):
#         y_pred = np.array(y_pred)
#     if not isinstance(labels_unmatched, np.ndarray):
#         labels_unmatched = np.array(labels_unmatched)
#     tsne = TSNE(n_components=2, random_state=12214)
#     x_nw_tsne = tsne.fit_transform(x_nw)
#     cluster_mapper = {}
#     for p in np.unique(y_pred):
#         y_clusters = labels_unmatched[y_pred == p]
#         for idx, value in enumerate(np.unique(y_clusters)):
#             cluster_mapper[value] = '{}-{}'.format(le.inverse_transform([p])[0], idx)
#     palette = np.concatenate((sns.color_palette('pastel', cluster), sns.color_palette('dark', cluster)), axis=0)
#     hue = [cluster_mapper[x] for x in labels_unmatched]
#     hue_order = sorted(cluster_mapper.values(), key=lambda x: x.upper())
#
#     fig, ax = plt.subplots(dpi=300, figsize=(5, 5))
#     sns.scatterplot(x_nw_tsne[:, 0], x_nw_tsne[:, 1], hue=hue, hue_order=hue_order,
#                     palette=dict(zip(hue_order, palette)), ax=ax)
#     ax.legend(loc='upper center', bbox_to_anchor=(1, 1))
#     plt.savefig(save_name)
#     # plt.show()


def visual_(y_pred, labels_unmatched, le, x_nw, save_name, cluster):
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    if not isinstance(labels_unmatched, np.ndarray):
        labels_unmatched = np.array(labels_unmatched)
    _, s1 = x_nw.shape
    if s1 != 2:
        tsne = TSNE(n_components=2, random_state=12214)
        x_nw = tsne.fit_transform(x_nw)
    # x_nw_tsne = tsne.fit_transform(x_nw)
    cluster_mapper = {}
    for p in np.unique(y_pred):
        y_clusters = labels_unmatched[y_pred == p]
        for idx, value in enumerate(np.unique(y_clusters)):
            cluster_mapper[value] = '{}-{}'.format(le.inverse_transform([p])[0], idx)
    palette = np.concatenate((sns.color_palette('pastel', cluster), sns.color_palette('dark', cluster)), axis=0)
    hue = [cluster_mapper[x] for x in labels_unmatched]
    hue_order = sorted(cluster_mapper.values(), key=lambda x: x.upper())

    fig, ax = plt.subplots(dpi=300, figsize=(5, 5))
    sns.scatterplot(x_nw[:, 0], x_nw[:, 1], hue=hue, hue_order=hue_order,
                    palette=dict(zip(hue_order, palette)), ax=ax)
    ax.legend(loc='upper center', bbox_to_anchor=(1, 1))
    plt.savefig(save_name)


def visual_tsne_pytorch(y_pred, labels_unmatched, le, x_nw, save_name, cluster):
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    if not isinstance(labels_unmatched, np.ndarray):
        labels_unmatched = np.array(labels_unmatched)
    # from MulticoreTSNE import MulticoreTSNE as TSNE
    from tsne_torch import TorchTSNE as TSNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=100, verbose=True)
    a = np.array_split(x_nw, 5)
    list_tsne = []
    for i in a:
        print(i.shape)
        i_tsne = tsne.fit_transform(i)
        list_tsne.append(i_tsne)
    x_nw_tsne = np.concatenate(list_tsne, axis=0)
    print(x_nw_tsne.shape)
    print(labels_unmatched)
    cluster_mapper = {}
    for p in np.unique(y_pred):
        y_clusters = labels_unmatched[y_pred == p]
        for idx, value in enumerate(np.unique(y_clusters)):
            cluster_mapper[value] = '{}-{}'.format(le.inverse_transform([p])[0], idx)
    palette = np.concatenate((sns.color_palette('pastel', cluster), sns.color_palette('dark', cluster)), axis=0)
    hue = [cluster_mapper[x] for x in labels_unmatched]
    hue_order = sorted(cluster_mapper.values(), key=lambda x: x.upper())

    fig, ax = plt.subplots(dpi=300, figsize=(5, 5))
    sns.scatterplot(x_nw_tsne[:, 0], x_nw_tsne[:, 1], hue=hue, hue_order=hue_order,
                    palette=dict(zip(hue_order, palette)), ax=ax)
    ax.legend(loc='upper center', bbox_to_anchor=(1, 1))
    plt.savefig(save_name)
    # plt.show()


def visual_gpu(y_pred, labels_unmatched, le, x_nw, save_name, cluster):
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    if not isinstance(labels_unmatched, np.ndarray):
        labels_unmatched = np.array(labels_unmatched)
    print(x_nw.shape)
    _, ndim = x_nw.shape
    npdata = x_nw.astype('float32')

    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix(ndim, 2)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)
    # tsne = TSNE(n_components=2, random_state=12214)
    # x_nw_tsne = tsne.fit_transform(x_nw)
    cluster_mapper = {}
    for p in np.unique(y_pred):
        y_clusters = labels_unmatched[y_pred == p]
        for idx, value in enumerate(np.unique(y_clusters)):
            cluster_mapper[value] = '{}-{}'.format(le.inverse_transform([p])[0], idx)
    palette = np.concatenate((sns.color_palette('pastel', cluster), sns.color_palette('dark', cluster)), axis=0)
    hue = [cluster_mapper[x] for x in labels_unmatched]
    hue_order = sorted(cluster_mapper.values(), key=lambda x: x.upper())

    fig, ax = plt.subplots(dpi=300, figsize=(5, 5))
    sns.scatterplot(npdata[:, 0], npdata[:, 1], hue=hue, hue_order=hue_order,
                    palette=dict(zip(hue_order, palette)), ax=ax)
    ax.legend(loc='upper center', bbox_to_anchor=(1, 1))
    plt.savefig(save_name)
    # plt.show()
