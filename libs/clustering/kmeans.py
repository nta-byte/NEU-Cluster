from hdbscan import HDBSCAN
import faiss
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import euclidean_distances
from sklearn.utils.extmath import row_norms, stable_cumsum
from sklearn.utils import check_array


def _kmeans_plusplus(X, n_clusters, x_squared_norms,
                     random_state, n_local_trials=None):
    """Computational component for initialization of n_clusters by
    k-means++. Prior validation of data is assumed.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The data to pick seeds for.

    n_clusters : int
        The number of seeds to choose.

    x_squared_norms : ndarray of shape (n_samples,)
        Squared Euclidean norm of each data point.

    random_state : RandomState instance
        The generator used to initialize the centers.
        See :term:`Glossary <random_state>`.

    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The inital centers for k-means.

    indices : ndarray of shape (n_clusters,)
        The index location of the chosen centers in the data array X. For a
        given index and center, X[index] = center.
    """
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly and track index of point
    center_id = random_state.randint(n_samples)
    indices = np.full(n_clusters, -1, dtype=int)
    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]
    indices[0] = center_id

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
        squared=True)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                        rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1,
                out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates,
                   out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]
        indices[c] = best_candidate

    return centers, indices


class FaissKMeans:
    def __init__(self, n_clusters=8, n_init=10, max_iter=300, seed=1234):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.seed = seed

    def _init_centroids(self, X, x_squared_norms, init, random_state,
                        init_size=None):
        """Compute the initial centroids.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        x_squared_norms : ndarray of shape (n_samples,)
            Squared euclidean norm of each data point. Pass it if you have it
            at hands already to avoid it being recomputed here.

        init : {'k-means++', 'random'}, callable or ndarray of shape \
                (n_clusters, n_features)
            Method for initialization.

        random_state : RandomState instance
            Determines random number generation for centroid initialization.
            See :term:`Glossary <random_state>`.

        init_size : int, default=None
            Number of samples to randomly sample for speeding up the
            initialization (sometimes at the expense of accuracy).

        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
        """
        n_samples = X.shape[0]
        n_clusters = self.n_clusters

        if init_size is not None and init_size < n_samples:
            init_indices = random_state.randint(0, n_samples, init_size)
            X = X[init_indices]
            x_squared_norms = x_squared_norms[init_indices]
            n_samples = X.shape[0]

        if isinstance(init, str) and init == 'k-means++':
            centers, _ = _kmeans_plusplus(X, n_clusters,
                                          random_state=random_state,
                                          x_squared_norms=x_squared_norms)
        elif isinstance(init, str) and init == 'random':
            seeds = random_state.permutation(n_samples)[:n_clusters]
            centers = X[seeds]
        elif hasattr(init, '__array__'):
            centers = init
        elif callable(init):
            centers = init(X, n_clusters, random_state=random_state)
            centers = check_array(
                centers, dtype=X.dtype, copy=False, order='C')
            self._validate_center_shape(X, centers)

        if sp.issparse(centers):
            centers = centers.toarray()

        return centers

    def _validate_center_shape(self, X, centers):
        """Check if centers is compatible with X and n_clusters."""
        if centers.shape[0] != self.n_clusters:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of clusters {self.n_clusters}.")
        if centers.shape[1] != X.shape[1]:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of features of the data {X.shape[1]}.")

    def fit(self, X):
        rs = np.random.RandomState(seed=self.seed)
        best_inertia = None

        for i in range(self.n_init):
            x_squared_norms = row_norms(X, squared=True)
            centers_init, _ = _kmeans_plusplus(X, self.n_clusters,
                                               random_state=rs,
                                               x_squared_norms=x_squared_norms)
            kmeans = faiss.Kmeans(d=X.shape[1],
                                  gpu=True,
                                  k=self.n_clusters,
                                  niter=self.max_iter,
                                  # nredo=self.n_init,
                                  seed=rs.randint(2 ** 30))
            kmeans.train(X.astype(np.float32), init_centroids=centers_init)
            centers = kmeans.centroids

            inertia = kmeans.obj[-1]

            # determine if these results are the best so far
            if best_inertia is None or inertia < best_inertia:
                self.kmeans = kmeans
                # best_labels = labels
                self.centers = centers
                best_inertia = inertia
        self.inertia_ = best_inertia

    def predict(self, X):
        _, I = self.kmeans.index.search(X.astype(np.float32), 1)

        return np.array([int(n[0]) for n in I])
        # return self.kmeans.index.search(X.astype(np.float32), 1)[1]


def run_kmeans3(X, nmb_clusters, verbose=False, n_init=50, max_centroid_points=1000000, seed=1234):
    """Runs kmeans on 1 GPU.
    Args:
        X: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = X.shape
    # X = torch.from_numpy(X)
    clus2 = FaissKMeans(n_clusters=nmb_clusters, n_init=n_init, max_iter=300, seed=seed)
    clus2.fit(X)
    # _, I = clus2.index.search(x, 1)
    # del clus2
    return clus2.predict(X)


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
