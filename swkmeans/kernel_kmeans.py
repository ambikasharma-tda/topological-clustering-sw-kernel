"""Kernel k-means with k-means++ initialisation in feature space (Algorithm 1)."""
import numpy as np


def kernel_kmeans(K, n_clusters, n_init=10, max_iter=100, seed=0):
    """Cluster with kernel k-means on the Gram matrix K.

    Runs `n_init` restarts (restart r uses seed `seed + r`), each initialised by k-means++
    in the feature space of K, and returns the labels of the restart with the smallest
    within-cluster sum of squared feature-space distances.
    """
    K = np.asarray(K, dtype=float)
    n = K.shape[0]
    diagK = np.diag(K)
    best_labels, best_obj = None, np.inf

    for run in range(n_init):
        rng = np.random.RandomState(seed + run)
        seeds = [rng.randint(n)]
        for _ in range(n_clusters - 1):
            d = np.full(n, np.inf)
            for s in seeds:
                d = np.minimum(d, diagK + K[s, s] - 2 * K[:, s])
            d = np.maximum(d, 0)
            seeds.append(rng.choice(n, p=d / d.sum()) if d.sum() > 0 else rng.randint(n))

        labels = np.column_stack([diagK + K[s, s] - 2 * K[:, s] for s in seeds]).argmin(axis=1)
        for c in range(n_clusters):
            if not np.any(labels == c):
                labels[rng.randint(n)] = c

        for _ in range(max_iter):
            d2 = np.empty((n, n_clusters))
            for c in range(n_clusters):
                idx = np.where(labels == c)[0]
                if len(idx) == 0:
                    d2[:, c] = np.inf
                    continue
                d2[:, c] = diagK - 2.0 * K[:, idx].mean(axis=1) + K[np.ix_(idx, idx)].mean()
            new_labels = d2.argmin(axis=1)
            for c in range(n_clusters):
                if not np.any(new_labels == c):
                    new_labels[d2[np.arange(n), new_labels].argmax()] = c
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

        obj = sum(d2[labels == c, c].sum() for c in range(n_clusters))
        if obj < best_obj:
            best_obj, best_labels = obj, labels.copy()
    return best_labels
