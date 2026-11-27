"""Convex combination of per-degree SW kernels and weight selection (Stages 2 and 3 of Algorithm 2)."""
import itertools

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .kernel_kmeans import kernel_kmeans


def simplex_grid(n_degrees, step=0.1, include_uniform=True):
    """Weight vectors lambda on the simplex with the given step.

    For three degrees and step 0.1 this gives the 66 triples of the paper; with
    `include_uniform` the uniform vector (1/3, 1/3, 1/3) is appended when it is not on the grid.
    """
    m = int(round(1.0 / step))
    grid = [tuple(round(c / m, 4) for c in combo) + (round((m - sum(combo)) / m, 4),)
            for combo in itertools.product(range(m + 1), repeat=n_degrees - 1) if sum(combo) <= m]
    if include_uniform:
        uniform = (round(1.0 / n_degrees, 4),) * n_degrees
        if uniform not in grid:
            grid.append(uniform)
    return grid


def combine(grams, weights):
    """K_lambda = sum_q lambda_q K^(q)."""
    K = np.zeros_like(grams[0])
    for w, Kq in zip(weights, grams):
        K = K + w * Kq
    return K


def rkhs_distances(K):
    """Feature-space distance d(i, j) = sqrt(K_ii + K_jj - 2 K_ij) induced by the kernel."""
    d = np.diag(K)
    D2 = d[:, None] + d[None, :] - 2.0 * K
    return np.sqrt(np.maximum(D2, 0.0))


def kernel_silhouette(K, labels):
    """Silhouette score of a partition under the RKHS metric of K (needs no labels)."""
    if len(np.unique(labels)) < 2:
        return -1.0
    return float(silhouette_score(rkhs_distances(K), labels, metric="precomputed"))


def select_weights(grams, n_clusters, grid=None, step=0.1, n_init=10, max_iter=100, seed=0, y_true=None):
    """Run kernel k-means for every weight vector on the grid and pick one."""
    if grid is None:
        grid = simplex_grid(len(grams), step)
    labels, sils = [], []
    for w in grid:
        K = combine(grams, w)
        lab = kernel_kmeans(K, n_clusters, n_init=n_init, max_iter=max_iter, seed=seed)
        labels.append(lab)
        sils.append(kernel_silhouette(K, lab))
    i_sil = int(np.argmax(sils))
    out = {"weights": grid, "labels": labels, "silhouette": sils,
           "selected_weight": grid[i_sil], "selected_labels": labels[i_sil]}
    if y_true is not None:
        aris = [adjusted_rand_score(y_true, lab) for lab in labels]
        i_or = int(np.argmax(aris))
        out.update({"ari": aris, "oracle_weight": grid[i_or], "oracle_labels": labels[i_or]})
    return out
