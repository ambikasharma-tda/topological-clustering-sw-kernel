"""Topological clustering with the sliced Wasserstein kernel.

Code for: A. Sharma, V. Aithal and A. Kumar, "Topological Clustering via Sliced Wasserstein Kernels",
Transactions on Machine Learning Research (2026).
"""
from .diagrams import rips_diagrams, alpha_diagrams, ripser_diagrams, keep_most_persistent
from .sw_kernel import sw_distance, sw_distance_matrix, median_bandwidth, gaussian_kernel, sw_gram, sw_grams_by_degree
from .kernel_kmeans import kernel_kmeans
from .weights import simplex_grid, combine, kernel_silhouette, rkhs_distances, select_weights
from .alignment import centred_alignment, select_weights_by_alignment


def sw_kernel_kmeans(diagrams_per_cloud, n_clusters, n_directions=50, step=0.1,
                     n_init=10, max_iter=100, seed=0, y_true=None):
    """Algorithm 2 of the paper: SW kernel k-means with a convex combination of per-degree kernels.

    Parameters
    ----------
    diagrams_per_cloud : list over clouds; diagrams_per_cloud[n][q] is the (k, 2) array of the
        degree-q persistence diagram of cloud n (e.g. from `rips_diagrams`).
    n_clusters : number of clusters.
    n_directions : number L of SW directions.
    step : step of the weight grid on the simplex.
    y_true : optional labels, used only to report the ARI and the oracle weight.

    Returns the dictionary of `select_weights`, plus 'grams' and 'sigmas'.
    Use result['selected_labels'] and result['selected_weight'] for the label-free output.
    """
    grams, sigmas = sw_grams_by_degree(diagrams_per_cloud, n_directions)
    out = select_weights(grams, n_clusters, step=step, n_init=n_init, max_iter=max_iter,
                         seed=seed, y_true=y_true)
    out.update({"grams": grams, "sigmas": sigmas})
    return out


__all__ = [
    "sw_kernel_kmeans",
    "rips_diagrams", "alpha_diagrams", "ripser_diagrams", "keep_most_persistent",
    "sw_distance", "sw_distance_matrix", "median_bandwidth", "gaussian_kernel", "sw_gram", "sw_grams_by_degree",
    "kernel_kmeans",
    "simplex_grid", "combine", "kernel_silhouette", "rkhs_distances", "select_weights",
    "centred_alignment", "select_weights_by_alignment",
]
