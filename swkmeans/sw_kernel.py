"""Sliced Wasserstein distance and kernel for persistence diagrams (Carriere et al., 2017)."""
import numpy as np


def _diagonal_projection(dgm):
    if len(dgm) == 0:
        return np.zeros((0, 2))
    mid = (dgm[:, 0] + dgm[:, 1]) / 2.0
    return np.column_stack([mid, mid])


def sw_distance(dgm1, dgm2, n_directions=50):
    """Sliced Wasserstein distance between two persistence diagrams.

    Each diagram is augmented with the diagonal projections of the other, both are projected
    onto `n_directions` equally spaced lines through the origin with angles in [-pi/2, pi/2),
    and the mean over directions of the 1-D optimal transport cost is returned.
    """
    dgm1, dgm2 = np.asarray(dgm1, dtype=float).reshape(-1, 2), np.asarray(dgm2, dtype=float).reshape(-1, 2)
    if len(dgm1) + len(dgm2) == 0:
        return 0.0
    a1 = np.vstack([dgm1, _diagonal_projection(dgm2)])
    a2 = np.vstack([dgm2, _diagonal_projection(dgm1)])
    thetas = np.linspace(-np.pi / 2, np.pi / 2, n_directions, endpoint=False)
    dirs = np.column_stack([np.cos(thetas), np.sin(thetas)])
    p1 = np.sort(a1 @ dirs.T, axis=0)
    p2 = np.sort(a2 @ dirs.T, axis=0)
    return float(np.abs(p1 - p2).sum() / n_directions)


def sw_distance_matrix(diagrams, n_directions=50):
    """Matrix S with S[n, m] = SW(diagrams[n], diagrams[m])."""
    n = len(diagrams)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            S[i, j] = S[j, i] = sw_distance(diagrams[i], diagrams[j], n_directions)
    return S


def median_bandwidth(S):
    """Bandwidth sigma of Algorithm 2: the median of the off-diagonal distances.

    If that median is zero (e.g. most diagrams are identical), the median of the non-zero
    distances is used; if every distance is zero, 1.0 is returned.
    """
    off = S[np.triu_indices(len(S), 1)]
    med = float(np.median(off))
    if med > 0:
        return med
    nz = off[off > 0]
    return float(np.median(nz)) if len(nz) else 1.0


def gaussian_kernel(S, sigma):
    """K = exp(-S / (2 sigma^2)) with unit diagonal."""
    K = np.exp(-S / (2.0 * sigma ** 2))
    np.fill_diagonal(K, 1.0)
    return K


def sw_gram(diagrams, n_directions=50):
    """Gram matrix of the SW kernel on a list of diagrams, with the median bandwidth.

    Returns (K, sigma).
    """
    S = sw_distance_matrix(diagrams, n_directions)
    sigma = median_bandwidth(S)
    return gaussian_kernel(S, sigma), sigma


def sw_grams_by_degree(diagrams_per_cloud, n_directions=50):
    """One SW Gram matrix per homology degree (Stage 1 of Algorithm 2).

    diagrams_per_cloud[n][q] is the degree-q diagram of cloud n.
    Returns (list of Gram matrices K^(q), list of bandwidths sigma_q).
    """
    n_degrees = len(diagrams_per_cloud[0])
    grams, sigmas = [], []
    for q in range(n_degrees):
        K, s = sw_gram([d[q] for d in diagrams_per_cloud], n_directions)
        grams.append(K)
        sigmas.append(s)
    return grams, sigmas
