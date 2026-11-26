"""Centred kernel-target alignment (Cortes et al., 2012), used in the weight-significance experiment."""
import numpy as np


def centre(K):
    """Kc = H K H with H = I - (1/n) 1 1^T."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def target_kernel(y):
    """Ideal kernel Y with Y_ij = 1 if y_i = y_j and 0 otherwise."""
    y = np.asarray(y)
    return (y[:, None] == y[None, :]).astype(float)


def centred_alignment(K, y):
    """A(K, Y) = <Kc, Yc>_F / (||Kc||_F ||Yc||_F)."""
    Kc, Yc = centre(K), centre(target_kernel(y))
    return float((Kc * Yc).sum() / (np.linalg.norm(Kc) * np.linalg.norm(Yc) + 1e-12))


def select_weights_by_alignment(grams, y, grid, tol=1e-9):
    """Weight vector on `grid` maximising the centred alignment of K_lambda = sum_q lambda_q K^(q).

    Uses A(K_lambda, Y) = lambda^T b / (sqrt(lambda^T G lambda) ||Yc||) with
    b_q = <Kc_q, Yc> and G_qj = <Kc_q, Kc_j>, so b and G are computed once (equation 22).
    If several grid points attain the maximum (within `tol`), their centroid is returned.

    Returns (selected weight vector, alignment of each single-degree kernel).
    """
    grid = np.asarray(grid, dtype=float)
    Yc = centre(target_kernel(y))
    nYc = np.linalg.norm(Yc)
    Kc = [centre(K) for K in grams]
    p = len(Kc)
    single = np.array([(Kc[q] * Yc).sum() / (np.linalg.norm(Kc[q]) * nYc + 1e-12) for q in range(p)])
    b = np.array([(Kc[q] * Yc).sum() for q in range(p)])
    G = np.array([[(Kc[i] * Kc[j]).sum() for j in range(p)] for i in range(p)])
    scores = (grid @ b) / (np.sqrt(np.einsum("gi,ij,gj->g", grid, G, grid)) * nYc + 1e-12)
    lam = grid[scores >= scores.max() - tol].mean(0)
    return lam, single
