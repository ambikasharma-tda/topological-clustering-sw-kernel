"""Persistence diagrams of point clouds, split by homology degree."""
import numpy as np
import gudhi
from scipy.spatial.distance import cdist


def enclosing_radius(cloud):
   
    D = cdist(cloud, cloud)
    return float(np.min(np.max(D, axis=1)))


def rips_diagrams(cloud, max_dim=2):
    
    cloud = np.asarray(cloud, dtype=float)
    r_max = enclosing_radius(cloud)
    st = gudhi.RipsComplex(points=cloud, max_edge_length=r_max).create_simplex_tree(
        max_dimension=max_dim + 1)
    st.persistence()
    dgms = []
    for q in range(max_dim + 1):
        dgm = np.array(st.persistence_intervals_in_dimension(q), dtype=float).reshape(-1, 2)
        dgm[np.isinf(dgm)] = r_max
        dgm = dgm[dgm[:, 1] > dgm[:, 0] + 1e-12]
        if len(dgm) == 0:
            dgm = np.array([[0.0, 0.01 * r_max]])
        dgms.append(dgm)
    return dgms


def alpha_diagrams(cloud, max_dim=2):
    
    st = gudhi.AlphaComplex(points=np.asarray(cloud, dtype=float)).create_simplex_tree()
    pers = st.persistence()
    out = [[] for _ in range(max_dim + 1)]
    for q, (b, d) in pers:
        if q <= max_dim and np.isfinite(d):
            out[q].append([np.sqrt(b), np.sqrt(d)])
    return [np.array(p) if p else np.zeros((0, 2)) for p in out]


def ripser_diagrams(cloud, max_dim=2):
   
    from ripser import ripser
    dgms = ripser(np.asarray(cloud, dtype=float), maxdim=max_dim)["dgms"]
    return [(d[np.isfinite(d).all(1)] if len(d) else np.zeros((0, 2))) for d in dgms]


def keep_most_persistent(dgms, k):
    
    out = []
    for d in dgms:
        if len(d) > k:
            pers = d[:, 1] - d[:, 0]
            d = d[np.argsort(pers)[-k:]]
        out.append(d)
    return out
