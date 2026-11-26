"""Example: SW kernel k-means on the circle-sphere-torus data at one noise level.

Generates 10 circles, 10 spheres and 10 tori (100 points each),
computes the H0, H1, H2 Vietoris-Rips diagrams, and runs Algorithm 2.

"""
import numpy as np
from sklearn.metrics import adjusted_rand_score

from swkmeans import rips_diagrams, sw_kernel_kmeans


def make_point_clouds(n_per_class, n_points, noise):
    def jitter(k):
        return noise * (np.random.rand(k) - 0.5)
    circles, spheres, tori = [], [], []
    for _ in range(n_per_class):
        t = np.arange(n_points ** 2, dtype=float)
        circles.append(np.column_stack([np.sin(t) + jitter(len(t)), np.cos(t) + jitter(len(t)), np.zeros(len(t))]))
        s, t = np.meshgrid(np.arange(n_points, dtype=float), np.arange(n_points, dtype=float), indexing="ij")
        s, t = s.ravel(), t.ravel()
        spheres.append(np.column_stack([np.cos(s) * np.cos(t) + jitter(len(s)),
                                        np.cos(s) * np.sin(t) + jitter(len(s)),
                                        np.sin(s) + jitter(len(s))]))
        tori.append(np.column_stack([(2 + np.cos(s)) * np.cos(t) + jitter(len(s)),
                                     (2 + np.cos(s)) * np.sin(t) + jitter(len(s)),
                                     np.sin(s) + jitter(len(s))]))
    return circles + spheres + tori, np.repeat(np.arange(3), n_per_class)


def minmax_normalise(cloud):
    mn, mx = cloud.min(), cloud.max()
    return ((cloud - mn) / (mx - mn if mx > mn else 1.0) - 0.5) * 2.0


if __name__ == "__main__":
    np.random.seed(0)                                  
    clouds, y = make_point_clouds(n_per_class=10, n_points=10, noise=1.0)

    print("computing persistence diagrams ...")
    diagrams = [rips_diagrams(minmax_normalise(c), max_dim=2) for c in clouds]

    print("running SW kernel k-means ...")
    res = sw_kernel_kmeans(diagrams, n_clusters=3, n_directions=50, step=0.1, seed=0, y_true=y)

    print(f"bandwidths sigma_q (H0, H1, H2): {np.round(res['sigmas'], 4)}")
    print(f"silhouette-selected weight: {res['selected_weight']}   "
          f"ARI = {adjusted_rand_score(y, res['selected_labels']):.4f}")
    print(f"oracle weight (uses labels): {res['oracle_weight']}   ARI = {max(res['ari']):.4f}")
