# Topological Clustering via Sliced Wasserstein Kernels

The method clusters point clouds by their topology. For each cloud it computes the persistence
diagrams of the homology degrees H0, H1, H2, builds one sliced Wasserstein (SW) kernel per degree,
combines them as a convex combination

    K_lambda = lambda_0 K_H0 + lambda_1 K_H1 + lambda_2 K_H2,

and runs kernel k-means on `K_lambda`. 

## Installation

```bash
git clone https://github.com/ambikasharma-tda/sw-kernel-kmeans.git
cd sw-kernel-kmeans
pip install -r requirements.txt
```

## Files

| File | Contents |
|---|---|
| `swkmeans/diagrams.py` | Persistence diagrams: Vietoris-Rips, alpha complex |
| `swkmeans/sw_kernel.py` | SW distance, median bandwidth, SW Gram matrix per homology degree (Stage 1) |
| `swkmeans/kernel_kmeans.py` | Kernel k-means with k-means++ initialisation (Algorithm 1) |
| `swkmeans/weights.py` | Weight grid on the simplex, convex combination, silhouette selection (Stages 2-3) |
| `swkmeans/alignment.py` | Centred kernel-target alignment  |
| `example.py` | Quick example on the circle-sphere-torus |
| `weight_significance.py` | Weight-significance experiment|
