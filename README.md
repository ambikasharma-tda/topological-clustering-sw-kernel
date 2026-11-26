# Topological Clustering via Sliced Wasserstein Kernels

Code for the paper

> Ambika Sharma, Vikram Aithal and Ajit Kumar. **Topological Clustering via Sliced Wasserstein Kernels.**
> *Transactions on Machine Learning Research*, 2026. [OpenReview](https://openreview.net/forum?id=V418l9gQWy)

The method clusters point clouds by their topology. For each cloud it computes the persistence
diagrams of the homology degrees H0, H1, H2, builds one sliced Wasserstein (SW) kernel per degree,
combines them as a convex combination

    K_lambda = lambda_0 K_H0 + lambda_1 K_H1 + lambda_2 K_H2,

and runs kernel k-means on `K_lambda`. The weights are chosen without labels by the silhouette
score in the kernel's feature space (Algorithm 2 of the paper).

## Installation

```bash
git clone https://github.com/<username>/sw-kernel-kmeans.git
cd sw-kernel-kmeans
pip install -r requirements.txt
```

Tested with Python 3.13, gudhi 3.13.0, ripser 0.6, NumPy, SciPy and scikit-learn.

## Usage

```python
from swkmeans import rips_diagrams, sw_kernel_kmeans

# clouds: list of (n_points, d) arrays
diagrams = [rips_diagrams(c, max_dim=2) for c in clouds]   # H0, H1, H2 diagrams of each cloud
res = sw_kernel_kmeans(diagrams, n_clusters=3)

res["selected_labels"]    # cluster labels (label-free choice of weights)
res["selected_weight"]    # chosen weights (lambda_0, lambda_1, lambda_2)
```

Pass `y_true=labels` to also get the ARI of every weight and the oracle weight.

## Files

| File | Contents |
|---|---|
| `swkmeans/diagrams.py` | Persistence diagrams: Vietoris-Rips (truncated at the enclosing radius), alpha complex |
| `swkmeans/sw_kernel.py` | SW distance, median bandwidth, SW Gram matrix per homology degree (Stage 1) |
| `swkmeans/kernel_kmeans.py` | Kernel k-means with k-means++ initialisation (Algorithm 1) |
| `swkmeans/weights.py` | Weight grid on the simplex, convex combination, silhouette selection (Stages 2-3) |
| `swkmeans/alignment.py` | Centred kernel-target alignment (Section 4.6.3) |
| `example.py` | Quick example on the circle-sphere-torus data (Section 4.2, one trial, about 20 s) |
| `weight_significance.py` | Weight-significance experiment (Section 4.6, Tables 8-11) |

## Examples

```bash
python example.py                         # one trial of the circle-sphere-torus data at noise s = 1
python weight_significance.py --quick     # short test of the weight-significance experiment (about 2 min)
python weight_significance.py             # full run, reproduces Tables 8-11 (about 1.5 h)
```

`weight_significance.py` uses the seeds of the paper (trial t of dataset S_q at noise s uses
seed 1000q + 100s + t), so the full run gives the values in Tables 8-11. The settings are those
of Appendix B, Table 15.

This repository contains the method and the weight-significance experiment. Scripts for the
other experiments in the paper are available from the authors on request.

## Citation

```bibtex
@article{sharma2026topological,
  title   = {Topological Clustering via Sliced {W}asserstein Kernels},
  author  = {Sharma, Ambika and Aithal, Vikram and Kumar, Ajit},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=V418l9gQWy}
}
```

## License

MIT License. See `LICENSE`.
