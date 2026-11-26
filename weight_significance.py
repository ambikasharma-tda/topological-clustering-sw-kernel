"""Weight-significance experiment.

Three two-class datasets S0, S1, S2 in which the class signal is planted in a single homology
degree q* (H0: two vs three blobs; H1: circle vs figure-eight; H2: sphere vs punctured sphere).
For every trial the weights of the convex combination of the H0, H1, H2 SW kernels are chosen
by centred kernel-target alignment, and we check whether the largest weight lands on q*.

    python weight_significance.py            # full run: 3 datasets x 7 noise levels x 50 trials
    python weight_significance.py --quick    # short test: 5 trials, noise levels 0, 1 and 2

The full run uses the seeds of the paper and reproduces Tables 8-11 (about 1.5 hours on Colab CPU).
"""
import argparse
import time

import numpy as np

from swkmeans import alpha_diagrams, ripser_diagrams, keep_most_persistent, sw_grams_by_degree
from swkmeans import simplex_grid, select_weights_by_alignment

# ---------------- settings (Appendix B, Table 15) ----------------
N_PER_CLASS = 10                          # clouds per class
N_PTS       = {0: 120, 1: 120, 2: 300}    # points per cloud
NOISE       = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
N_TRIALS    = 50                          # trial t of S_q at noise s uses seed 1000 q + 100 s + t
KEEP_TOP    = 3                           # most persistent points kept per degree (essential bars dropped)
N_DIR       = 30                          # number L of SW directions
GRID_STEP   = 0.05                        # 231 weight triples
NAMES       = ["H0", "H1", "H2"]


# ---------------- datasets ----------------
def circle(n, rng, r=1.0):
    t = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def blobs(n, centers, sd, rng):
    k, per, pts = len(centers), n // len(centers), []
    for i, c in enumerate(centers):
        m = per if i < k - 1 else n - per * (k - 1)
        pts.append(rng.normal(c, sd, size=(m, 2)))
    return np.vstack(pts)


def figure_eight(n, rng):
    n1 = n // 2
    return np.vstack([circle(n1, rng) + [-1, 0], circle(n - n1, rng) + [1, 0]])


def sphere(n, rng, r=1.0):
    v = rng.normal(size=(n, 3))
    return r * v / np.linalg.norm(v, axis=1, keepdims=True)


def punctured_sphere(n, rng, r=1.0, zcap=0.5):
    """Unit sphere with the cap z > zcap removed (a disk), sampled from the same surface as the sphere."""
    pts = []
    while len(pts) < n:
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        if v[2] <= zcap:
            pts.append(v)
    return r * np.array(pts)


SAMPLERS = {
    0: (lambda n, rng: blobs(n, [[-10, 0], [10, 0]], 0.02, rng),
        lambda n, rng: blobs(n, [[-10, 0], [10, 0], [0, 17]], 0.02, rng)),
    1: (circle, figure_eight),
    2: (sphere, punctured_sphere),
}


def preprocess(X, s, rng):
    """Centre and divide by the global standard deviation, add Uniform[-s/2, s/2] noise, then rotate randomly."""
    X = (X - X.mean(0)) / (X.std() + 1e-9)
    X = X + s * rng.uniform(-0.5, 0.5, size=X.shape)
    d = X.shape[1]
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return X @ Q


def make_dataset(q_star, s, rng):
    gA, gB = SAMPLERS[q_star]
    n = N_PTS[q_star]
    clouds = [preprocess(gA(n, rng), s, rng) for _ in range(N_PER_CLASS)] + \
             [preprocess(gB(n, rng), s, rng) for _ in range(N_PER_CLASS)]
    return clouds, np.array([0] * N_PER_CLASS + [1] * N_PER_CLASS)


def diagrams(X, q_star):
    """Vietoris-Rips (ripser) for the planar S0, S1; alpha complex for S2 in R^3; top-3 points per degree."""
    d = alpha_diagrams(X, max_dim=2) if q_star == 2 else ripser_diagrams(X, max_dim=2)
    return keep_most_persistent(d, KEEP_TOP)


# ---------------- experiment ----------------
def run(q_star, noise, n_trials, grid):
    res = {}
    t0 = time.time()
    for s in noise:
        lams, singles = [], []
        for t in range(n_trials):
            rng = np.random.default_rng(1000 * q_star + int(100 * s) + t)
            clouds, y = make_dataset(q_star, s, rng)
            grams, _ = sw_grams_by_degree([diagrams(X, q_star) for X in clouds], N_DIR)
            lam, single = select_weights_by_alignment(grams, y, grid)
            lams.append(lam)
            singles.append(single)
        lams, singles = np.array(lams), np.array(singles)
        hits = (lams.argmax(1) == q_star).astype(float)
        res[s] = {"lam": lams, "align": singles, "hit": hits}
        print(f"  s = {s:<4}  lambda* = {np.round(lams.mean(0), 3)}  hit = {hits.mean():.2f}  "
              f"alignment = {np.round(singles.mean(0), 3)}  [{time.time() - t0:.0f}s]", flush=True)
    return res


def ms(x):
    return f"{np.mean(x):.2f} +/- {np.std(x, ddof=1):.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="5 trials at noise 0, 1 and 2 only")
    ap.add_argument("--datasets", type=int, nargs="+", default=[0, 1, 2], help="which of S0, S1, S2 to run")
    args = ap.parse_args()

    noise = [0, 1.0, 2.0] if args.quick else NOISE
    n_trials = 5 if args.quick else N_TRIALS
    grid = simplex_grid(3, GRID_STEP, include_uniform=False)
    print(f"{len(grid)} weight triples, {n_trials} trials per noise level, noise levels {noise}")

    R = {}
    for q in args.datasets:
        print(f"\n=== S{q} (signal planted in H{q}) ===")
        R[q] = run(q, noise, n_trials, grid)

    print("\nTable 11 (mean +/- std over all trials of each dataset)")
    for key, title in [("lam", "weight-confusion matrix W (mean selected weight)"), ("align", "centred alignment A(K_Hq, Y)")]:
        print(f"\n  {title}")
        print("  planted   " + "".join(f"-> {n:<16}" for n in NAMES))
        diag = []
        for q in args.datasets:
            v = np.concatenate([R[q][s][key] for s in noise])
            print(f"  H{q:<8}" + "".join(f"{ms(v[:, j]):<19}" for j in range(3)))
            diag.append(v[:, q].mean())
        print(f"  diagonal mass = {np.mean(diag):.2f}")
    for q in args.datasets:
        h = np.concatenate([R[q][s]["hit"] for s in noise])
        print(f"\n  S{q}: hit rate over all trials = {h.mean():.2f}")


if __name__ == "__main__":
    main()
