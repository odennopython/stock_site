"""Nearest-neighbour search over fingerprints, and forecasts built from it."""
import numpy as np

K = 120         # neighbours used for the outcome distribution
PRIOR_W = 10    # pseudo-count pulling event probabilities toward the base rate


def neighbours(emb, t, pool, k=K):
    """Indices in `pool` sorted by distance to month t (closest first)."""
    d = np.linalg.norm(emb[pool] - emb[t], axis=1)
    order = np.argsort(d)[:k]
    return pool[order], d[order]


def episodes(idx, dist, n=5, gap=18):
    """Pick distinct historical episodes: no two picks within `gap` months."""
    picks = []
    for i, d in zip(idx, dist):
        if all(abs(i - j) >= gap for j, _ in picks):
            picks.append((i, d))
        if len(picks) == n:
            break
    return picks


def event_prob(hits, k, base):
    return (hits + PRIOR_W * base) / (k + PRIOR_W)


def logistic_fit(x, y, l2=1.0, iters=50):
    """Ridge logistic regression by Newton's method."""
    X = np.hstack([np.ones((len(x), 1)), x])
    w = np.zeros(X.shape[1])
    reg = np.eye(X.shape[1]) * l2
    reg[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + reg @ w
        H = X.T @ (X * (p * (1 - p))[:, None]) + reg
        w -= np.linalg.solve(H, g)
    return w


def logistic_predict(w, x):
    X = np.hstack([np.ones((len(x), 1)), x])
    return 1 / (1 + np.exp(-X @ w))
