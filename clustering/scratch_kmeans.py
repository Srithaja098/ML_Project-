"""NumPy-only K-Means required by the assignment, evaluated on bundled Iris data."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from config import OUTPUTS_DIR, SEED

OUT = OUTPUTS_DIR / "assignment_clustering"


def kmeans(X, k, seed=SEED, max_iter=300, tol=1e-6):
    """Lloyd iterations with seeded k-means++ initialization and safe empty-cluster repair."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not 1 <= k <= len(X) or not np.isfinite(X).all():
        raise ValueError("X must be finite 2D data and k must be in [1, n_samples]")
    rng = np.random.default_rng(seed)
    centers = [X[rng.integers(len(X))]]
    closest_sq = np.sum((X - centers[0]) ** 2, axis=1)
    for _ in range(1, k):
        total = closest_sq.sum()
        idx = rng.choice(len(X), p=closest_sq / total) if total > 0 else rng.integers(len(X))
        centers.append(X[idx])
        closest_sq = np.minimum(closest_sq, np.sum((X - X[idx]) ** 2, axis=1))
    centers = np.asarray(centers).copy()
    for iteration in range(1, max_iter + 1):
        distances = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = distances.argmin(axis=1)
        updated = centers.copy()
        nearest_error = distances.min(axis=1)
        for cluster in range(k):
            members = X[labels == cluster]
            if len(members):
                updated[cluster] = members.mean(axis=0)
            else:
                # Move the empty center to the currently worst represented observation.
                replacement = int(np.argmax(nearest_error))
                updated[cluster] = X[replacement]
                nearest_error[replacement] = -1
        shift = np.linalg.norm(updated - centers)
        centers = updated
        if shift <= tol:
            break
    distances = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = distances.argmin(axis=1)
    wcss = float(np.sum((X - centers[labels]) ** 2))
    return labels, centers, wcss, iteration


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_iris()
    X = StandardScaler().fit_transform(data.data)
    rows = []
    for k in range(2, 7):
        labels, centers, wcss, iters = kmeans(X, k)
        rows.append({"k": k, "wcss": wcss, "iterations": iters, "clusters_found": len(np.unique(labels))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "wcss.csv", index=False)
    (OUT / "metadata.json").write_text(json.dumps({"dataset": "scikit-learn Iris", "features": list(data.feature_names),
        "preprocessing": "StandardScaler; all rows used for unsupervised fit", "seed": SEED,
        "algorithm": "NumPy Lloyd K-Means with k-means++ initialization", "results": rows}, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.plot(df.k, df.wcss, marker="o", lw=2)
    ax.set(xticks=list(range(2, 7)), xlabel="Number of clusters (K)", ylabel="Within-cluster sum of squares (WCSS)", title="K-Means Elbow Curve (Iris)")
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(OUT / "elbow_curve.png", dpi=180); plt.close(fig)
    print(df.to_string(index=False)); return df


if __name__ == "__main__":
    run()
