"""Covariance/eigendecomposition PCA, demonstrated on bundled diabetes data."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

from config import OUTPUTS_DIR

OUT = OUTPUTS_DIR / "assignment_pca"


def fit_pca(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or not np.isfinite(X).all():
        raise ValueError("X must be a finite 2D matrix with at least two rows")
    mean = X.mean(axis=0)
    centered = X - mean
    covariance = np.cov(centered, rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    ratios = eigenvalues / eigenvalues.sum()
    return {"mean": mean, "covariance": covariance, "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors, "explained_variance_ratio": ratios,
            "cumulative_explained_variance": np.cumsum(ratios), "centered": centered}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_diabetes(as_frame=True)
    Xraw = data.data.to_numpy()
    scaler = StandardScaler().fit(Xraw)
    X = scaler.transform(Xraw)
    pca = fit_pca(X)
    scores = X @ pca["eigenvectors"]
    pd.DataFrame(scores[:, :2], columns=["PC1", "PC2"]).to_csv(OUT / "diabetes_2_components.csv", index=False)
    pd.DataFrame(scores[:, :3], columns=["PC1", "PC2", "PC3"]).to_csv(OUT / "diabetes_3_components.csv", index=False)
    pd.DataFrame({"component": np.arange(1, X.shape[1] + 1), "eigenvalue": pca["eigenvalues"],
                  "explained_variance_ratio": pca["explained_variance_ratio"],
                  "cumulative_explained_variance": pca["cumulative_explained_variance"]}).to_csv(OUT / "pca_statistics.csv", index=False)
    np.savetxt(OUT / "eigenvectors.csv", pca["eigenvectors"], delimiter=",")
    # Mathematical consistency checks: orthonormal vectors and covariance eigen-equation.
    assert np.allclose(pca["eigenvectors"].T @ pca["eigenvectors"], np.eye(X.shape[1]), atol=1e-10)
    assert np.allclose(pca["covariance"] @ pca["eigenvectors"], pca["eigenvectors"] * pca["eigenvalues"], atol=1e-9)
    assert np.isclose(pca["explained_variance_ratio"].sum(), 1.0)
    (OUT / "metadata.json").write_text(json.dumps({"dataset": "scikit-learn diabetes", "features": list(data.feature_names),
       "preprocessing": "StandardScaler fit on full unsupervised PCA dataset, then explicit centering",
       "n_samples": len(X), "n_features": X.shape[1], "eigenvalues": pca["eigenvalues"].tolist(),
       "explained_variance_ratios": pca["explained_variance_ratio"].tolist(),
       "cumulative_explained_variance": pca["cumulative_explained_variance"].tolist(),
       "dimensions": {"two_components": list(scores[:, :2].shape), "three_components": list(scores[:, :3].shape)},
       "checks": {"eigenvectors_orthonormal": True, "covariance_eigen_equation": True, "ratios_sum_to_one": True}}, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7, 4.5)); comp = np.arange(1, X.shape[1] + 1)
    ax.plot(comp, pca["cumulative_explained_variance"], marker="o"); ax.axhline(.9, color="grey", ls="--", lw=1)
    ax.set(xticks=comp, xlabel="Number of principal components", ylabel="Cumulative explained variance", ylim=(0, 1.03), title="PCA cumulative explained variance")
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(OUT / "cumulative_explained_variance.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5)); scatter = ax.scatter(scores[:, 0], scores[:, 1], c=data.target, cmap="viridis", s=23, alpha=.8)
    ax.set(xlabel="PC1", ylabel="PC2", title="Diabetes dataset projected onto first two PCs"); fig.colorbar(scatter, ax=ax, label="Disease progression")
    fig.tight_layout(); fig.savefig(OUT / "pca_2d_visualization.png", dpi=180); plt.close(fig)
    return pca


if __name__ == "__main__":
    run()
