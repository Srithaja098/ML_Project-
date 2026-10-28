"""Build a single assignment report from the actual experiment output files."""
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "ML_Assignment_Report.md"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def md_table(df):
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.4f}"
        return str(value)
    headers = [str(column) for column in df.columns]
    rows = [[cell(value) for value in row] for row in df.itertuples(index=False, name=None)]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    line = lambda values: "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(values)) + " |"
    return "\n".join([line(headers), "| " + " | ".join("-" * width for width in widths) + " |",
                       *(line(row) for row in rows)])


def build():
    class_dir = ROOT / "outputs" / "classification"
    cnn_dir = ROOT / "outputs" / "cnn"
    reg_dir = ROOT / "outputs" / "assignment_regression"
    km_dir = ROOT / "outputs" / "assignment_clustering"
    pca_dir = ROOT / "outputs" / "assignment_pca"
    cmaps = {"knn": "KNN", "decision_tree": "Decision Tree", "svm_rbf": "RBF SVM", "cnn": "1D CNN"}
    class_rows = []
    cm_rows = []
    cm_arrays = {}
    roc_assets = []
    for key, label in cmaps.items():
        folder = cnn_dir if key == "cnn" else class_dir
        name = "cnn" if key == "cnn" else key
        met = read_json(folder / f"{name}_test_metrics.json")
        class_rows.append({"Classifier": label, **{k: met[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}})
        cm = pd.read_csv(folder / f"{name}_confusion_matrix.csv", index_col=0)
        cm_arrays[label] = cm.to_numpy()
        cm_rows.append({"Classifier": label, "TN": int(cm.iloc[0, 0]), "FP": int(cm.iloc[0, 1]), "FN": int(cm.iloc[1, 0]), "TP": int(cm.iloc[1, 1])})
        rel = Path("..") / "outputs" / folder.name / f"{name}_roc_curve.png"
        roc_assets.append(f"![{label} ROC curve]({rel.as_posix()})")
    cdf = pd.DataFrame(class_rows).sort_values("roc_auc", ascending=False)
    best_class = cdf.iloc[0]
    regmeta = read_json(reg_dir / "metrics.json")
    regrows = [{"Model": name, **{k: v for k, v in value.items() if k in ("mse", "rmse", "pearson_r", "r2")}}
               for name, value in regmeta["models"].items()]
    kmmeta = read_json(km_dir / "metadata.json")
    pcameta = read_json(pca_dir / "metadata.json")
    km = pd.read_csv(km_dir / "wcss.csv")
    complexity = pd.read_csv(reg_dir / "polynomial_complexity.csv")
    rid = pd.read_csv(reg_dir / "ridge_cv.csv")
    las = pd.read_csv(reg_dir / "lasso_cv.csv")
    eigenvectors = pd.read_csv(pca_dir / "eigenvectors.csv", header=None)
    eigenvectors.index = pcameta["features"]
    eigenvectors.columns = [f"PC{i}" for i in range(1, eigenvectors.shape[1]+1)]
    eigenvectors = eigenvectors.rename_axis("Feature").reset_index()

    report_fig_dir = ROOT / "outputs" / "assignment_report"
    report_fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.5))
    for ax, (label, matrix) in zip(axes.flat, cm_arrays.items()):
        ax.imshow(matrix, cmap="Blues")
        ax.set(title=label, xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="Actual",
               xticklabels=["No failure", "Failure"], yticklabels=["No failure", "Failure"])
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center", color="black")
    fig.tight_layout()
    confusion_plot = report_fig_dir / "classification_confusion_matrices.png"
    fig.savefig(confusion_plot, dpi=200)
    plt.close(fig)

    body = f"""# Machine Learning Assignment Report

**Submission report · generated from experiment outputs · reproducibility seed: 42**

## Assignment overview

This report evaluates classification, regression, clustering, and dimensionality reduction. Experiments use fixed public datasets, and numerical results below are read from saved run outputs. The classification and regression parts use an 80/20 train/test split. Clustering and PCA are unsupervised full-dataset demonstrations and do not use a test split.

## 1. Classification

### Dataset and setup

The classification experiment uses the UCI AI4I 2020 Predictive Maintenance dataset ([UCI dataset page](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)), 10,000 observations. The binary target is `Machine failure`; inputs are product type and five sensor measurements. `UDI` and `Product ID` are identifiers; failure-mode flags are excluded to avoid target leakage. A fixed seed 42 stratified 80/20 split is shared by KNN, decision tree, RBF-SVM, and CNN. The three classical models use stratified five-fold training-only CV, selecting positive-class F1. Preprocessing is inside their pipelines. The CNN takes the eight preprocessed tabular inputs in a fixed `(8, 1)` sequence and uses Conv1D layers; training uses only the training split with a validation subset for checkpointing. This sequence interpretation is a limitation because adjacent columns are not naturally spatial.

### Held-out metrics

Precision, recall, and F1 treat failure as the positive class. ROC-AUC uses probabilities or continuous decision scores. The best ROC-AUC is **{best_class['Classifier']} ({best_class['roc_auc']:.4f})**.

{md_table(cdf)}

Confusion counts (rows actual, columns predicted):

{md_table(pd.DataFrame(cm_rows))}

![Classifier confusion matrices]({(Path('..') / 'outputs/assignment_report/classification_confusion_matrices.png').as_posix()})

ROC curves:

{'  '.join(roc_assets)}

ROC-AUC assesses ranking across thresholds, while F1 and confusion counts use a particular threshold. The data are imbalanced, so accuracy alone understates the challenge. Model complexity differs: KNN can overfit at very small neighbor counts, trees can overfit as leaves/depth increase, RBF-SVM controls boundary flexibility through C and gamma, and CNN capacity/epoch selection affects generalization. Consult saved CV tables and CNN learning history for train-validation gaps.

## 2. Regression

### Dataset and setup

The required regression experiment uses the bundled scikit-learn diabetes dataset (442 records, ten numeric baseline measurements, continuous one-year disease progression target; [scikit-learn dataset documentation](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)). The fixed seed 42 split has {regmeta['n_train']} training and {regmeta['n_test']} test records. Gradient-descent linear regression explicitly minimizes `J(w,b)=mean((Xw+b-y)^2)/2`; its batch gradients are `X.T @ residual / n` and `mean(residual)`, with standardized inputs, learning rate 0.1, maximum 20,000 iterations, and a loss-change tolerance/patience stopping rule. Polynomial regression applies degree-two feature expansion and linear regression. Ridge and Lasso alpha values are selected by five-fold CV on training data only, inside standardized pipelines.

### Held-out results

Pearson **R** is shown separately from coefficient of determination **R²**.

{md_table(pd.DataFrame(regrows))}

Polynomial degree comparison, using training fit and training-only five-fold validation. Degree zero is a mean-only baseline and demonstrates underfitting through high training and validation error; degree five nearly interpolates training rows but has much larger validation error, showing overfitting:

{md_table(complexity)}

Gradient descent stopped after {regmeta['gradient_descent']['iterations']} iterations. The convergence check was `{regmeta['gradient_descent']['converged']}`; loss changed from {regmeta['gradient_descent']['initial_loss']:.6f} to {regmeta['gradient_descent']['final_loss']:.6f}.

Regularization CV results (lowest validation MSE selected):

Ridge: selected alpha **{regmeta['models']['Ridge']['selected_alpha']}**.

{md_table(rid)}

Lasso: selected alpha **{regmeta['models']['Lasso']['selected_alpha']}**.

{md_table(las)}

![Polynomial model complexity]({(Path('..') / 'outputs/assignment_regression/complexity_curve.png').as_posix()})

Increasing polynomial degree generally lowers training error while validation error reveals the point where added flexibility stops generalizing. Ridge shrinks coefficients smoothly; Lasso can shrink coefficients to zero. Alpha is chosen without using the test set. Actual and predicted values are plotted in `outputs/assignment_regression/actual_vs_predicted.png`.

## 3A. K-Means from scratch

The required implementation is NumPy Lloyd K-Means with seeded k-means++ initialization, squared Euclidean assignment, centroid means, a shift tolerance, an iteration cap, and empty-cluster repair. It uses all four standardized numeric features of the bundled Iris dataset (150 records; [scikit-learn Iris documentation](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset)). Labels are not used for fitting. WCSS was evaluated for K=2, 3, 4, 5, and 6.

{md_table(km)}

![Elbow plot]({(Path('..') / 'outputs/assignment_clustering/elbow_curve.png').as_posix()})

WCSS cannot increase as K increases because extra centroids can only improve or preserve the closest-center fit. An elbow is subjective and can be weak or dataset-dependent; it is not proof of a uniquely correct K.

## 3B. PCA from scratch

PCA uses the ten numeric features of the bundled diabetes dataset. Features are standardized, centered, the sample covariance matrix is computed, then `numpy.linalg.eigh` yields eigenpairs sorted in descending order. Explained variance is eigenvalue divided by total eigenvalue; cumulative variance is its cumulative sum. Eigenvectors are checked for orthonormality, the covariance eigen-equation is checked numerically, and ratios sum to one.

Eigenvalues, ratios, and cumulative ratios:

{md_table(pd.read_csv(pca_dir / 'pca_statistics.csv'))}

Eigenvectors (columns are principal components):

{md_table(eigenvectors)}

The saved two-component output has shape {pcameta['dimensions']['two_components']}; the three-component output has shape {pcameta['dimensions']['three_components']}.

![Cumulative explained variance]({(Path('..') / 'outputs/assignment_pca/cumulative_explained_variance.png').as_posix()})

![First two principal components]({(Path('..') / 'outputs/assignment_pca/pca_2d_visualization.png').as_posix()})

## Discussion and limitations

The experiments use one fixed split for supervised metrics. CV on training data guides hyperparameters; the held-out test is reserved for final comparison. Results may vary with a new sample or split. The CNN imposes an arbitrary sequence structure on tabular columns. Iris clustering uses standardized Euclidean distance and K-Means favors compact, roughly spherical clusters. PCA is linear and retains directions of variance rather than task-specific predictive information. Pearson R measures linear association, whereas R² measures explained variance relative to a mean baseline.

## Reproduction

From the repository root, install `requirements.txt`, then run:

```powershell
python -m classification.train
python -m cnn.train
python -m cnn.evaluate
python -m regression.assignment
python -m clustering.scratch_kmeans
python -m pca.scratch_pca
python -m reports.build_assignment_report
python -m unittest discover -v
```

### Compliance checklist

| Requirement | Implementation | Generated evidence |
|---|---|---|
| KNN CV, tree, RBF-SVM, deep CNN; 80/20, seed, stratification; metrics, confusion matrices, ROC | `classification/train.py`, `cnn/` | `outputs/classification/`, `outputs/cnn/` |
| GD linear, polynomial, Ridge/Lasso; training-only CV, Pearson R, complexity curve | `regression/assignment.py` | `outputs/assignment_regression/` |
| NumPy K-Means, K=2–6, WCSS and elbow | `clustering/scratch_kmeans.py` | `outputs/assignment_clustering/` |
| Covariance/eigen PCA, two and three components, variance and visualization | `pca/scratch_pca.py` | `outputs/assignment_pca/` |
| Automated coverage for required from-scratch algorithms | `test_assignment_algorithms.py` | `python -m unittest discover -v` |
| Reproducible combined report | `reports/build_assignment_report.py` | `reports/ML_Assignment_Report.md` |

"""
    OUT.write_text(body, encoding="utf-8")
    pdf_path = ROOT / "reports" / "ML_Assignment_Report.pdf"
    rendered_lines = []
    for line in body.splitlines():
        if line.startswith("![") or line.startswith("```"):
            continue
        rendered_lines.extend(textwrap.wrap(line, width=112, replace_whitespace=False, drop_whitespace=False) or [""])
    with PdfPages(pdf_path) as pdf:
        for page_index in range(0, len(rendered_lines), 66):
            fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
            fig.text(.07, .96, "Machine Learning Assignment Report", fontsize=9, color="#245b78", va="top")
            for row, line in enumerate(rendered_lines[page_index:page_index + 66]):
                fig.text(.07, .93 - row * .0132, line, fontsize=6.7, family="monospace", va="top", color="#222222")
            fig.text(.93, .025, str(page_index // 66 + 1), ha="right", fontsize=8, color="#666666")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        figures = [confusion_plot, class_dir / "knn_roc_curve.png", class_dir / "decision_tree_roc_curve.png",
                   class_dir / "svm_rbf_roc_curve.png", cnn_dir / "cnn_roc_curve.png",
                   reg_dir / "complexity_curve.png", reg_dir / "actual_vs_predicted.png",
                   km_dir / "elbow_curve.png", pca_dir / "cumulative_explained_variance.png",
                   pca_dir / "pca_2d_visualization.png"]
        for image_path in figures:
            image = plt.imread(image_path)
            fig, ax = plt.subplots(figsize=(8.27, 11.69), facecolor="white")
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(image_path.name, fontsize=14, pad=14)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"Wrote {OUT} and {pdf_path}")


if __name__ == "__main__":
    build()
