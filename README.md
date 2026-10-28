# Machine Learning Assignment

Reproducible classification, regression, clustering, and PCA experiments. The required assignment algorithms and one combined report are included. Random seed: `42`.

## Datasets

- **Classification:** UCI AI4I 2020 Predictive Maintenance (`data/ai4i2020.csv`), binary machine failure. IDs and failure-mode leakage columns are excluded. The same stratified 80/20 split is used for KNN, Decision Tree, RBF-SVM, and a 1D CNN.
- **Regression assignment:** scikit-learn diabetes dataset (bundled; no download), 10 numeric features and continuous disease progression target, 80/20 split.
- **K-Means:** scikit-learn Iris dataset (bundled), four standardized measurements, K=2 through 6, NumPy implementation.
- **PCA:** bundled diabetes dataset, ten standardized numeric features, covariance/eigendecomposition implementation.

The repository also retains the earlier Concrete regression and Palmer Penguins clustering extensions in `regression/train.py` and `clustering/cluster.py`.

## Setup and run

Python 3.10+ is recommended. Install the existing project dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run required experiments from the repository root:

```powershell
python -m classification.train
python -m cnn.train
python -m cnn.evaluate
python -m regression.assignment
python -m clustering.scratch_kmeans
python -m pca.scratch_pca
python -m reports.build_assignment_report
```

The classification and CNN workflows use `data/ai4i2020.csv`; the classifier loader downloads it from UCI if absent. Assignment regression, K-Means, and PCA use datasets bundled with scikit-learn.

## Outputs and report

- Comprehensive report: [`reports/ML_Assignment_Report.pdf`](reports/ML_Assignment_Report.pdf) and [`reports/ML_Assignment_Report.md`](reports/ML_Assignment_Report.md)
- Classification scores, confusion matrices, ROC curves, CV results: `outputs/classification/` and `outputs/cnn/`
- Required regression scores and plots: `outputs/assignment_regression/`
- From-scratch K-Means WCSS table and elbow plot: `outputs/assignment_clustering/`
- From-scratch PCA scores, eigenvectors, variance statistics, and plots: `outputs/assignment_pca/`

Generated datasets, outputs, and fitted artifacts are ignored by Git. Re-run the experiment commands to regenerate them before building the report.

## Tests

```powershell
python -m unittest discover -v
```

Focused required-algorithm tests can be run with `python -m unittest -v test_assignment_algorithms`.

## Limitations

The CNN represents ordered tabular columns as a 1D sequence, so neighboring inputs do not have an inherent spatial relationship. A single fixed supervised split does not measure variation across splits. K-Means and PCA are unsupervised demonstrations and use all rows in their selected datasets.
