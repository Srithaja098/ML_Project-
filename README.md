# AI4I 2020 Predictive Maintenance Classification

This project predicts the `Machine failure` label in the AI4I 2020 Predictive Maintenance dataset using KNN,
Decision Tree, RBF-SVM, and a 1D CNN. It includes 5-fold cross-validated classical model selection, CNN validation
checkpointing, and held-out test evaluation for all four models.

> **CNN interpretation needs instructor confirmation.** The CNN treats the ordered tabular feature vector as a
> one-dimensional sequence. It is not an image model, and instructor acceptance of this approach has not been confirmed.

## Dataset and features

The project uses the [UCI AI4I 2020 Predictive Maintenance dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset),
with 10,000 rows. The binary target is `Machine failure` (1 = failure, positive class; 0 = no failure). The failure
class is rare, at roughly 3.4% of rows.

Inputs are `Type` and five numeric sensor measurements: air temperature, process temperature, rotational speed,
torque, and tool wear. `UDI` and `Product ID` are excluded as identifiers. `TWF`, `HDF`, `PWF`, `OSF`, and `RNF`
are excluded because they are failure-mode labels that leak target information.

KNN and RBF-SVM use one-hot encoding for `Type` and standard scaling for numeric features. The Decision Tree uses
ordinal encoding (`L < M < H`) and unscaled numeric values. Preprocessing is part of each classical pipeline, so it
is fit within each CV training fold. The CNN uses one-hot `Type` plus standardized numeric values; its preprocessor
is fit on the CNN fit subset only. The resulting eight features are passed as a `(8, 1)` sequence to Conv1D layers.

## Split and model selection

All models share the same stratified 80/20 outer split, with seed 42: 8,000 training rows and 2,000 held-out test
rows. The ordered test row indices in all four model metadata files match exactly. The test set is reserved for
final evaluation and is not used for tuning or checkpoint selection.

The classical models use stratified 5-fold cross-validation on the 8,000 training rows. `GridSearchCV` selects the
candidate with the highest failure-class F1 and refits that pipeline on all training rows. The configurations and
search grids are in `classification/train.py` and are recorded with the saved model metadata.

| Model | Selected hyperparameters | Best mean CV F1 (± std) |
|---|---|---:|
| KNN | `n_neighbors=1`, `weights=uniform` | 0.4928 ± 0.0731 |
| Decision Tree | `max_depth=10`, `min_samples_leaf=5`, `class_weight=None` | 0.7245 ± 0.0528 |
| RBF-SVM | `C=100`, `gamma=scale`, `class_weight=None` | 0.6912 ± 0.0182 |

The CNN holds out a stratified 10% of the training portion for validation: 7,200 fit rows and 800 validation rows.
Balanced class weights are calculated from the 7,200 fit labels. Early stopping, learning-rate reduction, and
checkpoint selection monitor validation PR-AUC. The saved checkpoint is epoch 12 of 22, with validation PR-AUC
0.7742 and ROC-AUC 0.9830. CNN test predictions use the fixed 0.5 threshold recorded in the test metrics file;
the threshold was not tuned on the test set. Test metrics are unweighted. ROC-AUC, average precision, and ROC
curves use continuous ranking scores (sigmoid probabilities for CNN; decision scores for SVM).

## Held-out test results

Precision, recall, and F1 refer to the failure class (1). Metrics are unweighted and rounded to four decimal places.
Average precision is the positive-class area under the precision-recall curve.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| KNN | 0.9600 | 0.4032 | 0.3676 | 0.3846 | 0.6742 | 0.1697 |
| Decision Tree | 0.9860 | 0.8226 | 0.7500 | **0.7846** | 0.8910 | 0.7453 |
| RBF-SVM | 0.9800 | 0.7593 | 0.6029 | 0.6721 | **0.9767** | **0.7862** |
| 1D CNN | 0.9340 | 0.3261 | **0.8824** | 0.4762 | 0.9714 | 0.7242 |

**Best ROC-AUC:** RBF-SVM (0.9767). **Best failure-class F1:** Decision Tree (0.7846). The CNN has the highest
failure recall (0.8824) at its fixed 0.5 threshold, alongside the lowest precision (0.3261) among the four.

### Confusion matrices

Rows are actual labels; columns are predicted labels. TP/FP/TN/FN treat failure class 1 as positive.

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| KNN | 1,895 | 37 | 43 | 25 |
| Decision Tree | 1,921 | 11 | 17 | 51 |
| RBF-SVM | 1,919 | 13 | 27 | 41 |
| 1D CNN | 1,808 | 124 | 8 | 60 |

Saved confusion matrices:

- `outputs/classification/knn_confusion_matrix.csv`
- `outputs/classification/decision_tree_confusion_matrix.csv`
- `outputs/classification/svm_rbf_confusion_matrix.csv`
- `outputs/cnn/cnn_confusion_matrix.csv`

### ROC curves

- `outputs/classification/knn_roc_curve.png`
- `outputs/classification/decision_tree_roc_curve.png`
- `outputs/classification/svm_rbf_roc_curve.png`
- `outputs/cnn/cnn_roc_curve.png`

The displayed ROC-AUC values match the saved test metric JSON files to plot rounding. The SVM curve uses
`decision_function()`; SVC probability estimation is disabled. KNN, Decision Tree, and CNN use their continuous
positive-class probability scores.

## Overfitting observations and limitations

The KNN best CV candidate (`n_neighbors=1`, uniform weights) has mean training-fold F1 1.0000 and mean CV F1
0.4928, a large 0.5072 gap. Its held-out F1 is 0.3846, which supports a clear overfitting concern for this setting.

The Decision Tree best candidate has mean training-fold F1 0.8269 versus mean CV F1 0.7245, a 0.1024 gap. Its
held-out F1 is 0.7846, above the CV mean, so the test result does not show a further generalization drop. The RBF-SVM
gap is larger (0.8721 training-fold F1 versus 0.6912 CV F1), but held-out F1 (0.6721) is close to its CV mean; this
is a training-to-validation gap without a large additional held-out decline.

For the CNN, validation PR-AUC peaked at 0.7742 at epoch 12. By epoch 22, training PR-AUC had risen to 0.7681 while
validation PR-AUC fell to 0.6806; validation ROC-AUC also fell from 0.9830 to 0.9759. This is evidence of late-stage
overfitting, and the saved best-epoch checkpoint avoids selecting the later epoch. CNN training loss is class
weighted while validation/test metrics are unweighted; these loss values should not be compared directly.

The KNN and Decision Tree validation curves were regenerated from their existing CV result CSVs with numeric
x-axis ordering; the tree's `max_depth=None` candidate is labeled `None`. No model was retrained and the saved CV
scores and test results were not changed.

The CNN operates on tabular measurements arranged in an arbitrary feature order. Convolution therefore assumes
local relationships between adjacent columns that may not be meaningful. Instructor acceptance of this model
interpretation remains unconfirmed. The failure class is also imbalanced, so accuracy alone is not an adequate
comparison; precision, recall, F1, average precision, ROC-AUC, and confusion counts provide additional context.

## Reproducing the experiment

Run commands from the project root. Training commands rerun tuning/training and overwrite the corresponding saved
experiment outputs; they are included for reproducibility and were not run to prepare this README.

```powershell
# Classical models: 5-fold CV, fit best estimator, save test outputs
python -m classification.train

# CNN: train using fit/validation subsets of the shared outer training split
python -m cnn.train

# Evaluate the saved best CNN checkpoint on the shared held-out rows
python -m cnn.evaluate

# Existing self-checks and evaluation output tests
python -m test_member1
python -m unittest -v test_classification_evaluation test_cnn_evaluation test_validation_curve_plotting

# Build the report from saved results, then package source, data, artifacts, and outputs
python -m reports.build_report
python -m reports.build_submission_zip
```

The dataset is read from `data/ai4i2020.csv`. If absent, the data loader downloads it from UCI. The evaluation
command verifies that the CNN test row indices match all three classical metadata files before producing metrics.

## Saved artifacts and outputs

- `artifacts/classification/<model>.joblib` and `artifacts/cnn/ai4i_cnn.keras`: fitted model files.
- `artifacts/classification/<model>_metadata.json` and `artifacts/cnn/ai4i_cnn_metadata.json`: feature exclusions,
  seeds, split row indices, model settings, and CV or validation results.
- `outputs/classification/<model>_cv_results.csv`: all classical grid candidates and train/CV scores.
- `outputs/classification/<model>_test_metrics.json` and `outputs/cnn/cnn_test_metrics.json`: saved test metrics.
- `outputs/classification/<model>_validation_curve.png`: classical training-fold and CV F1 plots.
- `outputs/cnn/history.csv` and `outputs/cnn/learning_curves.png`: epoch history and CNN learning curves.
- Confusion matrix CSVs and ROC curve PNGs are listed above.
- `reports/AI4I_2020_Classification_Report.pdf`: final report generated from saved results without retraining.
- `submission/AI4I_2020_Submission.zip`: curated source/data/model/results archive; excludes `.git`, `.venv`, and caches.
- `reports/build_report.py` and `reports/build_submission_zip.py`: scripts to recreate those two deliverables.

---

# Clustering: Palmer Penguins Dataset

The `clustering` module implements unsupervised analysis on the [UCI Palmer Penguins dataset](https://archive.ics.uci.edu/dataset/690/palmer+penguins-3) across 342 complete observations and 4 standardized morphological measurements (`bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`).

Four clustering paradigms are compared:
1. **K-Means Clustering:** Evaluated for $k \in [2, 10]$ via Elbow Inertia and Silhouette score curves ($k=3$ selected).
2. **Hierarchical (Agglomerative) Clustering:** Evaluated linkage strategies; Ward linkage tree cut at $k=3$ with full dendrogram.
3. **DBSCAN:** Density-based clustering with $k$-distance knee analysis ($\varepsilon=0.6, \text{min\_samples}=7$).
4. **Gaussian Mixture Model (GMM):** Soft/hard probabilistic clustering with BIC/AIC component selection ($k=3$, full covariance).

### Clustering Summary Metrics

| Model | Hyperparameters | Clusters | Noise | Silhouette | Calinski-Harabasz | Davies-Bouldin | ARI (Species) | NMI (Species) | V-Measure |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **GMM** | $k=3$, full cov | 3 | 0 | **0.4545** | 425.96 | 0.8956 | **0.9603** | **0.9349** | **0.9349** |
| **Hierarchical (Ward)** | $k=3$, euclidean | 3 | 0 | 0.4541 | 420.72 | **0.8465** | **0.9159** | **0.9007** | **0.9007** |
| **K-Means** | $k=3$, k-means++ | 3 | 0 | 0.4472 | 441.68 | 0.9436 | 0.7928 | 0.7855 | 0.7855 |
| **DBSCAN** | $\varepsilon=0.6, \text{min}=7$ | 3 | 42 | 0.4933* | 458.80* | 0.7673* | 0.7662 | 0.7279 | 0.7279 |

*\*Internal metrics on non-noise points.*

### Running Clustering & Tests

```powershell
# Run end-to-end clustering pipeline
python -m clustering.cluster

# Run clustering unit tests
python -m unittest -v test_clustering
```

Saved figures are in `outputs/clustering/` (`kmeans_elbow_silhouette.png`, `hierarchical_dendrogram.png`, `dbscan_k_distance.png`, `gmm_bic_aic.png`, `pca_clustering_comparison.png`, `feature_pairwise_clusters.png`), and models are in `artifacts/clustering/`. Detailed documentation is in [`clustering/README.md`](clustering/README.md).

## Submission checklist and open questions

- The repository contains this project documentation, saved metrics, figures, model metadata, a draft outline in
  `REPORT_OUTLINE.md`, the final report PDF, and the submission ZIP.
- No assignment brief or rubric was present in the repository when this documentation was prepared. Confirm required
  report sections, format, citation style, and whether source code, figures, PDF, or ZIP are required.
- Confirm with the instructor whether a 1D CNN over tabular features is an acceptable interpretation of the task.
- If selecting a different CNN decision threshold is required, choose it using validation data only and report that
  policy separately; do not tune it using held-out test results.

