# Draft Report Outline: AI4I 2020 Predictive Maintenance

**Status:** Planning outline only. This is not the final report. No assignment rubric or submission format was
present in the repository, so the outline follows the project's documented scope and marks rubric-dependent items
for instructor confirmation.

## 1. Title page and abstract

- Project title, author/course details, date, and dataset attribution.
- Briefly state the task (binary machine-failure prediction), models compared, shared test design, and primary
  findings: Decision Tree has the highest failure-class F1 (0.7846), while RBF-SVM has the highest ROC-AUC (0.9767).
- Do not claim that the instructor approved the tabular 1D CNN interpretation.

## 2. Problem and objectives

- Explain predictive maintenance and the `Machine failure` target.
- State the comparison objective and the importance of measuring rare failure detection, rather than accuracy alone.
- Define the positive class as failure (label 1).

## 3. Dataset and preprocessing

- Describe AI4I 2020 (10,000 rows) and the six input fields: `Type` plus five sensor measurements.
- Explain exclusion of identifiers (`UDI`, `Product ID`) and target-leakage failure-mode columns (`TWF`, `HDF`,
  `PWF`, `OSF`, `RNF`).
- Document model-specific transformations: one-hot plus scaling for KNN/SVM; ordinal `Type` plus raw numeric values
  for Decision Tree; one-hot plus scaling for CNN.
- Note that the CNN treats the eight tabular features in a fixed order as a sequence; this interpretation remains
  subject to instructor acceptance.

**Suggested table:** feature, type, preprocessing, and inclusion/exclusion rationale.

## 4. Experimental design

- Describe the common stratified 80/20 split (seed 42), with 8,000 training and 2,000 held-out test rows.
- State that ordered test indices were verified identical across all four models.
- Describe classical stratified 5-fold CV, the searched grids, and selection by positive-class F1.
- Describe CNN's stratified 7,200/800 fit/validation split within training, fit-only class weights, best epoch 12,
  and validation PR-AUC checkpoint selection. The test set is not used for tuning or checkpoint selection.
- Document test decision thresholds (default/fixed 0.5 for thresholded predictions) and continuous scores used for
  ROC-AUC and average precision.

**Suggested table:** model, preprocessing, grid search dimensions, CV scheme, and model selection criterion.

## 5. Results

Use the verified held-out results below. Precision, recall, and F1 are for failure class 1; average precision is
the positive-class PR-AUC.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| KNN | 0.9600 | 0.4032 | 0.3676 | 0.3846 | 0.6742 | 0.1697 |
| Decision Tree | 0.9860 | 0.8226 | 0.7500 | 0.7846 | 0.8910 | 0.7453 |
| RBF-SVM | 0.9800 | 0.7593 | 0.6029 | 0.6721 | 0.9767 | 0.7862 |
| 1D CNN | 0.9340 | 0.3261 | 0.8824 | 0.4762 | 0.9714 | 0.7242 |

**Confusion-count table (TN, FP, FN, TP):** KNN (1895, 37, 43, 25); Decision Tree (1921, 11, 17, 51);
RBF-SVM (1919, 13, 27, 41); CNN (1808, 124, 8, 60).

**Suggested figures and tables:**

- Main test-metric comparison table above.
- Confusion-count table above, plus saved CSVs if the report requires visual matrices.
- ROC comparison figure made from the saved curves:
  - `outputs/classification/knn_roc_curve.png`
  - `outputs/classification/decision_tree_roc_curve.png`
  - `outputs/classification/svm_rbf_roc_curve.png`
  - `outputs/cnn/cnn_roc_curve.png`
- Classical CV summary table derived from `outputs/classification/knn_cv_results.csv`,
  `outputs/classification/decision_tree_cv_results.csv`, and `outputs/classification/svm_rbf_cv_results.csv`.
- The KNN and Decision Tree validation-curve PNGs have been regenerated from the unchanged CV CSVs. Numeric
  candidates are now ordered numerically and the Decision Tree null-depth candidate is labeled `None`; these figures
  can be included. The SVM curve's C-axis is also numerically ordered.
- CNN learning curves from `outputs/cnn/learning_curves.png`, supported by `outputs/cnn/history.csv`.

## 6. Overfitting and error analysis

- KNN: best-candidate mean train F1 1.0000 versus CV F1 0.4928 (gap 0.5072); test F1 is 0.3846. Discuss as the
  clearest overfitting/generalization concern.
- Decision Tree: train-fold F1 0.8269 versus CV F1 0.7245 (gap 0.1024), while test F1 is 0.7846. Report the CV gap
  without claiming a test-set decline.
- RBF-SVM: train-fold F1 0.8721 versus CV F1 0.6912 (gap 0.1809); test F1 is 0.6721, close to CV. Distinguish
  the training/CV gap from held-out performance.
- CNN: validation PR-AUC peaks at 0.7742 at epoch 12, then falls to 0.6806 by epoch 22 while training PR-AUC rises
  to 0.7681. Discuss late-stage overfitting and the saved best-epoch checkpoint. Do not compare class-weighted
  training loss directly to unweighted validation/test loss.
- Explain threshold behavior and false positives/false negatives using the confusion-count table.

## 7. Discussion and limitations

- Interpret the distinction between best F1 (Decision Tree) and best ROC-AUC (RBF-SVM); discuss the CNN's high
  recall and lower precision at threshold 0.5 in the context of rare failures.
- Explain that ROC-AUC and average precision measure ranking, whereas confusion matrices and threshold metrics
  depend on a chosen threshold.
- Discuss limitations: class imbalance, one dataset/split, arbitrary feature adjacency for the CNN, and the CNN's
  tabular interpretation needing instructor acceptance.
- Avoid claiming statistical significance or broad deployment readiness; no repeated-split uncertainty analysis is
  included in the saved artifacts.

## 8. Conclusion

- Summarize which model leads on F1 and ROC-AUC, and which has highest recall.
- State the practical choice depends on the relative cost of missed failures and false alarms.
- Record instructor confirmation of the CNN interpretation if obtained before submission.

## 9. Reproducibility and appendix

- Include commands from the README for classical training, CNN training, CNN evaluation, and tests. Warn that training
  commands rerun experiments and overwrite outputs; the saved results used in the report should remain unchanged.
- Link to metadata, CV results, metrics JSONs, confusion CSVs, and history for auditability.
- Add dataset citation and any other references required by the course.

## Requirements still uncertain

- Exact assignment rubric, mandatory report sections, page/word limit, and required citation format.
- Whether CNN-on-tabular input satisfies the instructor's CNN requirement.
- Whether the submission requires a PDF, source archive, code comments/screenshots, or specific figures.
- Whether an explicitly validation-selected CNN threshold is expected instead of the documented fixed 0.5 policy.
