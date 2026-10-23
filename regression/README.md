# Regression: Concrete Compressive Strength Dataset

This module implements predictive regression models for the **Concrete Compressive Strength dataset** ([UCI Machine Learning Repository #165](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)) by Prof. I-Cheng Yeh (1998).

The objective is to predict the continuous compressive strength of concrete (in MPa) given its raw mixture proportions and curing age.

Six diverse regression models are trained and systematically evaluated using **5-fold cross-validation** on an 80% training set (824 rows) followed by unbiased evaluation on a shared 20% held-out test set (206 rows, seed=42):
1. **Ridge Regression** (Linear baseline with $L_2$ regularization)
2. **K-Nearest Neighbors Regressor (KNN)** (Distance-weighted non-parametric regression)
3. **Decision Tree Regressor** (Interpretable non-linear tree)
4. **Random Forest Regressor** (Bagging ensemble of decision trees)
5. **Gradient Boosting Regressor** (Sequential gradient boosted decision trees)
6. **Support Vector Regression (SVR RBF)** (Kernel margin regression)

---

## Dataset & Feature Engineering

- **Source:** [UCI Machine Learning Repository #165](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)
- **Instances:** 1,030 experimental concrete formulations.
- **Split:** 80:20 outer split (824 training rows, 206 held-out test rows, seed=42).
- **Target Variable:** `Concrete compressive strength` (continuous numerical target in MPa, range: 2.33 – 82.60 MPa, mean: 35.82 MPa).
- **Input Features (8 continuous numeric predictors):**
  1. `Cement`: Amount of cement ($\text{kg/m}^3$)
  2. `Blast Furnace Slag`: Amount of blast furnace slag ($\text{kg/m}^3$)
  3. `Fly Ash`: Amount of fly ash ($\text{kg/m}^3$)
  4. `Water`: Amount of mixing water ($\text{kg/m}^3$)
  5. `Superplasticizer`: Amount of superplasticizing admixture ($\text{kg/m}^3$)
  6. `Coarse Aggregate`: Amount of coarse aggregate ($\text{kg/m}^3$)
  7. `Fine Aggregate`: Amount of fine aggregate ($\text{kg/m}^3$)
  8. `Age`: Curing age in days (1 to 365 days)
- **Preprocessing:** Standard scaling (`StandardScaler`) is fit strictly within each CV training fold for distance/kernel models (Ridge, KNN, SVR), avoiding data leakage.

---

## Held-Out Test Evaluation Results

All models were tuned via 5-fold cross-validation on the 824 training instances (optimizing negative RMSE) and evaluated on the 206 held-out test instances:

| Model | Selected Hyperparameters | 5-Fold CV RMSE (± std) | 5-Fold CV $R^2$ (± std) | Test $R^2$ (↑) | Test RMSE (↓) | Test MAE (↓) | Test MedAE (↓) | Test MAPE (↓) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Gradient Boosting** | `learning_rate=0.2`, `max_depth=3`, `n_estimators=300` | **4.782 (±0.448)** | **0.918 (±0.018)** | **0.9336** | **4.1377** | **2.9048** | **1.8889** | **9.40%** |
| **Random Forest** | `max_depth=None`, `min_samples_leaf=1`, `n_estimators=200` | 5.304 (±0.429) | 0.899 (±0.017) | **0.8813** | 5.5304 | 3.7875 | 2.3663 | 12.43% |
| **Support Vector Regression (SVR RBF)** | $C=200.0$, $\varepsilon=0.2$, $\gamma=0.1$ | 6.222 (±0.714) | 0.861 (±0.031) | **0.8735** | 5.7103 | 3.9584 | 2.3681 | 12.53% |
| **Decision Tree** | `max_depth=15`, `min_samples_leaf=2` | 6.636 (±0.752) | 0.842 (±0.033) | 0.8028 | 7.1279 | 4.8254 | 3.2625 | 14.98% |
| **K-Nearest Neighbors (KNN)** | $k=7$, `weights='distance'` | 8.378 (±0.517) | 0.750 (±0.030) | 0.7761 | 7.5960 | 5.6931 | 4.1819 | 19.95% |
| **Ridge Regression** | $\alpha=0.1$ | 10.627 (±0.477) | 0.597 (±0.047) | 0.6276 | 9.7964 | 7.7462 | 5.8587 | 29.28% |

---

## Physical & Engineering Insights

1. **Non-Linear Dominance:**
   - Linear Ridge regression achieves only $R^2 = 0.6276$, whereas non-linear models (Gradient Boosting, Random Forest, SVR) achieve $R^2 = 0.87 - 0.93$. Concrete compressive strength is governed by complex chemical hydration kinetics that are inherently non-linear.
2. **Top Predictive Drivers (Feature Importance):**
   - **`Age` (Curing duration):** Dominates strength development due to the continuous formation of calcium silicate hydrate (C-S-H) gel over time.
   - **`Cement` content:** Primary binding agent directly correlating with ultimate strength capacity.
   - **`Water` content:** Inversely relates to strength according to Abram's Law (higher water-cement ratio creates porous capillary voids that degrade strength).
3. **Best Model Performance:**
   - **Gradient Boosting Regressor** achieves the top performance on held-out test data with **$R^2 = 0.9336$**, an **RMSE of 4.14 MPa**, and a **Mean Absolute Percentage Error of only 9.40%**.

---

## Generated Diagnostic Plots and Artifacts

Saved in `outputs/regression/`:
- `actual_vs_predicted.png`: 6-panel comparison of actual vs. predicted compressive strength with identity line ($y=x$).
- `residuals_plot.png`: Residuals vs. predicted values for all models, verifying homoscedasticity and absence of systematic bias.
- `feature_importance.png`: Feature importance bar charts for Gradient Boosting and Random Forest.
- `cv_model_comparison.png`: 5-fold cross-validation RMSE and $R^2$ bar charts with standard deviation error bars.
- `regression_summary_metrics.csv`: Comparative metrics summary table.
- `<model>_cv_results.csv`: Complete hyperparameter candidate scores across all CV folds.
- `<model>_test_metrics.json`: Detailed held-out test evaluation JSON.

Saved in `artifacts/regression/`:
- Fitted model pipelines: `ridge.joblib`, `knn.joblib`, `decision_tree.joblib`, `random_forest.joblib`, `gradient_boosting.joblib`, `svm_rbf.joblib`, `scaler.joblib`.
- Full experiment metadata: `metadata.json`.

---

## How to Run

From the project root:

```powershell
# Run the complete regression training and evaluation pipeline
python -m regression.train

# Run regression unit tests
python -m unittest -v test_regression
```
