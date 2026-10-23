"""Evaluate the saved best AI4I CNN checkpoint on the shared held-out test split.

Run from the repository root: python -m cnn.evaluate
"""
import json

import joblib
import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from classification.data import load_dataset, split_features_target, stratified_split
from cnn.data_loader import to_cnn_input
from config import ARTIFACTS_DIR, OUTPUTS_DIR

MODEL_DIR = ARTIFACTS_DIR / "cnn"
MODEL_PATH = MODEL_DIR / "ai4i_cnn.keras"
PREPROCESSOR_PATH = MODEL_DIR / "ai4i_cnn_preprocessor.joblib"
METADATA_PATH = MODEL_DIR / "ai4i_cnn_metadata.json"
CLASSICAL_MODELS = ("knn", "decision_tree", "svm_rbf")
CLASSICAL_METADATA_DIR = ARTIFACTS_DIR / "classification"
OUT_DIR = OUTPUTS_DIR / "cnn"
THRESHOLD = 0.5


def verify_test_indices(test_indices, cnn_metadata, classical_metadata_dir=CLASSICAL_METADATA_DIR):
    """Require the reconstructed test row indices to match CNN and every classical model exactly."""
    expected = list(test_indices)
    metadata_paths = {"cnn": cnn_metadata}
    metadata_paths.update({
        name: classical_metadata_dir / f"{name}_metadata.json" for name in CLASSICAL_MODELS
    })
    for model_name, path in metadata_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing {model_name} metadata required for split verification: {path}")
        saved_rows = json.loads(path.read_text()).get("rows", {}).get("test")
        if saved_rows is None:
            raise ValueError(f"{model_name} metadata does not contain rows.test: {path}")
        if list(saved_rows) != expected:
            raise ValueError(f"test row indices do not match {model_name} metadata ({path})")
    return len(expected)


def save_test_evaluation(y_true, probabilities, out_dir=OUT_DIR, threshold=THRESHOLD):
    """Save unweighted test metrics, thresholded confusion matrix, and sigmoid-score ROC curve."""
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    probabilities = np.asarray(probabilities).reshape(-1).astype(float)
    if len(y_true) != len(probabilities):
        raise ValueError("test labels and probability scores must have equal lengths")
    if not 0 <= threshold <= 1:
        raise ValueError("prediction threshold must be in [0, 1]")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("CNN sigmoid scores must be finite probabilities in [0, 1]")
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("test labels must contain both binary classes 0 and 1")

    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "model": "ai4i_cnn",
        "n_test": int(len(y_true)),
        "positive_class": 1,
        "score_type": "sigmoid_probability",
        "prediction_threshold": float(threshold),
        "threshold_policy": "fixed at 0.5; not selected using the test set",
        "metric_weighting": "unweighted; positive-class metrics use machine failure (class 1)",
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities, pos_label=1)),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cnn_test_metrics.json").write_text(json.dumps(metrics, indent=2))
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    pd.DataFrame(matrix, index=["actual_0", "actual_1"], columns=["predicted_0", "predicted_1"]).to_csv(
        out_dir / "cnn_confusion_matrix.csv", index=True, index_label="actual")

    fpr, tpr, _ = roc_curve(y_true, probabilities, pos_label=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC AUC = {metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="AI4I CNN: held-out ROC curve",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cnn_roc_curve.png", dpi=120)
    plt.close(fig)
    return metrics


def main():
    for path in (MODEL_PATH, PREPROCESSOR_PATH, METADATA_PATH):
        if not path.exists():
            raise FileNotFoundError(f"required trained CNN artifact not found: {path}")
    metadata = json.loads(METADATA_PATH.read_text())
    df = load_dataset()
    X, y = split_features_target(df)
    _, X_test, _, y_test = stratified_split(X, y)
    n_test = verify_test_indices(X_test.index.tolist(), METADATA_PATH)

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    if preprocessor.get_feature_names_out().tolist() != metadata["input_features"]:
        raise ValueError("saved preprocessor features do not match CNN training metadata")
    X_test_cnn = to_cnn_input(preprocessor, X_test)
    if list(X_test_cnn.shape[1:]) != metadata["input_shape"]:
        raise ValueError("transformed test shape does not match CNN training metadata")

    model = keras.models.load_model(MODEL_PATH, compile=False)
    probabilities = model.predict(X_test_cnn, verbose=0).reshape(-1)
    metrics = save_test_evaluation(y_test.to_numpy(), probabilities)
    print(f"Verified {n_test} held-out test rows against CNN, KNN, Decision Tree, and RBF-SVM metadata.")
    print("Test metrics (unweighted): " + "  ".join(
        f"{name}={metrics[name]:.4f}" for name in
        ("accuracy", "precision", "recall", "f1", "roc_auc", "average_precision")))
    print(f"Metrics, confusion matrix, and ROC curve saved under {OUT_DIR}")


if __name__ == "__main__":
    main()
