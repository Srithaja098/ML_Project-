"""Train the 1D CNN on AI4I 2020 (target: Machine failure), selecting the best epoch on the validation split.

Uses the same 80% training rows as KNN / Decision Tree / RBF-SVM; the 20% held-out test rows are only
recorded by index in the metadata and never touched (reserved for the shared evaluation phase).

Usage (from the repo root):
    python -m cnn.train
    python -m cnn.train --epochs 50 --batch-size 32 --learning-rate 5e-4 --dropout 0.2 --seed 1
"""
import argparse
import json

try:
    import joblib
    import keras
except ImportError as e:
    raise SystemExit(f"Missing dependency ({e.name}). Install the requirements first: pip install -r requirements.txt")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from classification.data import CATEGORICAL, ID_COLUMNS, LEAKAGE_COLUMNS, NUMERIC, TARGET
from cnn.data_loader import VAL_SIZE, load_ai4i_splits
from cnn.model import build_cnn
from config import ARTIFACTS_DIR, OUTPUTS_DIR, SEED, TEST_SIZE

# Default hyperparameters; the CLI flags below override the common ones.
CONFIG = {
    "conv_filters": [32, 64],
    "kernel_size": 3,
    "dense_units": 64,
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 100,               # upper bound; early stopping usually ends sooner
    "early_stop_patience": 10,   # epochs without val PR-AUC improvement before stopping
    "lr_patience": 4,            # epochs without improvement before halving the learning rate
    "balanced_class_weight": True,  # failures are ~3%: weight them by n / (2 * n_class), from fit rows only
    "seed": SEED,                # weight init + validation rows; the train/test split always uses config.SEED
}
# Accuracy is ~97% for "never fails", so the best epoch is picked by validation PR-AUC of the failure class.
MONITOR = "val_pr_auc"
MODEL_DIR = ARTIFACTS_DIR / "cnn"
MODEL_PATH = MODEL_DIR / "ai4i_cnn.keras"
PREPROCESSOR_PATH = MODEL_DIR / "ai4i_cnn_preprocessor.joblib"
METADATA_PATH = MODEL_DIR / "ai4i_cnn_metadata.json"
OUT_DIR = OUTPUTS_DIR / "cnn"


def plot_history(history, path):
    """Plot metrics per epoch; training loss is class-weighted, validation loss is unweighted."""
    metrics = ["loss", "pr_auc", "accuracy"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4))
    for ax, metric in zip(axes, metrics):
        epochs = range(1, len(history[metric]) + 1)
        if metric == "loss":
            ax.plot(epochs, history[metric], marker="o", label="train (class-weighted)")
            ax.plot(epochs, history[f"val_{metric}"], marker="o", label="validation (unweighted)")
        else:
            ax.plot(epochs, history[metric], marker="o", label="train")
            ax.plot(epochs, history[f"val_{metric}"], marker="o", label="validation")
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.set_title(f"CNN {metric}")
        if metric == "loss":
            ax.text(0.5, -0.30, "Different weighting: do not compare these loss values directly.",
                    transform=ax.transAxes, ha="center", va="top", fontsize="small")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key in ["epochs", "batch_size", "learning_rate", "dropout", "seed"]:
        parser.add_argument(f"--{key.replace('_', '-')}", type=type(CONFIG[key]), default=CONFIG[key])
    config = {**CONFIG, **vars(parser.parse_args())}

    keras.utils.set_random_seed(config["seed"])  # seeds Python, NumPy and the backend
    splits, prep = load_ai4i_splits(val_seed=config["seed"])
    (x_train, y_train, train_rows), (x_val, y_val, val_rows) = splits["train"], splits["val"]
    test_rows = splits["test_rows"]
    sizes = {"train": len(y_train), "val": len(y_val), "test": len(test_rows)}
    feature_names = prep.get_feature_names_out().tolist()
    print(f"Split sizes: {sizes}  input shape: {x_train.shape[1:]}  features: {feature_names}")
    print(f"Failure rate  train: {y_train.mean():.2%}  val: {y_val.mean():.2%}")

    class_weight = None
    if config["balanced_class_weight"]:
        counts = np.bincount(y_train.astype(int), minlength=2)
        class_weight = {c: len(y_train) / (2 * counts[c]) for c in (0, 1)}

    model = build_cnn(n_features=x_train.shape[1], conv_filters=config["conv_filters"],
                      kernel_size=config["kernel_size"], dense_units=config["dense_units"],
                      dropout=config["dropout"], learning_rate=config["learning_rate"])
    model.summary()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(prep, PREPROCESSOR_PATH)  # needed to transform the held-out rows identically at evaluation
    callbacks = [
        keras.callbacks.EarlyStopping(monitor=MONITOR, mode="max", patience=config["early_stop_patience"],
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor=MONITOR, mode="max", factor=0.5,
                                          patience=config["lr_patience"], min_lr=1e-5, verbose=1),
        # Saves the best-val-PR-AUC model, so the file is the best epoch even if training ran to the end.
        keras.callbacks.ModelCheckpoint(str(MODEL_PATH), monitor=MONITOR, mode="max", save_best_only=True),
        keras.callbacks.CSVLogger(str(OUT_DIR / "history.csv")),
    ]
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=config["epochs"],
                        batch_size=config["batch_size"], class_weight=class_weight, callbacks=callbacks,
                        verbose=2).history
    history = {k: [float(v) for v in vals] for k, vals in history.items()}

    best_epoch = max(range(len(history[MONITOR])), key=history[MONITOR].__getitem__)
    metadata = {
        "model": "ai4i_cnn",
        "model_file": MODEL_PATH.name,
        "preprocessor_file": PREPROCESSOR_PATH.name,
        "target": TARGET,
        "positive_class": 1,
        "output": "sigmoid probability of machine failure (class 1); threshold 0.5 by default",
        "raw_features": {"categorical": CATEGORICAL, "numeric": NUMERIC},
        "input_features": feature_names,
        "input_shape": list(x_train.shape[1:]),
        "excluded_columns": {"identifiers": ID_COLUMNS, "target_leakage": LEAKAGE_COLUMNS},
        "split": {"seed": SEED, "test_size": TEST_SIZE, "val_size": VAL_SIZE, "val_seed": config["seed"],
                  "sizes": sizes, "train_failure_rate": float(y_train.mean()),
                  "val_failure_rate": float(y_val.mean())},
        # DataFrame row positions in ai4i2020.csv (UDI = row + 1), to verify the split matches the classical models.
        "rows": {"train": train_rows.tolist(), "val": val_rows.tolist(), "test": test_rows.tolist()},
        "config": config,
        "class_weight": class_weight,
        "selection_metric": MONITOR,
        "epochs_run": len(history["loss"]),
        "best_epoch": best_epoch + 1,
        "best_val_metrics": {k: v[best_epoch] for k, v in history.items() if k.startswith("val_")},
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    plot_history(history, OUT_DIR / "learning_curves.png")
    print(f"\nBest epoch {best_epoch + 1}: " +
          "  ".join(f"{k}={v:.4f}" for k, v in metadata["best_val_metrics"].items()))
    print(f"Model, preprocessor and metadata saved in {MODEL_DIR}\nHistory and plots in {OUT_DIR}")


if __name__ == "__main__":
    main()
