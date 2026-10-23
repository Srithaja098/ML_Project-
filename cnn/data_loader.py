"""AI4I 2020 data for the 1D CNN, built on the exact split the classical models use.

    classification.data.stratified_split (80:20, SEED) -> train / held-out test  (same rows as KNN, DT, SVM)
    10% of the train portion (stratified)              -> CNN validation (early stopping, LR schedule, checkpoint)
The preprocessor (one-hot Type + StandardScaler, as for KNN/SVM) is fit on the CNN fit rows only.
Each row becomes a length-n_features sequence with one channel, (n, 8, 1): no image reshape.
The held-out test rows are returned only as row indices; their features are never transformed or used here.
"""
import numpy as np
from sklearn.model_selection import train_test_split

from classification.data import load_dataset, make_preprocessor, split_features_target, stratified_split
from config import SEED

VAL_SIZE = 0.10


def to_cnn_input(preprocessor, X):
    """Raw AI4I predictors (Type + 5 numerics) -> float32 (n, n_features, 1). Use the saved preprocessor at evaluation."""
    return preprocessor.transform(X).astype("float32")[..., np.newaxis]


def make_splits(df, val_seed=SEED, val_size=VAL_SIZE):
    """Return ({"train"|"val": (x, y, row_index), "test_rows": index}, fitted preprocessor).

    The outer train/test split always uses config.SEED so it matches the classical models; val_seed only
    moves the validation rows inside the training portion.
    """
    X_train, X_test, y_train, _ = stratified_split(*split_features_target(df))
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, random_state=val_seed)
    prep = make_preprocessor(scale=True).set_params(sparse_threshold=0).fit(X_fit)  # dense output for Keras
    splits = {name: (to_cnn_input(prep, X), y.to_numpy("float32"), X.index)
              for name, X, y in [("train", X_fit, y_fit), ("val", X_val, y_val)]}
    splits["test_rows"] = X_test.index
    return splits, prep


def load_ai4i_splits(val_seed=SEED):
    """Load ai4i2020.csv (downloaded once to data/ by classification.data) and split it."""
    return make_splits(load_dataset(), val_seed=val_seed)
