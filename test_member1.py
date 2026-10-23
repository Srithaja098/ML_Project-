"""Self-check for the shared split and leakage-sensitive data logic, on synthetic data (no downloads).

Needs numpy, pandas and scikit-learn; the CNN model checks run only if TensorFlow/Keras is installed.

Run from the repo root:  python -m test_member1
"""
import numpy as np
import pandas as pd

from classification.data import (ID_COLUMNS, LEAKAGE_COLUMNS, NUMERIC, TARGET, TYPE_ORDER,
                                         make_preprocessor, split_features_target, stratified_split)
# --- classification: no IDs / failure modes in X, stratified 80:20 split, preprocessors fit ---
n = 200
rng = np.random.default_rng(0)
df = pd.DataFrame({"UDI": range(n), "Product ID": [f"M{i}" for i in range(n)],
                   "Type": rng.choice(TYPE_ORDER, n), **{c: rng.normal(300, 10, n) for c in NUMERIC},
                   TARGET: (np.arange(n) % 10 == 0).astype(int)})
for c in LEAKAGE_COLUMNS:
    df[c] = df[TARGET]

X, y = split_features_target(df)
assert not set(X.columns) & set(ID_COLUMNS + LEAKAGE_COLUMNS + [TARGET])
X_tr, X_te, y_tr, y_te = stratified_split(X, y)
assert len(X_te) == 40 and y_te.sum() == 4, "20% test split must keep the 10% failure rate"
assert make_preprocessor(scale=True).fit_transform(X_tr).shape == (160, 3 + len(NUMERIC))  # one-hot Type
assert make_preprocessor(scale=False).fit_transform(X_tr).shape == (160, 1 + len(NUMERIC))  # ordinal Type
try:
    split_features_target(df.drop(columns="Torque [Nm]"))
    raise AssertionError("missing column should raise")
except ValueError:
    pass

# --- CNN: same train/test rows as the classical models, validation carved from train only ---
from cnn.data_loader import make_splits, to_cnn_input

n = 1000
df = pd.DataFrame({"UDI": range(1, n + 1), "Product ID": [f"M{i}" for i in range(n)],
                   "Type": rng.choice(TYPE_ORDER, n), **{c: rng.normal(300, 10, n) for c in NUMERIC},
                   TARGET: (np.arange(n) % 20 == 0).astype(int)})
for c in LEAKAGE_COLUMNS:
    df[c] = df[TARGET]
X_tr, X_te, _, _ = stratified_split(*split_features_target(df))  # exactly what classification.train does
s, prep = make_splits(df)
s2, _ = make_splits(df)
(x_fit, y_fit, fit_rows), (x_val, y_val, val_rows) = s["train"], s["val"]
assert s["test_rows"].equals(X_te.index), "CNN held-out rows must equal the classical test rows"
assert fit_rows.union(val_rows).sort_values().equals(X_tr.index.sort_values()), "CNN train+val = classical train"
assert not fit_rows.intersection(val_rows).size and not X_tr.index.intersection(X_te.index).size
assert fit_rows.equals(s2["train"][2]) and val_rows.equals(s2["val"][2]), "split must be reproducible"
assert [len(y_fit), len(y_val), len(s["test_rows"])] == [720, 80, 200]
assert y_fit.sum() == 36 and y_val.sum() == 4, "validation split must be stratified"
feature_names = prep.get_feature_names_out().tolist()
assert len(feature_names) == 3 + len(NUMERIC)
assert not any(leak in f for f in feature_names for leak in ID_COLUMNS + LEAKAGE_COLUMNS + [TARGET])
assert x_fit.shape == (720, 3 + len(NUMERIC), 1) and x_fit.dtype == np.float32
assert np.allclose(x_fit[:, 3:, 0].mean(axis=0), 0, atol=1e-5), "scaler must be fit on CNN fit rows only"
assert to_cnn_input(prep, X_te).shape == (200, 3 + len(NUMERIC), 1)

# --- CNN model + config (needs TensorFlow/Keras) ---
try:
    import keras
except ImportError:
    print("SKIPPED CNN model checks: keras/tensorflow not installed")
else:
    from cnn.model import build_cnn
    from cnn.train import CONFIG, METADATA_PATH, MODEL_PATH, MONITOR, PREPROCESSOR_PATH
    from config import ARTIFACTS_DIR, SEED

    model = build_cnn(n_features=x_fit.shape[1], conv_filters=CONFIG["conv_filters"], dropout=CONFIG["dropout"])
    assert sum(isinstance(l, keras.layers.Conv1D) for l in model.layers) >= 2
    assert model.input_shape == (None, x_fit.shape[1], 1) and model.output_shape == (None, 1)
    assert model.loss == "binary_crossentropy" and model.layers[-1].activation.__name__ == "sigmoid"
    p = model.predict(x_val, verbose=0)
    assert p.shape == (80, 1) and ((p >= 0) & (p <= 1)).all()
    assert MODEL_PATH.suffix == ".keras" and {MODEL_PATH.parent, PREPROCESSOR_PATH.parent, METADATA_PATH.parent} == {
        ARTIFACTS_DIR / "cnn"}
    assert CONFIG["seed"] == SEED and MONITOR.startswith("val_")

print("self-check passed")
