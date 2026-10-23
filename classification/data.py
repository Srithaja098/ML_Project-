"""AI4I 2020 Predictive Maintenance: loading, feature/target separation, stratified split, preprocessing.

The evaluation phase should call `load_splits()` to get exactly the same held-out test set.
"""
import io
import urllib.request
import zipfile

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from config import DATA_DIR, SEED, TEST_SIZE

UCI_PAGE = "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
CSV_PATH = DATA_DIR / "ai4i2020.csv"

TARGET = "Machine failure"
ID_COLUMNS = ["UDI", "Product ID"]                      # identifiers, carry no signal
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]   # failure modes = the target itself, so excluded
CATEGORICAL = ["Type"]
NUMERIC = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TYPE_ORDER = ["L", "M", "H"]  # product quality: low / medium / high


def download_dataset(path=CSV_PATH):
    """Fetch the CSV from the official UCI zip, only if it is not already on disk."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading AI4I 2020 from {UCI_ZIP_URL}")
    try:
        with urllib.request.urlopen(UCI_ZIP_URL, timeout=60) as resp:
            archive = zipfile.ZipFile(io.BytesIO(resp.read()))
        csv_name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
    except (OSError, zipfile.BadZipFile, StopIteration) as e:
        raise RuntimeError(
            f"Could not download the dataset ({e!r}).\n"
            f"Download it manually from {UCI_PAGE} and save ai4i2020.csv as {path}"
        ) from e
    path.write_bytes(archive.read(csv_name))
    return path


def load_dataset(path=CSV_PATH):
    """Read the AI4I CSV (downloading it once if missing)."""
    return pd.read_csv(download_dataset(path))


def describe_dataset(df):
    """Print column types, missing values and class balance."""
    print(f"Shape: {df.shape}")
    print("\nColumn dtypes:\n" + df.dtypes.to_string())
    print("\nMissing values per column:\n" + df.isna().sum().to_string())
    counts = df[TARGET].value_counts().sort_index()
    print(f"\nClass balance ({TARGET}):\n" + pd.DataFrame(
        {"count": counts, "fraction": (counts / len(df)).round(4)}).to_string())


def split_features_target(df):
    """Return predictors X (Type + numeric sensors) and target y. IDs and failure-mode columns are dropped."""
    missing = [c for c in CATEGORICAL + NUMERIC + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns {missing}. Found: {list(df.columns)}")
    if df[CATEGORICAL + NUMERIC + [TARGET]].isna().any().any():
        raise ValueError("Dataset has missing values; add an imputer to make_preprocessor() before training.")
    return df[CATEGORICAL + NUMERIC], df[TARGET].astype(int)


def stratified_split(X, y, seed=SEED, test_size=TEST_SIZE):
    """80:20 split that keeps the (rare) failure rate equal in train and test."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def load_splits(path=CSV_PATH):
    """Convenience: dataset -> X_train, X_test, y_train, y_test."""
    return stratified_split(*split_features_target(load_dataset(path)))


def make_preprocessor(scale):
    """Column-wise preprocessing; it lives inside each model Pipeline so it is fit on training folds only.

    scale=True  (KNN, SVM): one-hot Type + standardised numerics (both models are distance-based).
    scale=False (tree):     ordinal Type (L<M<H) + raw numerics (splits are scale-invariant).
    """
    if scale:
        return ColumnTransformer([
            ("type", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ])
    return ColumnTransformer([
        ("type", OrdinalEncoder(categories=[TYPE_ORDER], handle_unknown="use_encoded_value", unknown_value=-1),
         CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])
