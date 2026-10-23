"""Concrete Compressive Strength dataset loading, splitting, and preprocessing.

UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
"""
from pathlib import Path
import urllib.request
import zipfile
import io
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import DATA_DIR, SEED, TEST_SIZE

UCI_PAGE = "https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip"
CSV_PATH = DATA_DIR / "concrete.csv"

FEATURES = [
    "Cement",
    "Blast Furnace Slag",
    "Fly Ash",
    "Water",
    "Superplasticizer",
    "Coarse Aggregate",
    "Fine Aggregate",
    "Age",
]

TARGET = "Concrete compressive strength"


def download_dataset(path: Path = CSV_PATH) -> Path:
    """Fetch Concrete Compressive Strength dataset, saving locally to CSV."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching Concrete Compressive Strength dataset (UCI ID 165)...")
    
    # 1. Try ucimlrepo
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=165)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        # Rename target if needed to match standard
        if TARGET not in df.columns and len(ds.data.targets.columns) == 1:
            df.rename(columns={ds.data.targets.columns[0]: TARGET}, inplace=True)
        df.to_csv(path, index=False)
        print(f"Successfully downloaded and saved {len(df)} rows to {path}")
        return path
    except Exception as e_uci:
        print(f"ucimlrepo failed ({e_uci!r}), attempting fallback to UCI zip...")

    # 2. Fallback to UCI direct zip archive
    try:
        req = urllib.request.Request(UCI_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            archive = zipfile.ZipFile(io.BytesIO(resp.read()))
            # Look for xls or csv
            names = archive.namelist()
            xls_name = next((n for n in names if n.lower().endswith((".xls", ".xlsx"))), None)
            if xls_name:
                excel_bytes = archive.read(xls_name)
                df = pd.read_excel(io.BytesIO(excel_bytes))
                # Map column names if needed
                if len(df.columns) == 9:
                    df.columns = FEATURES + [TARGET]
                df.to_csv(path, index=False)
                return path
    except Exception as e_zip:
        raise RuntimeError(
            f"Could not download dataset ({e_zip!r}). Please manually download from {UCI_PAGE} and place at {path}"
        ) from e_zip

    return path


def load_dataset(path: Path = CSV_PATH) -> pd.DataFrame:
    """Read the Concrete CSV (downloading if missing)."""
    return pd.read_csv(download_dataset(path))


def describe_dataset(df: pd.DataFrame):
    """Print dataset summary statistics, column types, and missing values."""
    print("=" * 65)
    print("CONCRETE COMPRESSIVE STRENGTH DATASET SUMMARY")
    print("=" * 65)
    print(f"Shape: {df.shape} (1030 rows, 8 features + 1 target)")
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    print(f"\nTarget summary ('{TARGET}'):")
    print(df[TARGET].describe().round(2).to_string())
    print("\nFeature statistics:")
    print(df[FEATURES].describe().round(2).to_string())
    print("=" * 65)


def split_features_target(df: pd.DataFrame):
    """Separate input features X and continuous target y."""
    missing = [col for col in FEATURES + [TARGET] if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}. Found: {list(df.columns)}")
    X = df[FEATURES].copy()
    y = df[TARGET].copy().astype(float)
    return X, y


def train_test_split_dataset(X: pd.DataFrame, y: pd.Series, seed: int = SEED, test_size: float = TEST_SIZE):
    """Standard 80:20 train/test split shared across all models.
    
    Training: 824 instances (80%)
    Held-out Test: 206 instances (20%)
    """
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def load_splits(path: Path = CSV_PATH):
    """Convenience helper: returns X_train, X_test, y_train, y_test."""
    df = load_dataset(path)
    X, y = split_features_target(df)
    return train_test_split_dataset(X, y)


def make_preprocessor():
    """StandardScaler preprocessor fitted within CV training folds only."""
    return StandardScaler()
