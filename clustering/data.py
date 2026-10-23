"""Palmer Penguins dataset loading, cleaning, feature preparation and standardization.

UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/690/palmer+penguins-3
"""
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import DATA_DIR, SEED

UCI_PAGE = "https://archive.ics.uci.edu/dataset/690/palmer+penguins-3"
GITHUB_RAW_URL = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"
CSV_PATH = DATA_DIR / "penguins.csv"

# Continuous morphological measurements used for clustering:
NUMERIC_FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

# Ground-truth species target (Adelie, Gentoo, Chinstrap) for external validation:
TARGET = "species"

# Metadata / categorical attributes:
CATEGORICAL_FEATURES = ["island", "sex"]


def download_dataset(path: Path = CSV_PATH) -> Path:
    """Download Palmer Penguins CSV if not present locally."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Palmer Penguins dataset from {GITHUB_RAW_URL}")
    try:
        urllib.request.urlretrieve(GITHUB_RAW_URL, path)
    except Exception as e:
        # Fallback to seaborn's penguins dataset if network fails
        try:
            import seaborn as sns
            df = sns.load_dataset("penguins")
            df.to_csv(path, index=False)
            print("Successfully loaded and cached Palmer Penguins via seaborn fallback.")
            return path
        except Exception as e_inner:
            raise RuntimeError(
                f"Could not download Palmer Penguins dataset ({e!r}).\n"
                f"Please manually download the CSV and place it at {path.resolve()}"
            ) from e_inner
    return path


def load_dataset(path: Path = CSV_PATH) -> pd.DataFrame:
    """Read the Palmer Penguins CSV (downloading if missing)."""
    return pd.read_csv(download_dataset(path))


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing numerical morphological measurements.
    
    The raw dataset has 344 rows, with exactly 2 rows missing all 4 physical measurements.
    Dropping them yields 342 complete observations across the 3 species.
    """
    df_clean = df.dropna(subset=NUMERIC_FEATURES).copy()
    df_clean.reset_index(drop=True, inplace=True)
    return df_clean


def prepare_clustering_data(df: pd.DataFrame):
    """Separate numerical feature matrix X, ground-truth species y, and metadata.
    
    Returns:
        X (pd.DataFrame): 4 numeric features
        y (pd.Series): True species labels
        metadata (pd.DataFrame): island, sex, year
    """
    df_clean = clean_dataset(df)
    X = df_clean[NUMERIC_FEATURES].copy()
    y = df_clean[TARGET].copy()
    meta_cols = [c for c in CATEGORICAL_FEATURES + ["year"] if c in df_clean.columns]
    metadata = df_clean[meta_cols].copy()
    return X, y, metadata


def scale_features(X: pd.DataFrame, scaler: StandardScaler = None):
    """Standardize features with StandardScaler (zero mean, unit variance).
    
    Distance-based algorithms (K-Means, Agglomerative, DBSCAN) are scale-sensitive,
    making standardization essential for valid distance computations.
    
    Returns:
        X_scaled (np.ndarray): Scaled features
        fitted_scaler (StandardScaler): The fitted scaler
    """
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler


def describe_dataset(df: pd.DataFrame):
    """Print dataset summary stats, missing values, and species distribution."""
    print("=" * 60)
    print("PALMER PENGUINS DATASET SUMMARY")
    print("=" * 60)
    print(f"Total raw rows: {len(df)}")
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    print("\nSpecies distribution (raw):")
    print(df[TARGET].value_counts().to_string())
    df_clean = clean_dataset(df)
    print(f"\nRows after removing missing morphological values: {len(df_clean)}")
    print("\nNumerical features descriptive statistics (cleaned):")
    print(df_clean[NUMERIC_FEATURES].describe().round(2).to_string())
    print("=" * 60)
