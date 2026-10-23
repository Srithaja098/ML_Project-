"""Shared settings. Paths are relative to the project root (this file's folder), so nothing is machine-specific."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"            # downloaded datasets (git-ignored)
ARTIFACTS_DIR = ROOT / "artifacts"  # trained models + metadata (git-ignored)
OUTPUTS_DIR = ROOT / "outputs"      # CV tables, histories, plots (git-ignored)

SEED = 42
TEST_SIZE = 0.20  # assignment's 80:20 split, used by both parts
