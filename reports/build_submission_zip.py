"""Create a curated project archive without Git, caches, or local environments."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "AI4I_2020_Submission"
OUTPUT = ROOT / "submission" / "AI4I_2020_Submission.zip"

FILES = [
    "README.md", "REPORT_OUTLINE.md", "requirements.txt", "config.py",
    "classification/data.py", "classification/train.py",
    "cnn/data_loader.py", "cnn/model.py", "cnn/train.py", "cnn/evaluate.py",
    "test_member1.py", "test_classification_evaluation.py", "test_cnn_evaluation.py",
    "test_validation_curve_plotting.py",
    "reports/__init__.py", "reports/build_report.py", "reports/build_submission_zip.py",
    "reports/AI4I_2020_Classification_Report.pdf",
    "data/ai4i2020.csv",
]


def add_tree(files, directory, allowed_suffixes):
    base = ROOT / directory
    for path in sorted(base.rglob("*")):
        if path.is_file() and path.suffix.lower() in allowed_suffixes:
            files.append(path.relative_to(ROOT).as_posix())


def main():
    files = list(FILES)
    add_tree(files, "artifacts/classification", {".joblib", ".json"})
    add_tree(files, "artifacts/cnn", {".keras", ".joblib", ".json"})
    add_tree(files, "outputs/classification", {".csv", ".json", ".png"})
    add_tree(files, "outputs/cnn", {".csv", ".json", ".png"})
    missing = [name for name in files if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"required submission files are missing: {missing}")
    if len(files) != len(set(files)):
        raise ValueError("duplicate paths in submission file list")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=7) as archive:
        for name in files:
            archive.write(ROOT / name, f"{ARCHIVE_ROOT}/{name}")

    with zipfile.ZipFile(OUTPUT, "r") as archive:
        names = archive.namelist()
        forbidden_parts = ("/.git/", "/.venv/", "/__pycache__/", "/.pytest_cache/")
        if any(part in f"/{name}" for name in names for part in forbidden_parts):
            raise ValueError("archive contains a forbidden environment or cache path")
        corrupt = archive.testzip()
        if corrupt:
            raise ValueError(f"archive integrity check failed at {corrupt}")
        required = {
            f"{ARCHIVE_ROOT}/README.md",
            f"{ARCHIVE_ROOT}/reports/AI4I_2020_Classification_Report.pdf",
            f"{ARCHIVE_ROOT}/classification/train.py",
            f"{ARCHIVE_ROOT}/cnn/train.py",
            f"{ARCHIVE_ROOT}/outputs/classification/knn_test_metrics.json",
            f"{ARCHIVE_ROOT}/outputs/cnn/cnn_test_metrics.json",
        }
        if not required.issubset(names):
            raise ValueError(f"archive is missing required entries: {sorted(required - set(names))}")
        print(f"Created and verified {OUTPUT}")
        print(f"Entries: {len(names)}; uncompressed bytes: {sum(info.file_size for info in archive.infolist())}")
        for name in names:
            print(name)


if __name__ == "__main__":
    main()
