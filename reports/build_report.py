"""Build the submission report PDF from the saved experiment artifacts (no training)."""
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLASSICAL = ("knn", "decision_tree", "svm_rbf")
MODELS = (*CLASSICAL, "cnn")
CLASS_NAMES = {"knn": "KNN", "decision_tree": "Decision Tree", "svm_rbf": "RBF-SVM", "cnn": "1D CNN"}
OUTPUT = ROOT / "reports" / "AI4I_2020_Classification_Report.pdf"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def add_text_page(pdf, title, sections, page_number):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    fig.text(0.08, 0.95, title, fontsize=21, weight="bold", va="top", color="#17324d")
    y = 0.89
    for heading, paragraph in sections:
        fig.text(0.08, y, heading, fontsize=13, weight="bold", va="top", color="#245b78")
        y -= 0.031
        for line in textwrap.wrap(paragraph, width=104):
            fig.text(0.08, y, line, fontsize=10.3, va="top", color="#222222")
            y -= 0.021
        y -= 0.018
    fig.text(0.92, 0.035, str(page_number), ha="right", fontsize=9, color="#666666")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_table_page(pdf, title, tables, page_number):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    fig.text(0.08, 0.95, title, fontsize=21, weight="bold", va="top", color="#17324d")
    for heading_y, bottom, height, heading, columns, rows, widths, font_size in tables:
        fig.text(0.08, heading_y, heading, fontsize=13, weight="bold", color="#245b78")
        ax = fig.add_axes([0.08, bottom, 0.84, height])
        ax.axis("off")
        table = ax.table(cellText=rows, colLabels=columns, colWidths=widths,
                         cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        table.scale(1, 1.4)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor("#cbd5df")
            if row == 0:
                cell.set_facecolor("#17324d")
                cell.get_text().set_color("white")
                cell.get_text().set_weight("bold")
            elif row % 2 == 0:
                cell.set_facecolor("#edf3f7")
    fig.text(0.92, 0.035, str(page_number), ha="right", fontsize=9, color="#666666")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_figure_grid(pdf, title, entries, page_number):
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), facecolor="white")
    fig.suptitle(title, fontsize=20, weight="bold", color="#17324d", y=0.98)
    for ax, (label, path) in zip(axes.flat, entries):
        ax.axis("off")
        image = plt.imread(path)
        ax.imshow(image)
        ax.set_title(label, fontsize=10, pad=4)
    fig.text(0.96, 0.025, str(page_number), ha="right", fontsize=9, color="#666666")
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.94))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_confusion_grid(pdf, matrices, page_number):
    fig, axes = plt.subplots(2, 2, figsize=(8.27, 11.69), facecolor="white")
    fig.suptitle("Held-out confusion matrices", fontsize=21, weight="bold", color="#17324d", y=0.96)
    for ax, model in zip(axes.flat, MODELS):
        matrix = matrices[model].to_numpy(dtype=int)
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_title(CLASS_NAMES[model], fontsize=13, weight="bold")
        ax.set_xticks([0, 1], labels=["No failure", "Failure"])
        ax.set_yticks([0, 1], labels=["No failure", "Failure"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        threshold = matrix.max() / 2
        for row in range(2):
            for col in range(2):
                ax.text(col, row, f"{matrix[row, col]:,}", ha="center", va="center",
                        color="white" if matrix[row, col] > threshold else "#17212b", fontsize=14)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.text(0.94, 0.035, str(page_number), ha="right", fontsize=9, color="#666666")
    fig.tight_layout(rect=(0.04, 0.05, 0.96, 0.91))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    out_class = ROOT / "outputs" / "classification"
    out_cnn = ROOT / "outputs" / "cnn"
    artifacts_class = ROOT / "artifacts" / "classification"
    cnn_meta = read_json(ROOT / "artifacts" / "cnn" / "ai4i_cnn_metadata.json")

    # Refuse to write a report if the persisted split records no longer agree.
    expected_rows = cnn_meta["rows"]["test"]
    for model in CLASSICAL:
        meta = read_json(artifacts_class / f"{model}_metadata.json")
        if meta["rows"]["test"] != expected_rows:
            raise ValueError(f"{model} test indices do not match the CNN metadata")

    metrics = {model: read_json((out_cnn if model == "cnn" else out_class) /
                                f"{model}_test_metrics.json") for model in MODELS}
    cms = {model: pd.read_csv((out_cnn if model == "cnn" else out_class) /
                              f"{model}_confusion_matrix.csv", index_col=0) for model in MODELS}
    cv = {model: pd.read_csv(out_class / f"{model}_cv_results.csv") for model in CLASSICAL}
    history = pd.read_csv(out_cnn / "history.csv")

    # Threshold metrics shown in the report must match the saved confusion matrices.
    for model, matrix in cms.items():
        tn, fp, fn, tp = matrix.to_numpy().ravel()
        total = tn + fp + fn + tp
        expected = {
            "accuracy": (tn + tp) / total,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        }
        if any(abs(metrics[model][key] - value) > 1e-12 for key, value in expected.items()):
            raise ValueError(f"threshold metrics for {model} do not match its saved confusion matrix")

    best_cv = {}
    for model, frame in cv.items():
        index = frame["mean_test_f1"].idxmax()
        best_cv[model] = frame.loc[index]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT, metadata={"Title": "AI4I 2020 Classification Results",
                                    "Subject": "Saved KNN, Decision Tree, RBF-SVM, and tabular CNN experiments"}) as pdf:
        add_text_page(pdf, "AI4I 2020 Predictive Maintenance Classification", [
            ("Executive summary",
             "Four classifiers were evaluated on the same stratified 2,000-row held-out test set. RBF-SVM has the "
             "highest held-out ROC-AUC (0.9767); Decision Tree has the highest failure-class F1 (0.7846). The CNN "
             "has the highest failure recall (0.8824) at threshold 0.5. The results show that model choice depends "
             "on whether ranking quality, balanced failure detection, or missed-failure reduction is prioritized."),
            ("Dataset and task",
             "The UCI AI4I 2020 Predictive Maintenance dataset contains 10,000 rows. The target is Machine failure, "
             "with label 1 representing failure. The positive class is rare, around 3.4 percent. Inputs are product "
             "Type and five sensor measurements: air temperature, process temperature, rotational speed, torque, and "
             "tool wear."),
            ("Leakage controls and preprocessing",
             "UDI and Product ID are excluded as identifiers. TWF, HDF, PWF, OSF, and RNF are excluded because they "
             "are failure-mode labels that leak target information. KNN and RBF-SVM use one-hot Type and standardized "
             "numeric values. Decision Tree uses ordinal Type (L < M < H) and raw numeric values. CNN uses one-hot "
             "Type and standardized numeric values; its preprocessor is fit only on its 7,200-row fit subset."),
            ("CNN interpretation",
             "The CNN is a Conv1D model over the eight preprocessed tabular features in a fixed order, shaped as "
             "(8, 1). Adjacent columns do not have a guaranteed local relationship. Instructor acceptance of this "
             "tabular CNN interpretation has not been confirmed."),
        ], 1)

        add_text_page(pdf, "Experimental design and model selection", [
            ("Shared outer split",
             "All four models use the same stratified 80/20 split with random seed 42: 8,000 training rows and "
             "2,000 held-out test rows. The ordered test indices saved in the three classical metadata files match "
             "the CNN metadata. Test rows were not used for hyperparameter selection or CNN checkpoint selection."),
            ("Classical cross-validation",
             "KNN, Decision Tree, and RBF-SVM use stratified 5-fold cross-validation on training rows. GridSearchCV "
             "selects the candidate with the highest failure-class F1 and refits it on all training rows. "
             "Preprocessing is inside each pipeline. Best settings and CV F1 values below are read from the saved "
             "metadata and CV result tables."),
            ("CNN validation and checkpointing",
             "The CNN separates 800 validation rows from the 8,000-row training portion, leaving 7,200 fit rows. "
             "Class weights are calculated from fit labels only. Early stopping, learning-rate reduction, and "
             "checkpointing monitor validation PR-AUC. The saved best checkpoint is epoch "
             f"{cnn_meta['best_epoch']} of {cnn_meta['epochs_run']}; validation PR-AUC is "
             f"{cnn_meta['best_val_metrics']['val_pr_auc']:.4f} and validation ROC-AUC is "
             f"{cnn_meta['best_val_metrics']['val_roc_auc']:.4f}."),
            ("Metric interpretation",
             "Precision, recall, and F1 treat failure (1) as positive. Thresholded predictions use the saved "
             "threshold of 0.5. ROC-AUC and average precision use continuous positive-class scores: sigmoid "
             "probabilities for CNN and decision_function scores for SVM. Metrics are unweighted. The CNN's "
             "class-weighted training loss is not directly comparable to unweighted validation/test loss."),
        ], 2)

        model_rows = []
        for model in CLASSICAL:
            metadata = read_json(artifacts_class / f"{model}_metadata.json")
            params = metadata["best_params"]
            param_text = ", ".join(f"{key}={value}" for key, value in params.items())
            row = best_cv[model]
            model_rows.append([CLASS_NAMES[model], param_text,
                               f"{row['mean_test_f1']:.4f} ± {row['std_test_f1']:.4f}"])
        test_rows = [[CLASS_NAMES[m], *(f"{metrics[m][key]:.4f}" for key in
                       ("accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"))]
                     for m in MODELS]
        confusion_rows = []
        for model in MODELS:
            tn, fp, fn, tp = cms[model].to_numpy().ravel()
            confusion_rows.append([CLASS_NAMES[model], f"{tn:,}", f"{fp:,}", f"{fn:,}", f"{tp:,}"])
        add_table_page(pdf, "Saved model results", [
            (0.86, 0.70, 0.13, "Selected classical hyperparameters and CV failure-class F1", ["Model", "Selected parameters", "Mean CV F1 ± std"],
             model_rows, [0.19, 0.54, 0.27], 8.2),
            (0.62, 0.41, 0.17, "Held-out test metrics (2,000 shared rows)",
             ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Avg. precision"],
             test_rows, [0.19, 0.13, 0.13, 0.13, 0.12, 0.13, 0.17], 7.7),
            (0.34, 0.12, 0.18, "Confusion counts (failure class 1 is positive)", ["Model", "TN", "FP", "FN", "TP"],
             confusion_rows, [0.32, 0.17, 0.17, 0.17, 0.17], 9),
        ], 3)

        add_confusion_grid(pdf, cms, 4)
        roc_entries = []
        for model in MODELS:
            path = (out_cnn if model == "cnn" else out_class) / f"{model}_roc_curve.png"
            roc_entries.append((f"{CLASS_NAMES[model]} — ROC-AUC {metrics[model]['roc_auc']:.4f}", path))
        add_figure_grid(pdf, "Held-out ROC curves", roc_entries, 5)

        classical_gaps = {}
        for model in CLASSICAL:
            row = best_cv[model]
            classical_gaps[model] = (float(row["mean_train_f1"]), float(row["mean_test_f1"]))
        best_epoch = int(cnn_meta["best_epoch"])
        best_idx = best_epoch - 1
        last = history.iloc[-1]
        cnn_text = (f"CNN validation PR-AUC peaked at {history.iloc[best_idx]['val_pr_auc']:.4f} at epoch "
                    f"{best_epoch}. By epoch {len(history)}, training PR-AUC reached {last['pr_auc']:.4f} while "
                    f"validation PR-AUC was {last['val_pr_auc']:.4f}; validation ROC-AUC changed from "
                    f"{history.iloc[best_idx]['val_roc_auc']:.4f} to {last['val_roc_auc']:.4f}. This supports "
                    "late-stage overfitting; the saved checkpoint is the best validation epoch. Training loss is "
                    "class weighted and validation loss unweighted, so the loss curves should not be compared directly.")
        add_text_page(pdf, "Generalization and overfitting", [
            ("KNN",
             f"At its selected candidate, mean training-fold F1 is {classical_gaps['knn'][0]:.4f}, versus "
             f"mean CV F1 {classical_gaps['knn'][1]:.4f}, a gap of "
             f"{classical_gaps['knn'][0] - classical_gaps['knn'][1]:.4f}. Test F1 is "
             f"{metrics['knn']['f1']:.4f}. The large gap and lower held-out result are clear evidence of a "
             "generalization concern for the selected 1-neighbor model."),
            ("Decision Tree and RBF-SVM",
             f"Decision Tree mean training-fold F1 is {classical_gaps['decision_tree'][0]:.4f} and mean CV F1 is "
             f"{classical_gaps['decision_tree'][1]:.4f}; held-out F1 ({metrics['decision_tree']['f1']:.4f}) is "
             "above its CV mean. RBF-SVM mean training-fold F1 is "
             f"{classical_gaps['svm_rbf'][0]:.4f} versus CV F1 {classical_gaps['svm_rbf'][1]:.4f}; held-out "
             f"F1 ({metrics['svm_rbf']['f1']:.4f}) is close to CV. The training/CV gaps warrant attention but "
             "the held-out results do not show a large additional decline."),
            ("CNN",
             cnn_text),
        ], 6)
        curve_entries = [
            ("KNN: training-fold and CV F1", out_class / "knn_validation_curve.png"),
            ("Decision Tree: training-fold and CV F1", out_class / "decision_tree_validation_curve.png"),
            ("RBF-SVM: training-fold and CV F1", out_class / "svm_rbf_validation_curve.png"),
            ("CNN: training and validation history", out_cnn / "learning_curves.png"),
        ]
        add_figure_grid(pdf, "Training and validation diagnostics", curve_entries, 7)

        add_text_page(pdf, "Limitations, conclusion, and reproducibility", [
            ("Limitations",
             "The failure class is rare, and this report summarizes one fixed train/test split. No repeated-split "
             "uncertainty analysis or statistical significance testing is included. A 1D CNN assumes relationships "
             "between adjacent input positions even though the features are tabular and their order is arbitrary; "
             "instructor acceptance of this interpretation is unresolved. The CNN's threshold is fixed at 0.5 and "
             "was not optimized on held-out test data."),
            ("Conclusion",
             "RBF-SVM provides the strongest held-out ranking performance by ROC-AUC (0.9767), while Decision Tree "
             "achieves the highest failure-class F1 (0.7846). CNN recall is highest (0.8824) but comes with lower "
             "precision (0.3261) and many more false positives. The preferred model therefore depends on the cost "
             "of missed failures versus false alarms; the available results do not establish a deployment choice."),
            ("Reproducing the saved experiment",
             "From the project root, run `python -m classification.train` for classical tuning/training, "
             "`python -m cnn.train` for CNN training, and `python -m cnn.evaluate` to evaluate the saved CNN "
             "checkpoint. Training commands rerun experiments and overwrite outputs. Run `python -m test_member1` "
             "and `python -m unittest -v test_classification_evaluation test_cnn_evaluation test_validation_curve_plotting` "
             "for checks. This report was generated from saved results; models were not retrained."),
            ("Data citation and open submission question",
             "Dataset: AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository, "
             "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset. No assignment "
             "rubric was present in the repository. Confirm the required report format and whether the instructor "
             "accepts the tabular 1D CNN interpretation before final submission."),
        ], 8)

    print(f"Created {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
