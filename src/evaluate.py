import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray = None, class_names: list = None) -> dict:
    """Compute comprehensive classification metrics."""
    if class_names is None:
        class_names = ["meningioma", "glioma", "pituitary"]

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))

    per_class_f1 = f1_score(y_true, y_pred, average=None)
    per_class_p = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_r = recall_score(y_true, y_pred, average=None, zero_division=0)

    results = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1
    }

    for i, cname in enumerate(class_names):
        results[f"{cname}_f1"] = float(per_class_f1[i])
        results[f"{cname}_precision"] = float(per_class_p[i])
        results[f"{cname}_recall"] = float(per_class_r[i])

    if y_probs is not None:
        try:
            auc = float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro"))
            results["macro_roc_auc"] = auc
        except Exception:
            results["macro_roc_auc"] = None

    return results


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list, save_path: str, title: str = "Confusion Matrix"):
    """Plot row-normalized confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.title(title, fontsize=12, pad=12)
    plt.xlabel("Predicted Class", fontsize=10)
    plt.ylabel("True Class", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curves(y_true: np.ndarray, y_probs: np.ndarray, class_names: list, save_path: str, title: str = "One-vs-Rest ROC Curves"):
    """Plot multiclass One-vs-Rest ROC curves."""
    plt.figure(figsize=(7, 6))

    for i, cname in enumerate(class_names):
        binary_true = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_true, y_probs[:, i])
        try:
            auc_val = roc_auc_score(binary_true, y_probs[:, i])
            plt.plot(fpr, tpr, lw=2, label=f"{cname} (AUC = {auc_val:.3f})")
        except Exception:
            plt.plot(fpr, tpr, lw=2, label=f"{cname}")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=10)
    plt.ylabel("True Positive Rate", fontsize=10)
    plt.title(title, fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
