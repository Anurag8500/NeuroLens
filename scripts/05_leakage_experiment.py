import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import SliceDataset, get_train_transforms, get_eval_transforms
from src.models import create_model
from src.train import compute_class_weights, fit_model, evaluate_model
from src.evaluate import plot_confusion_matrix, plot_roc_curves


def main():
    parser = argparse.ArgumentParser(description="NeuroLens Experiment 0: Leakage Demonstration")
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and evaluate existing checkpoint")
    args = parser.parse_args()

    meta_path = "data/metadata.csv"
    leaky_split_path = "data/splits/leaky_split.json"
    summary_path = "outputs/results_summary.csv"
    class_names = ["meningioma", "glioma", "pituitary"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(meta_path) or not os.path.exists(leaky_split_path):
        print("[ERROR] metadata.csv or leaky_split.json missing.")
        return

    meta = pd.read_csv(meta_path)
    with open(leaky_split_path, "r") as f:
        leaky_splits = json.load(f)

    print("\n" + "=" * 60)
    print("  EXPERIMENT 0: LEAKAGE DEMONSTRATION EXPERIMENT")
    print("  Training/Evaluating ResNet-50 on Random Leaky Slice Split")
    print("=" * 60)

    # 1. Datasets & Loaders
    train_tf = get_train_transforms(image_size=224)
    eval_tf = get_eval_transforms(image_size=224)

    train_ds = SliceDataset(meta, leaky_splits["train"], "data/processed/images", transform=train_tf)
    val_ds = SliceDataset(meta, leaky_splits["val"], "data/processed/images", transform=eval_tf)
    test_ds = SliceDataset(meta, leaky_splits["test"], "data/processed/images", transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 2. Class weights
    train_labels = [int(meta[meta["slice_id"] == sid]["label"].values[0]) for sid in leaky_splits["train"]]
    class_weights = compute_class_weights(train_labels, num_classes=3)

    # 3. Model
    ckpt_path = "outputs/models/resnet50_leaky_best.pt"
    if args.eval_only and os.path.exists(ckpt_path):
        print(f"Loading existing checkpoint from {ckpt_path} for evaluation...")
        model = create_model("resnet50", num_classes=3, pretrained=False)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        fitted_model = model.to(device)
    else:
        model = create_model("resnet50", num_classes=3, pretrained=True)
        # 4. Train
        fitted_model, _ = fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=class_weights,
            epochs=args.epochs,
            lr=1e-4,
            weight_decay=0.01,
            patience=7,
            save_path=ckpt_path,
            device=device
        )

    # 5. Evaluate on Leaky Test Set
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    leaky_metrics, y_true, y_pred, y_probs = evaluate_model(fitted_model, test_loader, criterion, device)

    print("\n--- RESNET-50 LEAKY SPLIT RESULTS ---")
    print(f"Accuracy:         {leaky_metrics['accuracy']:.4f}")
    print(f"Macro-F1:         {leaky_metrics['macro_f1']:.4f}")
    print(f"Balanced Acc:     {leaky_metrics['balanced_accuracy']:.4f}")

    # Plot leaky confusion matrix
    cm_path = "outputs/figures/confusion_matrix_resnet50_leaky.png"
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path, title="ResNet-50 (LEAKY Split) — Confusion Matrix")

    # 6. Record in summary table
    record = {
        "model": "RESNET50",
        "split_type": "random_leaky",
        "accuracy": leaky_metrics["accuracy"],
        "macro_f1": leaky_metrics["macro_f1"],
        "balanced_accuracy": leaky_metrics["balanced_accuracy"],
        "meningioma_f1": leaky_metrics["meningioma_f1"],
        "glioma_f1": leaky_metrics["glioma_f1"],
        "pituitary_f1": leaky_metrics["pituitary_f1"],
        "macro_roc_auc": leaky_metrics.get("macro_roc_auc", None)
    }

    if os.path.exists(summary_path):
        existing_df = pd.read_csv(summary_path)
        combined_df = pd.concat([existing_df[~((existing_df["model"] == "RESNET50") & (existing_df["split_type"] == "random_leaky"))], pd.DataFrame([record])], ignore_index=True)
    else:
        combined_df = pd.DataFrame([record])
    combined_df.to_csv(summary_path, index=False)

    # 7. Compare Patient-Disjoint vs. Leaky and plot the Inflation Gap
    summary_df = pd.read_csv(summary_path)
    res_patient = summary_df[(summary_df["model"] == "RESNET50") & (summary_df["split_type"] == "patient_disjoint")]
    res_leaky = summary_df[(summary_df["model"] == "RESNET50") & (summary_df["split_type"] == "random_leaky")]

    if len(res_patient) > 0 and len(res_leaky) > 0:
        p_row = res_patient.iloc[0]
        l_row = res_leaky.iloc[0]

        metrics_names = ["Accuracy", "Macro-F1", "Balanced Acc"]
        p_scores = [p_row["accuracy"] * 100, p_row["macro_f1"] * 100, p_row["balanced_accuracy"] * 100]
        l_scores = [l_row["accuracy"] * 100, l_row["macro_f1"] * 100, l_row["balanced_accuracy"] * 100]

        x = np.arange(len(metrics_names))
        width = 0.35

        plt.figure(figsize=(8, 5))
        plt.bar(x - width/2, p_scores, width, label="Patient-Disjoint Split (Honest)", color="#2b5c8f", edgecolor="black")
        plt.bar(x + width/2, l_scores, width, label="Random Slice Split (Leaky / Inflated)", color="#d9534f", edgecolor="black")

        for i in range(len(metrics_names)):
            gap = l_scores[i] - p_scores[i]
            plt.text(x[i] - width/2, p_scores[i] + 1.0, f"{p_scores[i]:.1f}%", ha="center", fontsize=9, fontweight="bold")
            plt.text(x[i] + width/2, l_scores[i] + 1.0, f"{l_scores[i]:.1f}%", ha="center", fontsize=9, fontweight="bold")
            plt.text(x[i], max(p_scores[i], l_scores[i]) + 5.0, f"Δ +{gap:.1f}%", ha="center", color="#b52b27", fontsize=10, fontweight="bold")

        plt.ylabel("Score (%)", fontsize=11)
        plt.title("NeuroLens — The Leakage Inflation Gap (ResNet-50)", fontsize=13, pad=15)
        plt.xticks(x, metrics_names, fontsize=10)
        plt.ylim(50, 110)
        plt.legend(loc="lower right", frameon=True)
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()

        gap_plot_path = "outputs/figures/leakage_inflation_gap.png"
        plt.savefig(gap_plot_path, dpi=300)
        plt.close()
        print(f"\n[PROOF] Generated Leakage Inflation Gap plot: {gap_plot_path}")


if __name__ == "__main__":
    main()
