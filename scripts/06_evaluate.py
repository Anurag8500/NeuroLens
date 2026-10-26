import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    summary_path = "outputs/results_summary.csv"
    fig_path = "outputs/figures/model_comparison_benchmark.png"

    if not os.path.exists(summary_path):
        print(f"[ERROR] {summary_path} not found. Train models first.")
        return

    df = pd.read_csv(summary_path)

    print("\n" + "=" * 80)
    print("           NEUROLENS — COMPREHENSIVE BENCHMARK RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    # Plot comparative chart
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    plt.figure(figsize=(10, 6))
    
    # Create descriptive labels
    df["label"] = df["model"] + "\n(" + df["split_type"].apply(lambda s: "Patient-Disjoint" if s == "patient_disjoint" else "Leaky Split") + ")"

    x = np.arange(len(df))
    width = 0.25

    plt.bar(x - width, df["accuracy"] * 100, width, label="Accuracy", color="#2b5c8f", edgecolor="black")
    plt.bar(x, df["macro_f1"] * 100, width, label="Macro-F1", color="#3caea3", edgecolor="black")
    plt.bar(x + width, df["balanced_accuracy"] * 100, width, label="Balanced Acc", color="#ed553b", edgecolor="black")

    plt.xticks(x, df["label"], fontsize=9)
    plt.ylabel("Score (%)", fontsize=11)
    plt.title("NeuroLens — Brain Tumor MRI Classification Model Comparison", fontsize=13, pad=15)
    plt.ylim(50, 105)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"\nGenerated benchmark comparison plot: {fig_path}")


if __name__ == "__main__":
    main()
