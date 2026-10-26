import os
import sys
import json
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import SliceDataset, get_train_transforms, get_eval_transforms
from src.models import create_model
from src.train import compute_class_weights, fit_model, evaluate_model
from src.evaluate import plot_confusion_matrix, plot_roc_curves


def train_single_model(model_name: str, meta: pd.DataFrame, splits: dict, args):
    class_names = ["meningioma", "glioma", "pituitary"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"  Training {model_name.upper()} on Patient-Disjoint Split")
    print(f"=======================================================")

    # 1. Datasets & Loaders
    train_tf = get_train_transforms(image_size=args.image_size)
    eval_tf = get_eval_transforms(image_size=args.image_size)

    train_ds = SliceDataset(meta, splits["train"], args.images_dir, transform=train_tf)
    val_ds = SliceDataset(meta, splits["val"], args.images_dir, transform=eval_tf)
    test_ds = SliceDataset(meta, splits["test"], args.images_dir, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 2. Class weights (from training partition only)
    train_labels = [int(meta[meta["slice_id"] == sid]["label"].values[0]) for sid in splits["train"]]
    class_weights = compute_class_weights(train_labels, num_classes=3)
    print(f"Inverse class weights (train fold): {class_weights.numpy()}")

    # 3. Model creation
    is_pretrained = (model_name.lower() != "simplecnn")
    model = create_model(model_name, num_classes=3, pretrained=is_pretrained)

    # 4. Training
    ckpt_path = os.path.join("outputs/models", f"{model_name.lower()}_patient_best.pt")
    fitted_model, history = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        save_path=ckpt_path,
        device=device
    )

    # 5. Final Evaluation on Locked Test Set
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    test_metrics, y_true, y_pred, y_probs = evaluate_model(fitted_model, test_loader, criterion, device)

    print(f"\n--- {model_name.upper()} FINAL TEST RESULTS ---")
    print(f"Accuracy:         {test_metrics['accuracy']:.4f}")
    print(f"Macro-F1:         {test_metrics['macro_f1']:.4f}")
    print(f"Balanced Acc:     {test_metrics['balanced_accuracy']:.4f}")
    print(f"Macro ROC-AUC:    {test_metrics.get('macro_roc_auc', 'N/A')}")
    for c in class_names:
        print(f"  {c.capitalize():11s} F1: {test_metrics[f'{c}_f1']:.4f}")

    # 6. Plots
    cm_path = f"outputs/figures/confusion_matrix_{model_name.lower()}_patient.png"
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path, title=f"{model_name.upper()} (Patient Split) — Test Confusion Matrix")

    roc_path = f"outputs/figures/roc_{model_name.lower()}_patient.png"
    plot_roc_curves(y_true, y_probs, class_names, roc_path, title=f"{model_name.upper()} (Patient Split) — ROC Curves")

    # 7. Update results_summary.csv
    summary_path = "outputs/results_summary.csv"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    record = {
        "model": model_name.upper(),
        "split_type": "patient_disjoint",
        "accuracy": test_metrics["accuracy"],
        "macro_f1": test_metrics["macro_f1"],
        "balanced_accuracy": test_metrics["balanced_accuracy"],
        "meningioma_f1": test_metrics["meningioma_f1"],
        "glioma_f1": test_metrics["glioma_f1"],
        "pituitary_f1": test_metrics["pituitary_f1"],
        "macro_roc_auc": test_metrics.get("macro_roc_auc", None)
    }

    new_df = pd.DataFrame([record])
    if os.path.exists(summary_path):
        existing_df = pd.read_csv(summary_path)
        combined_df = pd.concat([existing_df[~((existing_df["model"] == record["model"]) & (existing_df["split_type"] == record["split_type"]))], new_df], ignore_index=True)
    else:
        combined_df = new_df
    combined_df.to_csv(summary_path, index=False)
    print(f"Updated results summary in {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Deep Learning Models on Patient-Disjoint MRI Splits")
    parser.add_argument("--model", type=str, default="resnet50", choices=["simplecnn", "resnet50", "efficientnet_b0", "all"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=25, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--image_size", type=int, default=224, help="Input image resolution")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker processes")
    parser.add_argument("--images_dir", type=str, default="data/processed/images", help="Processed images directory")
    args = parser.parse_args()

    meta_path = "data/metadata.csv"
    split_path = "data/splits/patient_split.json"

    if not os.path.exists(meta_path) or not os.path.exists(split_path):
        print("[ERROR] metadata.csv or patient_split.json not found.")
        return

    meta = pd.read_csv(meta_path)
    with open(split_path, "r") as f:
        splits = json.load(f)

    if args.model == "all":
        models_to_run = ["simplecnn", "resnet50", "efficientnet_b0"]
    else:
        models_to_run = [args.model]

    for m in models_to_run:
        train_single_model(m, meta, splits, args)


if __name__ == "__main__":
    main()
