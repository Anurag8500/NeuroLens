import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from src.evaluate import compute_metrics


def compute_class_weights(train_labels: list, num_classes: int = 3) -> torch.Tensor:
    """Compute normalized inverse class frequency weights from training labels."""
    counts = np.bincount(train_labels, minlength=num_classes).astype(np.float32)
    # Inverse frequency
    weights = 1.0 / (counts + 1e-5)
    # Normalize to mean 1.0
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None and device.type == "cuda":
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(np.array(all_targets), np.array(all_preds))
    metrics["loss"] = avg_loss
    return metrics


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            if device.type == "cuda":
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    avg_loss = total_loss / len(loader.dataset)
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    metrics = compute_metrics(y_true, y_pred, y_probs=y_probs)
    metrics["loss"] = avg_loss
    return metrics, y_true, y_pred, y_probs


def fit_model(
    model,
    train_loader,
    val_loader,
    class_weights: torch.Tensor,
    epochs: int = 25,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    patience: int = 7,
    save_path: str = None,
    device: torch.device = None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler() if (device.type == "cuda") else None

    best_macro_f1 = -1.0
    best_weights = None
    patience_counter = 0

    history = {
        "train_loss": [], "val_loss": [],
        "train_f1": [], "val_f1": [],
        "train_bal_acc": [], "val_bal_acc": []
    }

    print(f"\n--- Training on {device} ({epochs} epochs max, early stop patience {patience}) ---")

    for epoch in range(1, epochs + 1):
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_m, _, _, _ = evaluate_model(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_m["loss"])
        history["val_loss"].append(val_m["loss"])
        history["train_f1"].append(train_m["macro_f1"])
        history["val_f1"].append(val_m["macro_f1"])
        history["train_bal_acc"].append(train_m["balanced_accuracy"])
        history["val_bal_acc"].append(val_m["balanced_accuracy"])

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_m['loss']:.4f} | Val Loss: {val_m['loss']:.4f} | "
            f"Val Macro-F1: {val_m['macro_f1']:.4f} | Val Bal-Acc: {val_m['balanced_accuracy']:.4f}"
        )

        # Early stopping on validation Macro-F1
        if val_m["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_m["macro_f1"]
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(best_weights, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best Val Macro-F1: {best_macro_f1:.4f}")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history
