import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import extract_slice_features
from src.evaluate import compute_metrics, plot_confusion_matrix


def extract_or_load_features(metadata_df: pd.DataFrame, images_dir: str, cache_path: str = "data/features_cache.npz"):
    """Extract handcrafted features with local disk caching."""
    if os.path.exists(cache_path):
        print(f"Loading cached handcrafted features from {cache_path}...")
        data = np.load(cache_path)
        return data["features"], data["slice_ids"], data["labels"]

    print("Extracting handcrafted features (Intensity Stats + GLCM + HOG)...")
    features_list = []
    slice_ids = []
    labels = []

    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Handcrafted Features"):
        sid = int(row["slice_id"])
        img_path = os.path.join(images_dir, f"{sid:04d}.png")
        feat = extract_slice_features(img_path)

        features_list.append(feat)
        slice_ids.append(sid)
        labels.append(int(row["label"]))

    features = np.array(features_list, dtype=np.float32)
    slice_ids = np.array(slice_ids, dtype=np.int32)
    labels = np.array(labels, dtype=np.int32)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, features=features, slice_ids=slice_ids, labels=labels)
    print(f"Cached {len(features)} feature vectors to {cache_path}.")
    return features, slice_ids, labels


def main():
    meta_path = "data/metadata.csv"
    split_path = "data/splits/patient_split.json"
    images_dir = "data/processed/images"
    summary_path = "outputs/results_summary.csv"
    class_names = ["meningioma", "glioma", "pituitary"]

    if not os.path.exists(meta_path) or not os.path.exists(split_path):
        print("[ERROR] Required metadata or split file missing.")
        return

    meta = pd.read_csv(meta_path)
    with open(split_path, "r") as f:
        splits = json.load(f)

    # 1. Feature Extraction
    features, slice_ids, labels = extract_or_load_features(meta, images_dir)

    # Index mapping
    id_to_idx = {sid: i for i, sid in enumerate(slice_ids)}
    train_idx = [id_to_idx[sid] for sid in splits["train"]]
    val_idx = [id_to_idx[sid] for sid in splits["val"]]
    test_idx = [id_to_idx[sid] for sid in splits["test"]]

    X_train, y_train = features[train_idx], labels[train_idx]
    X_val, y_val = features[val_idx], labels[val_idx]
    X_test, y_test = features[test_idx], labels[test_idx]

    # 2. Fit StandardScaler strictly on training set
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 3. Train Classifiers
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "SVM (RBF)": SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
    }

    results = []

    for name, clf in models.items():
        print(f"\n--- Training Classical ML: {name} ---")
        clf.fit(X_train, y_train)

        # Test evaluation
        test_preds = clf.predict(X_test)
        test_probs = clf.predict_proba(X_test)
        metrics = compute_metrics(y_test, test_preds, y_probs=test_probs, class_names=class_names)

        print(f"[{name}] Test Accuracy: {metrics['accuracy']:.4f} | Macro-F1: {metrics['macro_f1']:.4f} | Balanced Acc: {metrics['balanced_accuracy']:.4f}")
        for c in class_names:
            print(f"  {c} F1: {metrics[f'{c}_f1']:.4f}")

        # Save confusion matrix
        cm_path = f"outputs/figures/confusion_matrix_{name.replace(' ', '_').lower()}.png"
        plot_confusion_matrix(y_test, test_preds, class_names, cm_path, title=f"{name} — Test Confusion Matrix")

        record = {
            "model": name,
            "split_type": "patient_disjoint",
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "meningioma_f1": metrics["meningioma_f1"],
            "glioma_f1": metrics["glioma_f1"],
            "pituitary_f1": metrics["pituitary_f1"],
            "macro_roc_auc": metrics.get("macro_roc_auc", None)
        }
        results.append(record)

    # Save / Update summary table
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    new_df = pd.DataFrame(results)

    if os.path.exists(summary_path):
        existing_df = pd.read_csv(summary_path)
        combined_df = pd.concat([existing_df[~existing_df["model"].isin(models.keys())], new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
