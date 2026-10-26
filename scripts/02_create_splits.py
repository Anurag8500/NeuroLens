import os
import json
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def make_patient_split(df: pd.DataFrame, seed: int = 42) -> dict:
    """Generate strict patient-disjoint train (70%) / val (15%) / test (15%) splits."""
    y = df["label"].values
    groups = df["pid"].values

    # Step 1: Split into Dev (~85% patients) and Test (~15% patients)
    sgkf_test = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    dev_idx, test_idx = next(iter(sgkf_test.split(df, y, groups)))

    dev_df = df.iloc[dev_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Step 2: Split Dev into Train (~70% of total) and Val (~15% of total)
    sgkf_val = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    dev_y = dev_df["label"].values
    dev_groups = dev_df["pid"].values
    tr_idx, val_idx = next(iter(sgkf_val.split(dev_df, dev_y, dev_groups)))

    train_df = dev_df.iloc[tr_idx]
    val_df = dev_df.iloc[val_idx]

    split_dict = {
        "seed": seed,
        "type": "patient_disjoint",
        "train": train_df["slice_id"].tolist(),
        "val": val_df["slice_id"].tolist(),
        "test": test_df["slice_id"].tolist()
    }
    return split_dict, train_df, val_df, test_df


def make_leaky_split(df: pd.DataFrame, seed: int = 42) -> dict:
    """Generate naive random slice-level train (70%) / val (15%) / test (15%) splits.
    
    WARNING: For Leakage Experiment ONLY. Slices from the same patient are scattered!
    """
    y = df["label"].values

    # Step 1: Split into Dev (~85% slices) and Test (~15% slices)
    skf_test = StratifiedKFold(n_splits=7, shuffle=True, random_state=seed)
    dev_idx, test_idx = next(iter(skf_test.split(df, y)))

    dev_df = df.iloc[dev_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Step 2: Split Dev into Train and Val
    skf_val = StratifiedKFold(n_splits=6, shuffle=True, random_state=seed)
    tr_idx, val_idx = next(iter(skf_val.split(dev_df, dev_df["label"].values)))

    train_df = dev_df.iloc[tr_idx]
    val_df = dev_df.iloc[val_idx]

    return {
        "seed": seed,
        "type": "random_leaky",
        "train": train_df["slice_id"].tolist(),
        "val": val_df["slice_id"].tolist(),
        "test": test_df["slice_id"].tolist()
    }


def main():
    meta_path = "data/metadata.csv"
    splits_dir = "data/splits"
    patient_out = os.path.join(splits_dir, "patient_split.json")
    leaky_out = os.path.join(splits_dir, "leaky_split.json")

    if not os.path.exists(meta_path):
        print(f"[ERROR] {meta_path} not found. Run scripts/01_prepare_data.py first.")
        return

    df = pd.read_csv(meta_path)
    os.makedirs(splits_dir, exist_ok=True)

    # 1. Create Patient-Disjoint Split
    patient_split, tr_df, val_df, te_df = make_patient_split(df, seed=42)

    # Assertions
    tr_patients = set(tr_df["pid"])
    val_patients = set(val_df["pid"])
    te_patients = set(te_df["pid"])

    assert len(tr_patients.intersection(val_patients)) == 0, "Leakage: Train and Val share patients!"
    assert len(tr_patients.intersection(te_patients)) == 0, "Leakage: Train and Test share patients!"
    assert len(val_patients.intersection(te_patients)) == 0, "Leakage: Val and Test share patients!"

    with open(patient_out, "w") as f:
        json.dump(patient_split, f, indent=2)
    print(f"Created patient-disjoint split: {patient_out}")
    print(f"  Train: {len(patient_split['train'])} slices ({len(tr_patients)} patients)")
    print(f"  Val:   {len(patient_split['val'])} slices ({len(val_patients)} patients)")
    print(f"  Test:  {len(patient_split['test'])} slices ({len(te_patients)} patients)")

    # 2. Create Random Leaky Split (for Exp 0)
    leaky_split = make_leaky_split(df, seed=42)
    with open(leaky_out, "w") as f:
        json.dump(leaky_split, f, indent=2)
    print(f"Created random leaky split (for Exp 0): {leaky_out}")
    print(f"  Train: {len(leaky_split['train'])} slices")
    print(f"  Val:   {len(leaky_split['val'])} slices")
    print(f"  Test:  {len(leaky_split['test'])} slices")


if __name__ == "__main__":
    main()
