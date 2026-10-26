import json
import os
import pandas as pd
import pytest


def test_patient_split_disjoint():
    """Verify that patient-disjoint split has zero patient leakage across partitions."""
    split_path = "data/splits/patient_split.json"
    metadata_path = "data/metadata.csv"

    if not os.path.exists(split_path) or not os.path.exists(metadata_path):
        pytest.skip("Splits or metadata not generated yet.")

    with open(split_path, "r") as f:
        splits = json.load(f)

    meta = pd.read_csv(metadata_path)

    train_ids = set(splits["train"])
    val_ids = set(splits["val"])
    test_ids = set(splits["test"])

    # 1. No slice overlap
    assert len(train_ids.intersection(val_ids)) == 0, "Train and Val share slice IDs!"
    assert len(train_ids.intersection(test_ids)) == 0, "Train and Test share slice IDs!"
    assert len(val_ids.intersection(test_ids)) == 0, "Val and Test share slice IDs!"

    # 2. Extract patient sets
    train_patients = set(meta[meta["slice_id"].isin(train_ids)]["pid"])
    val_patients = set(meta[meta["slice_id"].isin(val_ids)]["pid"])
    test_patients = set(meta[meta["slice_id"].isin(test_ids)]["pid"])

    # 3. Strict patient disjointness (NO PATIENT LEAKAGE)
    assert len(train_patients.intersection(val_patients)) == 0, "Patient leakage between Train and Val!"
    assert len(train_patients.intersection(test_patients)) == 0, "Patient leakage between Train and Test!"
    assert len(val_patients.intersection(test_patients)) == 0, "Patient leakage between Val and Test!"

    # 4. All 3 classes present in each split
    for name, s_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        sub = meta[meta["slice_id"].isin(s_ids)]
        unique_classes = set(sub["class_name"])
        assert len(unique_classes) == 3, f"{name} split is missing classes! Found: {unique_classes}"

    print("\n[SUCCESS] Strict patient-disjointness verified. Zero patient leakage detected.")
