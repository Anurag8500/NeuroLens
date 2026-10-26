# 02 — Dataset: Sources, Formats, Preprocessing & Splits

---

## 1. Primary Dataset: Figshare / Cheng Brain Tumor Dataset

### 1.1 Source & Overview
- **Landing Page:** https://figshare.com/articles/dataset/brain_tumor_dataset/1512427 (DOI: `10.6084/m9.figshare.1512427`).
- **Citation:** Cheng, Jun et al. (2015). *Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition.* PLOS ONE 10(10): e0140381.
- **Contents:**
  - **3,064 T1-weighted contrast-enhanced MRI slices** from **233 patients**.
  - **3 Classes:**
    - `meningioma`: 708 slices
    - `glioma`: 1,426 slices
    - `pituitary tumor`: 930 slices
  - **Planes:** Axial (994), Coronal (1,045), Sagittal (1,025).
  - **Format:** 4 zip files containing MATLAB `.mat` files named `1.mat` through `3064.mat`.

### 1.2 File Format & Loading Gotchas
Each `.mat` file is saved in **MATLAB v7.3 (HDF5 format)** and contains a struct named `cjdata`:
- `cjdata.label`: 1 = meningioma, 2 = glioma, 3 = pituitary tumor (map to 0, 1, 2).
- `cjdata.PID`: Patient ID string (stored as uint16 ASCII character array).
- `cjdata.image`: 2-D MRI slice (typically $512 \times 512$ int16 array).
- `cjdata.tumorBorder`: Contour coordinates of the tumor boundary.
- `cjdata.tumorMask`: Binary mask ($512 \times 512$ uint8, 1 = tumor).

> [!CAUTION]
> **HDF5 Column-Major Transpose Gotcha:**
> HDF5 matrices in Python are read in column-major order. You **MUST transpose (`.T`)** both `image` and `tumorMask`. Forgetting `.T` results in rotated/mirrored images.

### 1.3 Reference Python Loader
```python
import h5py
import numpy as np

LABEL_MAP = {1: "meningioma", 2: "glioma", 3: "pituitary"}
CLASS_TO_IDX = {"meningioma": 0, "glioma": 1, "pituitary": 2}

def load_slice(mat_path: str) -> dict:
    with h5py.File(mat_path, "r") as f:
        data = f["cjdata"]
        label = int(np.array(data["label"]).flatten()[0])
        pid_chars = np.array(data["PID"]).flatten()
        pid = "".join(chr(int(c)) for c in pid_chars)
        image = np.array(data["image"]).T               # Transpose is mandatory
        mask = np.array(data["tumorMask"]).T.astype(np.uint8)
    return {
        "label": label - 1,
        "class_name": LABEL_MAP[label],
        "pid": pid,
        "image": image,
        "mask": mask
    }
```

---

## 2. Preprocessing Pipeline

1. **Intensity Normalization:**
   - Raw MRI intensities are not calibrated 0–255.
   - For each slice: clip pixel intensities to $[1^{\text{st}}, 99.5^{\text{th}}]$ percentiles.
   - Scale linearly to $[0, 1]$, then convert to 8-bit $[0, 255]$ for standard vision pipelines.
2. **Resizing:**
   - Resize images to $256 \times 256$ using bilinear interpolation.
   - Resize masks to $256 \times 256$ using nearest-neighbor interpolation.
3. **Storage:**
   - Save processed images as PNGs under `data/processed/images/{slice_id:04d}.png`.
   - Save masks as PNGs under `data/processed/masks/{slice_id:04d}.png`.
   - Save extracted metadata to `data/metadata.csv`.

### Mandatory Metadata Assertions
```python
assert len(df) == 3064, f"Expected 3064 slices, got {len(df)}"
assert df["pid"].nunique() == 233, f"Expected 233 patients, got {df['pid'].nunique()}"
assert df["class_name"].value_counts().to_dict() == {
    "glioma": 1426,
    "pituitary": 930,
    "meningioma": 708
}
# Each patient has exactly one diagnosed tumor type
assert (df.groupby("pid")["label"].nunique() == 1).all()
```

---

## 3. Streamlined Split Strategy

We generate **two split files** once, save them to `data/splits/`, and commit them.

### Split A: Strict Patient-Disjoint Split (Primary Benchmark)
- **Train (70% of patients, ~163 patients):** Used to train model weights.
- **Val (15% of patients, ~35 patients):** Used for early stopping, learning-rate scheduling, and hyperparameter tuning.
- **Test (15% of patients, ~35 patients):** Locked test set; evaluated **only once** at the very end.
- **Rule:** Zero patient overlap between partitions. If patient $P_{123}$ is in Test, **none** of their slices appear in Train or Val.
- Saved to: `data/splits/patient_split.json`.

```python
from sklearn.model_selection import StratifiedGroupKFold

def make_patient_split(df, seed=42):
    # Split 85% dev / 15% test
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    dev_idx, test_idx = next(iter(sgkf.split(df, df.label, df.pid)))
    dev_df = df.iloc[dev_idx].reset_index(drop=True)
    
    # Split dev into train (~70% total) and val (~15% total)
    sgkf_inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    tr_idx, val_idx = next(iter(sgkf_inner.split(dev_df, dev_df.label, dev_df.pid)))
    
    return {
        "train": dev_df.iloc[tr_idx]["slice_id"].tolist(),
        "val": dev_df.iloc[val_idx]["slice_id"].tolist(),
        "test": df.iloc[test_idx]["slice_id"].tolist()
    }
```

### Split B: Random Slice Split (Leakage Experiment)
- 70% Train / 15% Val / 15% Test sampled **purely at the slice level** (`StratifiedKFold`), ignoring `patient_id`.
- Slices of the same patient are scattered across train, val, and test.
- Saved to: `data/splits/leaky_split.json`.
- Used strictly in **Phase 5 (Leakage Experiment)** to quantify the artificial performance inflation.

