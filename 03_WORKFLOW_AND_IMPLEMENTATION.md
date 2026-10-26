# 03 — Streamlined Workflow & Implementation Plan

---

## 0. Clean Repository Layout

```
NeuroLens/
├── 00_README_START_HERE.md
├── 01_RESEARCH_FINDINGS.md
├── 02_DATASETS.md
├── 03_WORKFLOW_AND_IMPLEMENTATION.md
├── requirements.txt
├── configs/
│   └── config.yaml                     # Single consolidated YAML configuration
├── data/
│   ├── raw/figshare_mat/               # 1.mat ... 3064.mat (downloaded once)
│   ├── processed/
│   │   ├── images/                     # 256x256 normalized PNGs
│   │   └── masks/                      # 256x256 binary mask PNGs
│   ├── metadata.csv                    # 3064 rows, slice_id, pid, label, etc.
│   └── splits/
│       ├── patient_split.json          # 70/15/15 strictly patient-disjoint
│       └── leaky_split.json            # 70/15/15 random slice split (for Exp 0)
├── src/
│   ├── __init__.py
│   ├── data.py                         # Dataset class, DataLoader, transforms
│   ├── features.py                     # HOG, GLCM, intensity stats extraction
│   ├── models.py                       # SimpleCNN, ResNet-50, EfficientNet-B0
│   ├── train.py                        # Training engine (AMP, class weights, early stop)
│   ├── evaluate.py                     # Macro-F1, balanced acc, confusion matrix, ROC
│   └── gradcam.py                      # Grad-CAM heatmap computation & overlay
├── scripts/
│   ├── 01_prepare_data.py              # Convert .mat -> PNGs + metadata.csv
│   ├── 02_create_splits.py             # Generate & verify patient and leaky splits
│   ├── 03_train_classical.py           # Train HOG/GLCM -> Random Forest & SVM
│   ├── 04_train_dl.py                  # Train SimpleCNN, ResNet-50, EfficientNet-B0
│   ├── 05_leakage_experiment.py        # Train on leaky split and quantify inflation gap
│   ├── 06_evaluate.py                  # Generate test evaluation tables & ROC curves
│   └── 07_gradcam.py                   # Generate Grad-CAM overlays vs. tumor masks
├── predict.py                          # CLI inference for a single MRI slice
├── outputs/
│   ├── models/                         # Saved weights (*.pt)
│   ├── figures/                        # Plots, confusion matrices, Grad-CAM overlays
│   └── results_summary.csv             # Final metric comparison table
└── tests/
    └── test_splits.py                  # Strict patient-disjoint assertion test
```

---

## Phase 0 — Environment & Setup

**Goal:** Working Python environment and project directories.
**Tasks:**
1. Create virtual environment with Python 3.12:
   ```bash
   py -3.12 -m venv .venv
   .venv\Scripts\activate
   ```
2. Install pinned dependencies in `requirements.txt`:
   - `torch`, `torchvision` (CUDA-enabled)
   - `timm` (pretrained backbones)
   - `h5py` (HDF5 reading for MATLAB v7.3)
   - `scikit-learn`, `scikit-image` (GLCM, HOG, metrics)
   - `opencv-python`, `pillow`, `matplotlib`, `seaborn`
   - `pyyaml`, `tqdm`, `pytest`
3. Verify GPU availability:
   `python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
4. Create base configuration `configs/config.yaml`.

---

## Phase 1 — Data Ingestion & Conversion

**Goal:** Standardized, verified dataset with orientation checks.
**Tasks (`scripts/01_prepare_data.py`):**
1. Extract `.mat` files from `data/raw/figshare_mat/`.
2. For each `.mat` file:
   - Read with `h5py`.
   - Transpose (`.T`) both image and tumor mask.
   - Decode `PID` character string.
   - Intensity normalize: clip $[p_1, p_{99.5}]$, scale to $[0, 255]$ uint8.
   - Resize to $256 \times 256$.
   - Save processed image to `data/processed/images/{slice_id:04d}.png`.
   - Save processed mask to `data/processed/masks/{slice_id:04d}.png`.
3. Save `data/metadata.csv`.
4. Generate `outputs/figures/orientation_check.png` showing 6 sample slices per class to visually confirm skull is upright and not mirrored.

**Acceptance Check:**
- 3,064 PNG files in `images/` and `masks/`.
- `metadata.csv` passes all assertions (3,064 rows, 233 unique PIDs, class counts: 1426 glioma, 930 pituitary, 708 meningioma).

---

## Phase 2 — Split Generation & Leakage Verification

**Goal:** Immutable, verified split files.
**Tasks (`scripts/02_create_splits.py`):**
1. Generate **Patient-Disjoint Split** (70% train / 15% val / 15% test grouped by `pid`).
   - Save to `data/splits/patient_split.json`.
2. Generate **Random Leaky Split** (70% train / 15% val / 15% test split purely by slice).
   - Save to `data/splits/leaky_split.json`.
3. Run `tests/test_splits.py`:
   - Assert zero patient overlap between train, val, and test in `patient_split.json`.
   - Verify every class is represented in all three partitions.

---

## Phase 3 — Classical ML Baseline

**Goal:** Floor performance benchmark using handcrafted feature extraction.
**Tasks (`scripts/03_train_classical.py`):**
1. Extract feature vector per slice ($256 \times 256$):
   - **Intensity Statistics:** Mean, std, skewness, kurtosis, 32-bin histogram.
   - **GLCM Texture:** Contrast, dissimilarity, homogeneity, energy, correlation (angles: 0, 45, 90, 135).
   - **HOG:** Histogram of Oriented Gradients (9 orientations, $16 \times 16$ cell size).
2. Fit `StandardScaler` strictly on the training partition.
3. Train two classical classifiers:
   - **Random Forest** (100 trees)
   - **Support Vector Machine** (RBF kernel)
4. Evaluate on the patient-disjoint test set.
5. Log Macro-F1, Balanced Accuracy, and overall Accuracy. (Expected range: ~65%–75%).

---

## Phase 4 — Deep Transfer Learning (Main Models)

**Goal:** High-performance, honest patient-level classification.
**Tasks (`scripts/04_train_dl.py`):**
1. Implement PyTorch `SliceDataset` with data augmentation:
   - Train: Random resized crop (224), random horizontal flip, rotation ($\pm 10^\circ$), color jitter.
   - Val/Test: Resize to 256, center crop to 224, ImageNet normalization.
2. Models evaluated:
   - **SimpleCNN:** 4 conv-batchnorm-relu blocks trained from scratch (to establish baseline deep learning floor).
   - **ResNet-50:** Pretrained on ImageNet.
   - **EfficientNet-B0:** Pretrained on ImageNet (fast, lightweight, highly accurate).
3. Training specifications:
   - **Loss:** Class-weighted Cross-Entropy (weights inversely proportional to class frequencies in the training split).
   - **Optimizer:** AdamW with cosine annealing learning rate schedule.
   - **Precision:** Mixed precision (`torch.cuda.amp.autocast`).
   - **Early stopping:** Monitor validation Macro-F1 with patience = 7 epochs.
4. Save best checkpoints to `outputs/models/`.

---

## Phase 5 — The Leakage Demonstration Experiment

**Goal:** Experimentally prove how naive random splitting inflates scores by 8–10%.
**Tasks (`scripts/05_leakage_experiment.py`):**
1. Train **ResNet-50** on `data/splits/leaky_split.json` using the **exact same** hyperparameters and seed as Phase 4.
2. Evaluate on the leaky test set.
3. Compare side-by-side with the patient-disjoint ResNet-50:
   - Patient-Disjoint ResNet-50: Honest generalization (~88–93%).
   - Leaky ResNet-50: Artificially inflated (~97–99%).
4. Generate `outputs/figures/leakage_inflation_gap.png` (a clean bar chart showing the gap in Accuracy, Macro-F1, and Balanced Accuracy).

---

## Phase 6 — Comprehensive Evaluation & Metrics

**Goal:** Clear, publication-grade figures and metrics on the locked test set.
**Tasks (`scripts/06_evaluate.py`):**
1. Load test predictions for:
   - Classical ML (Random Forest / SVM)
   - SimpleCNN (scratch)
   - ResNet-50 (patient-disjoint)
   - EfficientNet-B0 (patient-disjoint)
   - ResNet-50 (leaky split)
2. Generate summary table `outputs/results_summary.csv`:
   - Columns: `Model`, `Split Type`, `Accuracy`, `Macro-F1`, `Balanced Accuracy`, `Glioma F1`, `Meningioma F1`, `Pituitary F1`.
3. Generate visualization artifacts:
   - `outputs/figures/confusion_matrices.png`: Side-by-side row-normalized confusion matrices.
   - `outputs/figures/roc_curves.png`: Multiclass One-vs-Rest ROC curves with AUC values.

---

## Phase 7 — Explainability (Grad-CAM)

**Goal:** Verify whether the deep models attend to real tumor pathology or non-anatomical artefacts.
**Tasks (`scripts/07_gradcam.py`):**
1. Hook into the final convolutional layer:
   - ResNet-50: `layer4[-1]`
   - EfficientNet-B0: `conv_head`
2. Generate Grad-CAM heatmaps for test slices across all three classes.
3. Overlay heatmaps with ground-truth tumor contours (from `data/processed/masks/`).
4. Generate `outputs/figures/gradcam_gallery.png`:
   - 3 correct predictions per class showing tumor localization.
   - 2 misclassified predictions showing where the attention drifted.
5. Compute **Tumor Localization Hit Rate**: Percentage of test slices where the peak Grad-CAM activation pixel falls inside the ground-truth tumor mask.

---

## Phase 8 — CLI Inference & Final Report

**Goal:** Deliverable tool and concise summary.
**Tasks:**
1. Build `predict.py`:
   ```bash
   python predict.py --image path/to/mri_slice.png --model outputs/models/resnet50_best.pt
   ```
   Outputs:
   - Predicted class name + softmax confidence scores for all 3 classes.
   - Saves an optional Grad-CAM overlay image.
   - Prints clear research-only disclaimer.
2. Compile results into `outputs/FINAL_PROJECT_REPORT.md` documenting:
   - Motivation and clinical background.
   - The Leakage Gap proof.
   - Model comparison table.
   - Grad-CAM findings.
   - Limitations and honest conclusions.

---

## Key Hyperparameter Defaults (`configs/config.yaml`)

```yaml
seed: 42

data:
  raw_dir: "data/raw/figshare_mat"
  processed_img_dir: "data/processed/images"
  processed_mask_dir: "data/processed/masks"
  metadata_path: "data/metadata.csv"
  splits_dir: "data/splits"
  image_size: 224
  resize_canvas: 256

training:
  batch_size: 32
  num_workers: 2
  epochs: 25
  early_stopping_patience: 7
  lr: 1.0e-4
  weight_decay: 0.01
  amp: true

classical:
  rf_trees: 100
  svm_c: 1.0
```
