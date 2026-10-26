# Brain Tumor Classification (MRI) — Streamlined Project Master Brief

> **Core Philosophy:** Keep the machine learning 100% rigorous, scientifically honest, and leak-free, while cutting out unnecessary project bloat (excessive models, duplicate tracks, nested CV complexity). All focus is on clean, reproducible ML and the central finding: **patient-level data leakage**.

| File | Purpose |
|---|---|
| `00_README_START_HERE.md` | Project summary, simplified scope, non-negotiable rules, definition of done |
| `01_RESEARCH_FINDINGS.md` | Clinical background, why 99% accuracy papers are flawed, literature references |
| `02_DATASETS.md` | Figshare dataset structure, HDF5 loading, preprocessing, patient split design |
| `03_WORKFLOW_AND_IMPLEMENTATION.md` | Lean 8-phase implementation plan, models, metrics, leakage experiment, Grad-CAM |

---

## 1. What We Are Building (Streamlined Scope)

**Task:** Given a 2-D brain MRI slice (T1-weighted contrast-enhanced), classify the tumor type:
- **3 Classes:** `meningioma` (708 slices), `glioma` (1,426 slices), `pituitary tumor` (930 slices).
- **Dataset:** Jun Cheng / Figshare MRI dataset (3,064 slices across **233 patients**).
- **Core Scientific Contribution:** 
  1. A strict **patient-disjoint split** (zero slice leakage between patients across train/val/test).
  2. **Leakage Demonstration Experiment:** Compare the patient-disjoint split (honest ~88–93%) against the naive random slice split (inflated ~98–99%) to prove why so many papers report artificially high numbers.
  3. **Explainability (Grad-CAM):** Verify whether the model actually looks at the tumor or at irrelevant background shortcuts using ground-truth tumor masks.

### What We Streamlined Out (To Keep It Manageable & Realistic)
- **Dropped secondary Track B (Kaggle 4-class):** Avoids managing two separate datasets, cross-dataset containment checks, and duplicate-graph clustering. Figshare has genuine patient IDs and verified masks; it is all we need.
- **Dropped model bloat (from 7 backbones to 2 strong backbones + 2 baselines):**
  - Baseline 0: Classical ML (HOG/texture features + Random Forest or SVM).
  - Baseline 1: Simple custom CNN (from scratch).
  - Deep Models: **ResNet-50** (established clinical benchmark) and **EfficientNet-B0** (fast, parameter-efficient, modern).
- **Streamlined split structure:** Instead of nested 5-fold CV with inner 6-fold validation (which required 35+ DL training runs), use a clean, rock-solid **Patient-Disjoint Train (70%) / Val (15%) / Test (15%)** split. Fast to train, clean to evaluate, zero leakage.

---

## 2. The Core Finding: Patient-Level Data Leakage

Almost every popular tutorial and many published papers claim **98–99.9% accuracy**. These claims are flawed because:
1. **Patient-level leakage:** 3,064 slices come from only 233 patients (~13 slices per patient). Random splitting scatters slices of the *same tumor and patient* across both train and test. The model memorizes patient anatomy rather than tumor pathology.
2. In strict patient-disjoint benchmarks, the true generalization accuracy is **~88–93%**.
3. **If our patient-disjoint model gets 99%, we did something wrong.**

---

## 3. Simplified Project Architecture

```
Figshare .mat (3,064 slices, 233 patients)
   │
   ├──► Data Extraction (h5py, transpose, intensity normalization)
   │
   ├──► Splits Generation (Saved once to data/splits/*.json):
   │     ├── (A) Patient-Disjoint Split (70% Train / 15% Val / 15% Test grouped by patient_id)
   │     └── (B) Random Leaky Split (70% Train / 15% Val / 15% Test split by slice)
   │
   ├──► Baseline 0: Classical ML (HOG + GLCM + stats → Random Forest / SVM)
   ├──► Baseline 1: Simple CNN (trained from scratch)
   ├──► Transfer Learning: ResNet-50 & EfficientNet-B0 (fine-tuned with class weights)
   │
   ├──► Key Experiment: Patient-Disjoint vs. Leaky Split (quantify the ~8–10% inflation gap)
   ├──► Final Evaluation: Macro-F1, Balanced Accuracy, Confusion Matrix, ROC-AUC
   └──► Explainability: Grad-CAM heatmap overlays against ground-truth tumor masks
```

---

## 4. Non-Negotiable Rules

1. **Never split by image in the main pipeline:** Group by `patient_id` so that all slices of any patient belong strictly to one partition.
2. **Lock splits to disk:** Generate `data/splits/patient_split.json` and `data/splits/leaky_split.json` once. Never re-seed or regenerate them silently.
3. **Fit on training data only:** All scalers, normalizers, and class weights are computed from the training split only.
4. **HDF5 `.mat` reading:** Read with `h5py` and **always transpose (`.T`)** both image and mask.
5. **Primary metrics:** Macro-F1 and Balanced Accuracy (due to class imbalance: glioma ~47%). Accuracy is secondary.
6. **Hardware friendly:** Designed to run comfortably on an RTX 3050 (6GB VRAM) or standard Google Colab T4 GPU in under an hour.
7. **Research disclaimer:** Strictly an educational/research benchmark, not a clinical device.

---

## 5. Definition of Done

- [ ] Clean dataset conversion script from raw `.mat` to preprocessed PNGs and metadata CSV.
- [ ] Verified patient-disjoint and leaky split JSON files committed to the repo.
- [ ] Classical ML baseline (HOG + texture) evaluated.
- [ ] Deep transfer learning models (ResNet-50 & EfficientNet-B0) trained and evaluated.
- [ ] Leakage Experiment showing the direct performance gap between leaky vs. patient-disjoint splits.
- [ ] Grad-CAM heatmaps showing model attention on tumor regions vs. ground-truth masks.
- [ ] Clean evaluation summary (Confusion matrices, Macro-F1, Balanced Accuracy, per-class metrics).
- [ ] Concise final project report and CLI inference script (`predict.py`).

