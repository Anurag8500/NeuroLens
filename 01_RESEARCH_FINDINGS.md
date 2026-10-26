# 01 — Research Findings: Brain Tumor Classification from MRI

---

## 1. Problem Definition & Clinical Context

### 1.1 The Task
- **Input:** Single 2-D MRI slice (T1-weighted contrast-enhanced, grayscale).
- **Output:** Multi-class tumor classification (3 classes):
  1. **Glioma** (1,426 slices in Figshare): Intra-axial tumors arising from glial cells; malignant, infiltrative, heterogeneous enhancement.
  2. **Meningioma** (708 slices in Figshare): Typically benign, extra-axial, dural-based; well-circumscribed, homogeneous enhancement.
  3. **Pituitary Tumor** (930 slices in Figshare): Originates in the sellar/suprasellar region; anatomical location is a strong visual cue.
- **Goal:** Build an honest, robust classification model, prove the impact of patient-level data leakage, and explain model decisions using Grad-CAM.

### 1.2 Clinical Reality Check
Definitive brain tumor diagnosis requires histopathology and molecular profiling. Contrast-enhanced T1-MRI is the diagnostic imaging standard, but image-only 3-class classification is a computer-aided screening tool, **not** a standalone diagnostic device.

---

## 2. Why Most Published "99% Accuracy" Papers Are Flawed

In popular literature and GitHub repositories, models frequently report **97%–99.9% accuracy**. Multiple rigorous medical-AI audits (notably Vangala et al., arXiv:2610.00421; Yagis et al., 2021) have revealed why:

1. **Patient-Level Data Leakage:**
   - The Figshare dataset contains 3,064 slices from only **233 patients** (averaging ~13 slices per patient).
   - In a standard **random slice-level split**, adjacent slices from the *same patient* end up in both training and testing sets.
   - The neural network easily memorizes specific skull shapes, brain anatomy, and idiosyncratic contrast levels of individual patients.
   - Result: 98%+ test accuracy on slices from familiar patients, but drastic failure when evaluated on truly unseen patients.
2. **Honest Patient-Disjoint Benchmark:**
   - When models are evaluated on a strict **patient-level split** (unseen patients only), realistic accuracy drops to **~88%–93%** (e.g., Swin-T benchmark at 93.13% in University of M'sila study).
3. **The Lesson for This Project:**
   - We do not chase artificial 99% accuracy. We implement a strict patient-disjoint split, achieve an honest ~88–93%, and run a direct comparison (Leakage Experiment) demonstrating how naive random splitting inflates accuracy by ~8–10%.

---

## 3. Literature Evolution & Key Insights

1. **Classical Handcrafted ML (Cheng et al., PLOS ONE 2015):**
   - Introduced the Figshare dataset.
   - Used intensity histograms, GLCM (Gray Level Co-occurrence Matrix), and Bag-of-Words (BoW) with SVM.
   - Reported ~91% accuracy, but **only by using dilated, expert-annotated tumor ROI masks**. On full, unsegmented slices without ROI masks, classical ML accuracy is typically ~65%–75%.
2. **Deep CNNs & Transfer Learning (2017–2023):**
   - Pretrained ImageNet backbones (ResNet, DenseNet, EfficientNet) dominate due to the small sample size (233 patients). Training deep networks from scratch on 3,000 medical slices risks severe overfitting.
   - Class-weighted cross-entropy is essential because glioma accounts for ~47% of the dataset.
3. **Model Selection for Our Project:**
   - **ResNet-50:** The universal, battle-tested standard in medical imaging benchmarks.
   - **EfficientNet-B0:** Parameter-efficient, fast to train on 6GB VRAM, modern depthwise-separable architecture.
   - **Simple Custom CNN:** 4-layer conv net from scratch to demonstrate the exact value added by ImageNet transfer learning.
   - **Classical ML Baseline:** HOG + GLCM + intensity statistics $\rightarrow$ Random Forest / SVM.

---

## 4. Evaluation Methodology

- **Primary Metrics:**
  - **Macro-F1:** Unweighted mean of per-class F1-scores; penalizes poor performance on minority classes (meningioma).
  - **Balanced Accuracy:** Macro-averaged recall (1/3 for random chance).
- **Secondary Metrics:**
  - Overall Accuracy, per-class Precision/Recall/F1, Confusion Matrix, One-vs-Rest ROC-AUC.
- **Explainability (Grad-CAM):**
  - Compute Grad-CAM heatmaps on test slices for ResNet-50 and EfficientNet-B0.
  - Overlay heatmaps onto the MRI slices and compare visual peak activations against the ground-truth tumor masks provided in the Figshare metadata.

---

## 5. Key References

- **Cheng et al. (2015):** *Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition.* PLOS ONE 10(10): e0140381. (The original Figshare dataset paper).
- **Vangala et al. (2026):** *Scores That Hold, Benchmarks That Leak: Measuring Dataset Contamination in Public Brain-Tumor MRI Classification.* arXiv:2610.00421. (Documenting patient leakage and benchmark contamination).
- **Yagis et al. (2021):** *Effect of data leakage in brain MRI classification using 2D convolutional neural networks.* Scientific Reports. (Proved slice-level splits inflate brain MRI scores by up to 30%).
- **University of M'sila Thesis (2024):** *Deep Learning Architectures for Brain Tumor MRI Classification: A Comparative Benchmarking Study.* (Strict patient-level benchmark establishing ~93% realistic ceiling).
- **Selvaraju et al. (2017):** *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.* ICCV 2017.

