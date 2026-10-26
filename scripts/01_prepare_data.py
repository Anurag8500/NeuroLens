import os
import glob
import sys
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import load_raw_mat, normalize_intensity, LABEL_MAP


def main():
    raw_dir = "data/raw/figshare_mat"
    img_out_dir = "data/processed/images"
    mask_out_dir = "data/processed/masks"
    meta_path = "data/metadata.csv"
    fig_path = "outputs/figures/orientation_check.png"

    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    # Search for .mat files
    mat_files = sorted(glob.glob(os.path.join(raw_dir, "*.mat")), key=lambda p: int(os.path.splitext(os.path.basename(p))[0]) if os.path.splitext(os.path.basename(p))[0].isdigit() else 999999)

    # Exclude cvind.mat if present
    mat_files = [f for f in mat_files if os.path.basename(f) != "cvind.mat"]

    if len(mat_files) == 0:
        print(f"\n[ERROR] No .mat files found in {raw_dir}!")
        print("\nPlease download the 4 Figshare zip files from:")
        print("https://figshare.com/articles/dataset/brain_tumor_dataset/1512427")
        print("Extract all .mat files (1.mat ... 3064.mat) into: data/raw/figshare_mat/\n")
        return

    print(f"Found {len(mat_files)} .mat files. Processing into 256x256 normalized PNGs...")

    records = []
    sample_slices_by_class = {0: [], 1: [], 2: []}

    for path in tqdm(mat_files, desc="Converting slices"):
        slice_id = int(os.path.splitext(os.path.basename(path))[0])
        data = load_raw_mat(path)

        raw_img = data["image"]
        raw_mask = data["mask"]
        label = data["label"]
        cname = data["class_name"]
        pid = data["pid"]

        # Compute raw stats
        p1, p995 = np.percentile(raw_img, [1.0, 99.5])
        norm_img = normalize_intensity(raw_img, p1=1.0, p995=99.5)

        # Resize to 256x256
        pil_img = Image.fromarray(norm_img).resize((256, 256), Image.Resampling.BILINEAR)
        pil_mask = Image.fromarray(raw_mask).resize((256, 256), Image.Resampling.NEAREST)

        # Save PNGs
        img_file = os.path.join(img_out_dir, f"{slice_id:04d}.png")
        mask_file = os.path.join(mask_out_dir, f"{slice_id:04d}.png")
        pil_img.save(img_file)
        pil_mask.save(mask_file)

        mask_area = int(np.array(pil_mask).sum())

        records.append({
            "slice_id": slice_id,
            "pid": pid,
            "label": label,
            "class_name": cname,
            "orig_h": raw_img.shape[0],
            "orig_w": raw_img.shape[1],
            "raw_min": int(raw_img.min()),
            "raw_max": int(raw_img.max()),
            "p1": float(p1),
            "p995": float(p995),
            "mask_area": mask_area
        })

        if len(sample_slices_by_class[label]) < 6:
            sample_slices_by_class[label].append(pil_img)

    df = pd.DataFrame(records).sort_values("slice_id").reset_index(drop=True)

    # Mandatory Assertions
    print("\nRunning integrity assertions...")
    assert len(df) == 3064, f"Expected 3064 slices, got {len(df)}"
    assert df["pid"].nunique() == 233, f"Expected 233 patients, got {df['pid'].nunique()}"
    class_counts = df["class_name"].value_counts().to_dict()
    assert class_counts == {"glioma": 1426, "pituitary": 930, "meningioma": 708}, f"Class counts mismatch: {class_counts}"
    assert (df.groupby("pid")["label"].nunique() == 1).all(), "Some patients have multiple tumor labels!"

    df.to_csv(meta_path, index=False)
    print(f"Saved {meta_path} successfully.")
    print("Class distribution:")
    for k, v in class_counts.items():
        print(f"  {k}: {v} slices")

    # Generate orientation sanity check plot
    print("Generating orientation sanity check figure...")
    fig, axes = plt.subplots(3, 6, figsize=(14, 7))
    class_names = ["meningioma", "glioma", "pituitary"]

    for row_idx, c_idx in enumerate([0, 1, 2]):
        for col_idx, img in enumerate(sample_slices_by_class[c_idx]):
            ax = axes[row_idx, col_idx]
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            if col_idx == 0:
                ax.set_title(class_names[c_idx], fontsize=12, loc="left", fontweight="bold")

    plt.suptitle("NeuroLens — Figshare MRI Orientation & Quality Sanity Check", fontsize=14)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Sanity orientation figure saved to: {fig_path}")


if __name__ == "__main__":
    main()
