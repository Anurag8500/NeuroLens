import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from PIL import Image

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import create_model
from src.gradcam import GradCAM, overlay_cam_on_image, compute_cam_mask_hit
from src.data import CLASS_TO_IDX, IDX_TO_CLASS


def main():
    meta_path = "data/metadata.csv"
    split_path = "data/splits/patient_split.json"
    ckpt_path = "outputs/models/resnet50_patient_best.pt"
    images_dir = "data/processed/images"
    masks_dir = "data/processed/masks"
    out_gallery_path = "outputs/figures/gradcam_gallery.png"
    class_names = ["meningioma", "glioma", "pituitary"]

    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Trained model checkpoint {ckpt_path} not found. Train ResNet-50 first.")
        return

    meta = pd.read_csv(meta_path)
    with open(split_path, "r") as f:
        splits = json.load(f)

    test_ids = splits["test"]
    test_meta = meta[meta["slice_id"].isin(test_ids)].copy().reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading ResNet-50 checkpoint from {ckpt_path}...")

    model = create_model("resnet50", num_classes=3, pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Target layer for ResNet-50
    target_layer = model.layer4[-1]
    cam_generator = GradCAM(model, target_layer)

    eval_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Running Grad-CAM explainability on test slices...")
    hits = 0
    total_valid_masks = 0
    gallery_samples = []

    for _, row in test_meta.iterrows():
        sid = int(row["slice_id"])
        true_label = int(row["label"])
        true_cname = row["class_name"]

        img_path = os.path.join(images_dir, f"{sid:04d}.png")
        mask_path = os.path.join(masks_dir, f"{sid:04d}.png")

        raw_pil = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path), dtype=np.uint8) if os.path.exists(mask_path) else None

        input_tensor = eval_tf(raw_pil).unsqueeze(0).to(device)
        cam = cam_generator.generate(input_tensor, target_class=true_label)

        # Predict
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_label = int(np.argmax(probs))

        is_correct = (pred_label == true_label)

        if mask is not None and mask.sum() > 0:
            total_valid_masks += 1
            if compute_cam_mask_hit(cam, mask):
                hits += 1

        # Collect diverse gallery candidates
        if len(gallery_samples) < 9:
            gallery_samples.append({
                "sid": sid,
                "raw_pil": raw_pil,
                "cam": cam,
                "mask": mask,
                "true_cname": true_cname,
                "pred_cname": IDX_TO_CLASS[pred_label],
                "conf": probs[pred_label],
                "is_correct": is_correct
            })

    if total_valid_masks > 0:
        hit_rate = (hits / total_valid_masks) * 100
        print(f"\n[XAI RESULT] Tumor Localization Hit Rate: {hit_rate:.2f}% ({hits}/{total_valid_masks} test slices with peak attention inside ground-truth mask)")

    # Plot 3x3 Grad-CAM Gallery
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for idx, sample in enumerate(gallery_samples[:9]):
        ax = axes[idx]
        overlay = overlay_cam_on_image(sample["raw_pil"], sample["cam"], alpha=0.55)

        # Draw contour of ground-truth tumor mask in white
        if sample["mask"] is not None and sample["mask"].sum() > 0:
            mask_resized = cv2.resize(sample["mask"], (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

        ax.imshow(overlay)
        ax.axis("off")
        status = "CORRECT" if sample["is_correct"] else "ERROR"
        color = "green" if sample["is_correct"] else "red"
        ax.set_title(
            f"True: {sample['true_cname']} | Pred: {sample['pred_cname']} ({sample['conf']:.2f})\n[{status}] (White contour = True Tumor)",
            fontsize=9,
            color=color,
            fontweight="bold"
        )

    plt.suptitle("NeuroLens — Grad-CAM Tumor Localization Gallery (ResNet-50 on Test Patients)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(out_gallery_path, dpi=300)
    plt.close()
    print(f"Saved Grad-CAM gallery figure to: {out_gallery_path}")


if __name__ == "__main__":
    main()
