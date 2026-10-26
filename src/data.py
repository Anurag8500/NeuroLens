import os
import h5py
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

LABEL_MAP = {1: "meningioma", 2: "glioma", 3: "pituitary"}
CLASS_TO_IDX = {"meningioma": 0, "glioma": 1, "pituitary": 2}
IDX_TO_CLASS = {0: "meningioma", 1: "glioma", 2: "pituitary"}


def load_raw_mat(mat_path: str) -> dict:
    """Read one Figshare v7.3 .mat (HDF5) file.
    
    Mandatory: Transpose (.T) both image and mask because HDF5 reads column-major.
    """
    with h5py.File(mat_path, "r") as f:
        data = f["cjdata"]
        raw_label = int(np.array(data["label"]).flatten()[0])
        pid_chars = np.array(data["PID"]).flatten()
        pid = "".join(chr(int(c)) for c in pid_chars)
        image = np.array(data["image"]).T
        mask = np.array(data["tumorMask"]).T.astype(np.uint8)
        
    return {
        "label": raw_label - 1,  # 0: meningioma, 1: glioma, 2: pituitary
        "class_name": LABEL_MAP[raw_label],
        "pid": pid,
        "image": image,
        "mask": mask
    }


def normalize_intensity(image: np.ndarray, p1: float = 1.0, p995: float = 99.5) -> np.ndarray:
    """Percentile clip and scale raw MRI intensities to [0, 255] uint8."""
    image_f = image.astype(np.float32)
    low, high = np.percentile(image_f, [p1, p995])
    if high <= low:
        high = low + 1e-5
    clipped = np.clip(image_f, low, high)
    normalized = (clipped - low) / (high - low)
    return (normalized * 255.0).astype(np.uint8)


def get_train_transforms(image_size: int = 224):
    """Training augmentation with light spatial transforms suitable for brain MRI."""
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_eval_transforms(image_size: int = 224, resize_canvas: int = 256):
    """Deterministic validation and test transforms."""
    return T.Compose([
        T.Resize(resize_canvas),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class SliceDataset(Dataset):
    """PyTorch Dataset loading preprocessed MRI slice PNGs."""

    def __init__(self, metadata_df: pd.DataFrame, slice_ids: list, images_dir: str, masks_dir: str = None, transform=None):
        self.metadata = metadata_df[metadata_df["slice_id"].isin(slice_ids)].copy().reset_index(drop=True)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        slice_id = int(row["slice_id"])
        label = int(row["label"])
        pid = str(row["pid"])

        img_path = os.path.join(self.images_dir, f"{slice_id:04d}.png")
        # Load grayscale and convert to 3-channel RGB for ImageNet models
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = T.ToTensor()(img)

        sample = {
            "image": img_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "slice_id": slice_id,
            "pid": pid,
            "class_name": row["class_name"]
        }

        if self.masks_dir:
            mask_path = os.path.join(self.masks_dir, f"{slice_id:04d}.png")
            if os.path.exists(mask_path):
                mask = Image.open(mask_path)
                sample["mask"] = np.array(mask, dtype=np.uint8)

        return sample
