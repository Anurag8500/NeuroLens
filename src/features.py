import numpy as np
from skimage.feature import graycomatrix, graycoprops, hog
from scipy.stats import skew, kurtosis
from PIL import Image


def extract_slice_features(img_path: str) -> np.ndarray:
    """Extract a rich handcrafted feature vector from a single 256x256 grayscale MRI image.
    
    Includes:
    1. First-order intensity statistics & histogram (36 features)
    2. Gray Level Co-occurrence Matrix (GLCM) texture metrics (20 features)
    3. Histogram of Oriented Gradients (HOG) (downsampled to 128x128)
    """
    img = Image.open(img_path).convert("L")
    arr = np.array(img, dtype=np.float32)

    # 1. First-order intensity statistics
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    skew_val = float(skew(arr.ravel()))
    kurt_val = float(kurtosis(arr.ravel()))
    
    hist, _ = np.histogram(arr, bins=32, range=(0, 256), density=True)
    first_order = np.array([mean_val, std_val, skew_val, kurt_val] + hist.tolist(), dtype=np.float32)

    # 2. GLCM Texture features
    # Quantize to 16 gray levels for robust co-occurrence
    arr_16 = (arr / 16).astype(np.uint8)
    glcm = graycomatrix(
        arr_16,
        distances=[1, 3],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=16,
        symmetric=True,
        normed=True
    )
    
    glcm_props = []
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        vals = graycoprops(glcm, prop).ravel()
        glcm_props.extend([vals.mean(), vals.std()])
    glcm_features = np.array(glcm_props, dtype=np.float32)

    # 3. HOG features (on 128x128 resize to keep feature length compact)
    img_128 = img.resize((128, 128), Image.Resampling.BILINEAR)
    arr_128 = np.array(img_128, dtype=np.float32)
    hog_features = hog(
        arr_128,
        orientations=8,
        pixels_per_cell=(32, 32),
        cells_per_block=(2, 2),
        visualize=False,
        channel_axis=None
    ).astype(np.float32)

    return np.concatenate([first_order, glcm_features, hog_features])
