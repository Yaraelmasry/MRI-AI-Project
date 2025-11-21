"""
Loads brain MRI images from the Kaggle dataset folders and converts them
into numpy arrays that we can feed into a scikit-learn model.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image

# Map each folder name (class) to a numeric label
CLASS_MAP = {
    "glioma_tumor": 0,
    "meningioma_tumor": 1,
    "pituitary_tumor": 2,
}

# All images will be resized to this size (width, height)
IMG_SIZE = (64, 64)


def _load_single_image(img_path: Path) -> np.ndarray:
    """
    Load a single image, convert it to grayscale, resize and flatten it.

    Returns a 1D numpy array of shape (64*64,) with values in [0,1].
    """
    img = Image.open(img_path).convert("L")  #greyscale
    img = img.resize(IMG_SIZE)
    # normalize to [0,1]
    arr = np.array(img, dtype=np.float32) / 255.0  
    return arr.flatten()


def load_dataset(root_dir: str):
    """
    Load all images
"""
    root = Path(root_dir)
    X = []
    y = []

    for class_name, label in CLASS_MAP.items():
        class_dir = root / class_name
        if not class_dir.exists():
            print(f"[WARN] Folder not found: {class_dir} (skipping)")
            continue

        # Accept all image exceptions
        img_paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            img_paths.extend(class_dir.glob(ext))

        if not img_paths:
            print(f"[WARN] No images found in {class_dir}")
            continue

        for img_path in img_paths:
            try:
                feat = _load_single_image(img_path)
                X.append(feat)
                y.append(label)
            except Exception as e:
                print(f"[ERROR] Failed to load {img_path}: {e}")

    X = np.array(X)
    y = np.array(y)

    print(f"[INFO] Loaded {X.shape[0]} samples from {root_dir}")
    return X, y


def preprocess_single_image(image_path: str) -> np.ndarray:
    """
    Same preprocessing as training, but for one image.
    Used during prediction.

    Returns:
        X: numpy array of shape (1, n_features)
    """
    img_path = Path(image_path)
    feat = _load_single_image(img_path)
    return feat.reshape(1, -1)


if __name__ == "__main__":
    X_train, y_train = load_dataset("data/Training")
    print("Train X shape:", X_train.shape)
    print("Train y shape:", y_train.shape)
