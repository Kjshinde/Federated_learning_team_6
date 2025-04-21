#!/usr/bin/env python3
"""
split_dataset.py

Cross-platform dataset splitter for federated learning clients.
Unzips a dataset of class-folders and prepares per-client train/test subsets.
"""

from pathlib import Path
import zipfile
import tempfile
import shutil
import random

# === User Configuration ===
ZIP_FILE    = r"C:\Users\kjshi\Downloads\drive-download-20250420T232102Z-001.zip"   # Path to the zipped dataset
CLIENT_ID   = 1                                 # Numeric client ID (used as seed and folder name)
BASE_DIR    = r"flower-fl/client/data"         # Base directory for client data
FRACTION    = 0.2                               # Fraction of images per class to sample (0 < f ≤ 1)
TRAIN_RATIO = 0.8                               # Fraction of sampled images to use for training


def split_for_client(zip_path, client_id, base_client_dir, fraction, train_ratio):
    """
    Extract classes from zip_path, sample `fraction` of images per class,
    split into train/test by `train_ratio`, and copy them into:
    base_client_dir/client_<ID>/{train,test}/{class}/
    """
    zip_path = Path(zip_path)
    base_client_dir = Path(base_client_dir)
    client_dir = base_client_dir / f"client_{client_id}"
    train_root = client_dir / "train"
    test_root  = client_dir / "test"

    # Create train/test directories
    for path in (train_root, test_root):
        path.mkdir(parents=True, exist_ok=True)

    # Extract all data to a temporary folder
    with tempfile.TemporaryDirectory(prefix="fl_dataset_") as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Extract preserving internal structure
            for member in zf.namelist():
                zf.extract(member, tmpdir)

        # Identify class folders at top level
        classes = [p.name for p in tmp_path.iterdir() if p.is_dir()]

        # Seed RNG so each client gets a deterministic but unique split
        random.seed(client_id)

        for cls in classes:
            src_dir = tmp_path / cls
            all_images = [p.name for p in src_dir.iterdir() if p.is_file()]

            # Sample a subset
            k = max(1, int(len(all_images) * fraction))
            sampled = random.sample(all_images, k)

            # Split into train/test
            n_train = max(1, int(len(sampled) * train_ratio))
            train_imgs = sampled[:n_train]
            test_imgs  = sampled[n_train:]

            # Create class-specific train/test dirs
            train_cls_dir = train_root / cls
            test_cls_dir  = test_root / cls
            train_cls_dir.mkdir(parents=True, exist_ok=True)
            test_cls_dir.mkdir(parents=True, exist_ok=True)

            # Copy images
            for fname in train_imgs:
                shutil.copy2(src_dir / fname, train_cls_dir / fname)
            for fname in test_imgs:
                shutil.copy2(src_dir / fname, test_cls_dir / fname)

    print(f"Client {client_id} data created under {client_dir}")


if __name__ == "__main__":
    split_for_client(ZIP_FILE, CLIENT_ID, BASE_DIR, FRACTION, TRAIN_RATIO)
