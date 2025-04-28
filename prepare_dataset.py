#!/usr/bin/env python3
"""
prepare_dataset.py

Cross-platform dataset splitter for federated learning clients.
Unzips a dataset of class-folders and prepares per-client train/test subsets.
Handles zips with a single top-level directory or direct class folders,
and adds images to existing directories without overwriting.
"""

import sys
from pathlib import Path
import zipfile
import tempfile
import shutil
import random
import os

# === User Configuration ===
ZIP_FILE    = Path(r'C:\Users\kjshi\Desktop\ASU_sem_2\Intro_to_ml_with_fpga\Project\Federated_learning_team_6\ML dataset-20250426T225000Z-001.zip')  # Path to your zipped dataset
CLIENT_ID   = 2                                # Unique numeric client ID
BASE_DIR    = Path('flower-fl') / 'client' / 'data'  # Base directory for client data
FRACTION    = 0.2                              # Fraction of images per class to sample (0 < f ≤ 1)
TRAIN_RATIO = 0.8                              # Fraction of sampled images for training

# Sanity check for the ZIP file
print(f"ZIP_FILE: {ZIP_FILE}")
if not ZIP_FILE.is_file():
    print(f"ERROR: ZIP file not found at {ZIP_FILE}", file=sys.stderr)
    sys.exit(1)

# Ensure base data directory exists
BASE_DIR.mkdir(parents=True, exist_ok=True)


def split_for_client(zip_path: Path, client_id: int, base_client_dir: Path, fraction: float, train_ratio: float):
    """
    Extracts classes from zip_path, samples `fraction` of images per class,
    splits into train/test by `train_ratio`, and copies them into:
      base_client_dir/client_<ID>/{train,test}/{class}/

    Existing directories are reused and files are not overwritten.
    Supports zips with either class folders at top-level or one top-level folder containing classes.
    """
    client_dir = base_client_dir / f"client_{client_id}"
    train_root = client_dir / "train"
    test_root  = client_dir / "test"

    # Inform about existing structure
    if client_dir.exists():
        print(f"Directory {client_dir} already exists; new images will be added without overwriting.")
    else:
        print(f"Creating new directories at {client_dir}.")

    # Create train/test directories if missing
    for path in (train_root, test_root):
        path.mkdir(parents=True, exist_ok=True)

    # Extract ZIP to temporary folder
    with tempfile.TemporaryDirectory(prefix="fl_dataset_") as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(tmp_path)

        # Find actual class parent folder(s)
        root_dirs = [p for p in tmp_path.iterdir() if p.is_dir() and not p.name.startswith('__')]
        if not root_dirs:
            print(f"ERROR: No directories found in ZIP at {tmp_path}", file=sys.stderr)
            sys.exit(1)

        # Case A: direct class folders containing image files
        direct = [d for d in root_dirs if any((d / f).is_file() for f in os.listdir(d))]
        if direct:
            class_parent = tmp_path
            classes = [d.name for d in direct]
        else:
            # Case B: one top-level folder containing classes
            class_parent = root_dirs[0]
            classes = [d.name for d in class_parent.iterdir() if d.is_dir() and not d.name.startswith('__')]
            if not classes:
                print(f"ERROR: No class folders under {class_parent}", file=sys.stderr)
                sys.exit(1)

        print(f"Detected classes: {classes}")

        random.seed(client_id)
        # Process each class
        for cls in classes:
            src = class_parent / cls
            images = [f.name for f in src.iterdir() if f.is_file()]
            if not images:
                print(f"Warning: No images in class '{cls}'", file=sys.stderr)
                continue

            # Sample and split
            k = max(1, int(len(images) * fraction))
            sampled = random.sample(images, k)
            n_train = max(1, int(len(sampled) * train_ratio))
            train_imgs, test_imgs = sampled[:n_train], sampled[n_train:]

            for split, imgs in [('train', train_imgs), ('test', test_imgs)]:
                out_dir = client_dir / split / cls
                out_dir.mkdir(parents=True, exist_ok=True)
                for fname in imgs:
                    dest = out_dir / fname
                    if dest.exists():
                        print(f"Skipping existing file: {dest}")
                    else:
                        shutil.copy2(src / fname, dest)

    print(f"Client {client_id} data prepared at {client_dir}")


if __name__ == '__main__':
    split_for_client(ZIP_FILE, CLIENT_ID, BASE_DIR, FRACTION, TRAIN_RATIO)
    print("Dataset split complete.")
