# common/utils/data.py

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define your class names here:
CLASS_NAMES = ["Food", "movie", "notes", "real_life", "shopping"]


def load_client_data(client_id: str, batch_size: int = 32):
    """
    Load train/test DataLoaders for a given client.
    Expects directory structure at:
      <project_root>/client/data/client_<ID>/{train,test}/{CLASS_NAMES}/

    If class folders are missing, they will be created (empty).
    """
    # Determine project root (common/utils -> common -> project root)
    project_root = Path(__file__).parents[2]

    # Path to this client's data
    data_dir = project_root / "client" / "data" / f"client_{client_id}"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Ensure class subfolders exist under train/ and test/
    for split in ("train", "test"):
        for cls in CLASS_NAMES:
            cls_path = data_dir / split / cls
            cls_path.mkdir(parents=True, exist_ok=True)

    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    # Create datasets
    train_folder = data_dir / "train"
    test_folder = data_dir / "test"
    train_ds = datasets.ImageFolder(str(train_folder), transform=transform)
    test_ds  = datasets.ImageFolder(str(test_folder),  transform=transform)

    # Wrap in DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
