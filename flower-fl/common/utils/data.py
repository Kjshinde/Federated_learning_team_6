# common/utils/data.py

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_client_data(client_id: str, batch_size: int = 32):
    """
    Load train/test DataLoaders for a given client.
    Expects directory structure at:
      <project_root>/client/data/client_<ID>/{train,test}/{class_folders}/
    """
    # project_root = two levels up from this file (common/utils → common → project root)
    project_root = Path(__file__).parents[2]

    data_dir = project_root / "client" / "data" / f"client_{client_id}"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    # Create datasets
    train_ds = datasets.ImageFolder(data_dir / "train", transform=transform)
    test_ds  = datasets.ImageFolder(data_dir / "test",  transform=transform)

    # Wrap in DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
