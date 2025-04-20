# Implement load_client_data(client_id)
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_client_data(client_id: str, base_dir: str = None, batch_size: int = 32):
    """
    Load train/test DataLoaders for a given client.
    Assumes directory:
        <base_dir>/client/data/client_<ID>/{train,test}/<class_folders>/
    """
    if base_dir is None:
        base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "client", "data", f"client_{client_id}")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    test_ds  = datasets.ImageFolder(os.path.join(data_dir, "test"),  transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
