import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes: int = 5):
        """
        A simple 2‑layer CNN for 5‑way screenshot classification.
        Args:
            num_classes: number of output labels (default 5).
        """
        super(CNN, self).__init__()
        # Conv layer block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        # Conv layer block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # After two 2×2 poolings on 32×32 input → 8×8 feature maps
        self.fc1   = nn.Linear(64 * 8 * 8, 128)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def build_model(num_classes: int = 5) -> nn.Module:
    """
    Instantiate and return the CNN model for `num_classes` labels.
    """
    return CNN(num_classes=num_classes)
