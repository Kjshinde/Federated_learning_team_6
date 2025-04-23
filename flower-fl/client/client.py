#!/usr/bin/env python3
import os
import sys
# Ensure the project root is on PYTHONPATH so we can import common/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import flwr as fl
import torch

from common.models.cnn import build_model
from common.utils.data import load_client_data

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_parameters(self):
        # Return model parameters as a list of NumPy arrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        # Convert list of NumPy arrays back to model state_dict
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # Update model with received parameters
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=config.get("lr", 0.01))
        epochs = int(config.get("local_epochs", 1))
        for _ in range(epochs):
            for x, y in self.train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                outputs = self.model(x)
                loss = torch.nn.functional.cross_entropy(outputs, y)
                loss.backward()
                optimizer.step()
        # Return updated parameters and number of samples trained on
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        # Update model then evaluate on local test set
        self.set_parameters(parameters)
        self.model.eval()
        loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                outputs = self.model(x)
                loss += float(torch.nn.functional.cross_entropy(outputs, y, reduction="sum"))
                _, preds = outputs.max(1)
                total += y.size(0)
                correct += (preds == y).sum().item()
        loss /= total
        accuracy = correct / total
        # Return evaluation results
        return float(loss), total, {"accuracy": accuracy}


def main():
    parser = argparse.ArgumentParser(description="Flower federated learning client")
    parser.add_argument(
        "-c", "--client-id", type=str, required=True,
        help="Client ID matching data folder (e.g. '1' for client_1)"
    )
    parser.add_argument(
        "-s", "--server-address", type=str, default="127.0.0.1:8080",
        help="Address of the Flower server (ip:port)"
    )
    parser.add_argument(
        "-nc", "--num-classes", type=int, default=5,
        help="Number of classes in the dataset"
    )
    args = parser.parse_args()

    # Build the model for the specified number of classes
    model = build_model(num_classes=args.num_classes)
    # Load this client's data loaders
    train_loader, test_loader = load_client_data(client_id=args.client_id)

    # Start Flower client
    client = FLClient(model, train_loader, test_loader)
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client,
    )

if __name__ == "__main__":
    main()
