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
    def __init__(self, model, train_loader, test_loader, client_id):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.client_id = client_id

    def get_parameters(self, config):
        # Return model parameters as NumPy arrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        # Load parameters into model
        state_dict = {
            k: torch.tensor(v)
            for k, v in zip(self.model.state_dict().keys(), parameters)
        }
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # 1) Receive global parameters
        self.set_parameters(parameters)
        self.model.train()

        # 2) Local training
        epochs = int(config.get("local_epochs", 1))
        num_samples = len(self.train_loader.dataset)
        print(f"→ Client {self.client_id} training for {epochs} epoch(s) on {num_samples} samples")
        optimizer = torch.optim.SGD(self.model.parameters(), lr=config.get("lr", 0.01))
        for _ in range(epochs):
            for x, y in self.train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(self.model(x), y)
                loss.backward()
                optimizer.step()

        # 3) Return updated params + sample count + metadata
        new_params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        return new_params, num_samples, {"local_epochs": epochs}

    def evaluate(self, parameters, config):
        # 1) Receive global parameters
        self.set_parameters(parameters)
        self.model.eval()

        # 2) Evaluate on local test set
        total, correct = 0, 0
        loss_sum = 0.0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                outputs = self.model(x)
                loss_sum += float(
                    torch.nn.functional.cross_entropy(outputs, y, reduction="sum")
                )
                preds = outputs.argmax(dim=1)
                correct += int((preds == y).sum().item())
                total += y.size(0)

        # 3) Compute metrics
        loss = loss_sum / (total or 1)
        accuracy = correct / (total or 1)
        misclassified = (total or 0) - correct

        print(
            f"→ Client {self.client_id} eval → "
            f"loss={loss:.4f}, accuracy={accuracy:.4f}, "
            f"misclassified={misclassified}/{total}"
        )

        # 4) Return loss, sample count, and both metrics
        return float(loss), total, {
            "accuracy": accuracy,
            "misclassified": misclassified,
        }

def main():
    parser = argparse.ArgumentParser(description="Flower FL client")
    parser.add_argument(
        "-c", "--client-id",
        required=True,
        help="Client ID matching data folder (e.g. '1' for client_1')"
    )
    parser.add_argument(
        "-s", "--server-address",
        default="127.0.0.1:8080",
        help="Address of the Flower server (ip:port)"
    )
    parser.add_argument(
        "-nc", "--num-classes",
        type=int,
        default=5,
        help="Number of output classes"
    )
    parser.add_argument(
        "-e", "--local-epochs",
        type=int,
        default=1,
        help="Number of local epochs to train each round"
    )
    parser.add_argument(
        "-lr", "--learning-rate",
        type=float,
        default=0.01,
        help="SGD learning rate"
    )
    args = parser.parse_args()

    # Build model & load data
    model = build_model(num_classes=args.num_classes)
    train_loader, test_loader = load_client_data(client_id=args.client_id)

    # Wrap and start client
    client = FLClient(model, train_loader, test_loader, client_id=args.client_id)
    print(f"→ Client {args.client_id} connecting to server at {args.server_address}")
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client,
        config={"local_epochs": args.local_epochs, "lr": args.learning_rate},
    )

if __name__ == "__main__":
    main()
