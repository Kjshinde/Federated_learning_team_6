#!/usr/bin/env python3
import os
import sys
import argparse
import flwr as fl
import torch

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.models.cnn import build_model
from common.utils.data import load_client_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, cid):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cid = cid

    def get_parameters(self, config):
        return [v.cpu().numpy() for _, v in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = {
            k: torch.tensor(v)
            for k, v in zip(self.model.state_dict().keys(), parameters)
        }
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        epochs = int(config.get("local_epochs", 1))
        lr = float(config.get("lr", 0.01))
        print(f"→ Client {self.cid}: training {epochs} epochs on {len(self.train_loader.dataset)} samples (lr={lr})")
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.model.train()
        for _ in range(epochs):
            for x, y in self.train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(self.model(x), y)
                loss.backward()
                optimizer.step()
        new_params = [v.cpu().numpy() for _, v in self.model.state_dict().items()]
        return new_params, len(self.train_loader.dataset), {"local_epochs": epochs}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        total, correct = 0, 0
        loss_sum = 0.0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                outputs = self.model(x)
                loss_sum += float(torch.nn.functional.cross_entropy(outputs, y, reduction="sum"))
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total   += y.size(0)
        loss = loss_sum / total
        accuracy = correct / total
        miscls = total - correct
        print(f"→ Client {self.cid}: eval loss={loss:.4f}, acc={accuracy:.4f}, miscls={miscls}/{total}")
        return float(loss), total, {"accuracy": accuracy, "misclassified": miscls}

def main():
    parser = argparse.ArgumentParser(description="Flower FL client")
    parser.add_argument(
        "-c", "--client-id", required=True,
        help="Client ID matching data folder (e.g. '1' → client_1')"
    )
    parser.add_argument(
        "-s", "--server-address",
        default="127.0.0.1:8080",
        help="Server address (ip:port)"
    )
    parser.add_argument(
        "-nc", "--num-classes", type=int, default=5,
        help="Number of output classes"
    )
    args = parser.parse_args()

    model = build_model(num_classes=args.num_classes)
    train_loader, test_loader = load_client_data(client_id=args.client_id)

    client = FLClient(model, train_loader, test_loader, cid=args.client_id)
    print(f"→ Client {args.client_id} connecting to server at {args.server_address}")
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client,
    )

if __name__ == "__main__":
    main()
