#!/usr/bin/env python3
import flwr as fl
import numpy as np
import argparse

class DummyClient(fl.client.NumPyClient):
    def get_parameters(self):
        # Send a 2×2 “parameter matrix” of random floats
        params = [np.random.randn(2, 2).astype(np.float32)]
        print("→ get_parameters(): sending", params)
        return params

    def fit(self, parameters, config):
        print("→ fit()  : received", parameters, "with config", config)
        # Simulate “training” by adding 1.0 to each element
        new_params = [p + 1.0 for p in parameters]
        num_examples = 5
        print("→ fit()  : returning", new_params)
        return new_params, num_examples, {}

    def evaluate(self, parameters, config):
        print("→ evaluate(): received", parameters)
        # Return fixed loss/accuracy
        loss = float(np.mean(parameters[0]))
        num_examples = 5
        metrics = {"accuracy": float(np.mean(parameters[0]) * 0.01)}
        print(f"→ evaluate(): loss={loss:.4f}, acc={metrics['accuracy']:.4f}")
        return loss, num_examples, metrics

def main():
    parser = argparse.ArgumentParser(description="Dummy Flower Client")
    parser.add_argument(
        "--server", "-s", type=str, default="127.0.0.1:8080",
        help="Address of the Flower server (host:port)"
    )
    args = parser.parse_args()

    client = DummyClient()
    print("Connecting to server at", args.server)
    fl.client.start_numpy_client(
        server_address=args.server,
        client=client,
    )

if __name__ == "__main__":
    main()
