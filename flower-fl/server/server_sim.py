#!/usr/bin/env python3
import flwr as fl
import argparse
import numpy as np
from flwr.server.strategy import FedAvg

def main():
    parser = argparse.ArgumentParser(description="Dummy Flower Server (1‐client mode)")
    parser.add_argument("--port", "-p", type=int, default=8080)
    parser.add_argument("--rounds", "-r", type=int, default=3)
    args = parser.parse_args()

    # Dummy initial parameters (2×2 float32 matrix)
    initial_parameters = [np.zeros((2, 2), dtype=np.float32)]

    # FedAvg Strategy tweaked for a single client
    strategy = FedAvg(
        initial_parameters=initial_parameters,
        fraction_fit=1.0,              # have that one client participate in every fit round
        fraction_evaluate=1.0,         # have it participate in every eval round
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,       # don’t wait for more than one client
    )

    print(f"Starting dummy server on 0.0.0.0:{args.port}, rounds={args.rounds}")
    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
