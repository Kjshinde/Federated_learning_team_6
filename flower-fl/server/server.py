# server.py
#!/usr/bin/env python3
import argparse

import flwr as fl
from flwr.server.strategy import FedAvg

def main():
    parser = argparse.ArgumentParser(
        description="Flower federated learning server (1-client mode)"
    )
    parser.add_argument(
        "--num-rounds", "-r", type=int, default=10,
        help="Number of federated learning rounds"
    )
    parser.add_argument(
        "--port", "-p", type=int, default=8080,
        help="Port for the server to listen on"
    )
    args = parser.parse_args()

    print(f"Starting Flower server on 0.0.0.0:{args.port}, rounds={args.num_rounds}")

    # FedAvg strategy tuned for a single client
    strategy = FedAvg(
        fraction_fit=1.0,            # all clients (just one) participate in fit
        fraction_evaluate=1.0,       # all clients participate in eval
        min_fit_clients=1,           # don’t wait for more than 1 client to start fitting
        min_evaluate_clients=1,      # same for evaluation
        min_available_clients=1,     # start as soon as one client is available
    )

    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
