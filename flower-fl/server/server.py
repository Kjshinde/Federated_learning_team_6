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
        "--port", "-p", type=int, default=8080,
        help="Port for the server to listen on"
    )
    parser.add_argument(
        "--num-rounds", "-r", type=int, default=10,
        help="Number of federated learning rounds"
    )
    args = parser.parse_args()

    # FedAvg Strategy tuned for a single client
    strategy = FedAvg(
        fraction_fit=1.0,            # 100% of clients participate in fit
        fraction_evaluate=1.0,       # 100% participate in evaluate
        min_fit_clients=1,           # don't wait for more than one client
        min_evaluate_clients=1,
        min_available_clients=1,     # start as soon as one client is available
    )

    print(f"Starting Flower server on 0.0.0.0:{args.port}, rounds={args.num_rounds}")
    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
