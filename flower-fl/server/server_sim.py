#!/usr/bin/env python3
import flwr as fl
import argparse

def main():
    parser = argparse.ArgumentParser(description="Dummy Flower Server")
    parser.add_argument(
        "--port", "-p", type=int, default=8080,
        help="Port for the server to listen on"
    )
    parser.add_argument(
        "--rounds", "-r", type=int, default=3,
        help="Number of federated rounds"
    )
    args = parser.parse_args()

    print(f"Starting dummy server on 0.0.0.0:{args.port}, rounds={args.rounds}")
    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
    )

if __name__ == "__main__":
    main()
