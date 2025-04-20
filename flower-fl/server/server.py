# Flower server entrypoint
import flwr as fl
import argparse

def main():
    parser = argparse.ArgumentParser(description="Flower federated learning server")
    parser.add_argument(
        "--num-rounds", "-r", type=int, default=10,
        help="Number of federated learning rounds"
    )
    parser.add_argument(
        "--port", "-p", type=int, default=8080,
        help="Port for the server to listen on"
    )
    args = parser.parse_args()

    fl.server.start_server(
        server_address=f"[::]:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
    )

if __name__ == "__main__":
    main()
