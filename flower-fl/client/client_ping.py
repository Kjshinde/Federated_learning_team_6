#!/usr/bin/env python3
import socket
import argparse

def main():
    parser = argparse.ArgumentParser(description="Ping client for Internet access")
    parser.add_argument(
        "server_host",
        help="Public IP or hostname of your server"
    )
    parser.add_argument(
        "-p", "--port", type=int, default=8080,
        help="TCP port the server is listening on"
    )
    parser.add_argument(
        "-m", "--message", default="HELLO from client",
        help="Message payload to send"
    )
    args = parser.parse_args()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        print(f"Connecting to {args.server_host}:{args.port}…")
        client.connect((args.server_host, args.port))
        client.sendall(args.message.encode())
        data = client.recv(1024)
        print(f"Received from server: {data!r}")

if __name__ == "__main__":
    main()
