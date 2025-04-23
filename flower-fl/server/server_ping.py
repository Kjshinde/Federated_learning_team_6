#!/usr/bin/env python3
import socket
import argparse

def main():
    parser = argparse.ArgumentParser(description="Ping server for Internet access")
    parser.add_argument(
        "-H", "--host", default="0.0.0.0",
        help="Host/IP to bind (0.0.0.0 = all interfaces)"
    )
    parser.add_argument(
        "-p", "--port", type=int, default=8080,
        help="TCP port to listen on"
    )
    args = parser.parse_args()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((args.host, args.port))
        server.listen(1)
        print(f"Server listening on {args.host}:{args.port}…")
        conn, addr = server.accept()
        with conn:
            print(f"Connected by {addr}")
            data = conn.recv(1024)
            print(f"Received from client: {data!r}")
            conn.sendall(b"ACK from server")

if __name__ == "__main__":
    main()
