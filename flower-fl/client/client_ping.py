#!/usr/bin/env python3
import socket
import sys

if len(sys.argv) != 2:
    print("Usage: python client_ping.py <server-ip>")
    sys.exit(1)

SERVER_IP = sys.argv[1]
PORT      = 8080

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        print(f"Connecting to {SERVER_IP}:{PORT}…")
        client.connect((SERVER_IP, PORT))
        # Send a dummy packet
        client.sendall(b"HELLO from client")
        # Wait for ACK
        data = client.recv(1024)
        print(f"Received from server: {data!r}")

if __name__ == "__main__":
    main()
