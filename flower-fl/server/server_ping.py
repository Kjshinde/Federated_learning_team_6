#!/usr/bin/env python3
import socket

HOST = ""       # listen on all interfaces
PORT = 8080     # your FL port

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((HOST, PORT))
        server.listen(1)
        print(f"Server listening on port {PORT}…")
        conn, addr = server.accept()
        with conn:
            print(f"Connected by {addr}")
            data = conn.recv(1024)
            print(f"Received from client: {data!r}")
            # Echo back an acknowledgment
            conn.sendall(b"ACK from server")

if __name__ == "__main__":
    main()
