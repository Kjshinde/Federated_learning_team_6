#!/usr/bin/env python3
import os
import argparse


def write_files(file_dict):
    """
    Write a set of files given a mapping of path -> content.
    Creates parent directories as needed.
    """
    for path, content in file_dict.items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)


def create_common(base_dir):
    """
    Create shared 'common/models' and 'common/utils' directories and files.
    """
    dirs = [
        os.path.join(base_dir, "common", "models"),
        os.path.join(base_dir, "common", "utils"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    files = {
        os.path.join(base_dir, "common", "models", "cnn.py"): (
            "# Define build_model() here\n"
        ),
        os.path.join(base_dir, "common", "utils", "data.py"): (
            "# Implement load_client_data(client_id)\n"
        ),
        os.path.join(base_dir, "common", "utils", "metrics.py"): (
            "# Custom evaluation metrics\n"
        ),
    }
    write_files(files)


def create_server(base_dir):
    """
    Create 'server/' directory with server.py, requirements.txt, and Dockerfile.
    """
    server_dir = os.path.join(base_dir, "server")
    os.makedirs(server_dir, exist_ok=True)

    files = {
        os.path.join(server_dir, "server.py"): (
            "# Flower server entrypoint\n"
        ),
        os.path.join(server_dir, "requirements.txt"): (
            "flower\n"
        ),
        os.path.join(server_dir, "Dockerfile"): (
            "# Optional Dockerfile for server\n"
        ),
    }
    write_files(files)


def create_client(base_dir, client_id):
    """
    Create 'client/' directory, data partitions for the given client_id, plus client.py and requirements.txt.
    """
    client_base = os.path.join(base_dir, "client")
    client_data = os.path.join(client_base, "data", f"client_{client_id}")

    # Base train/test directories
    dirs = [
        os.path.join(client_data, "train"),
        os.path.join(client_data, "test"),
    ]

    # Subfolders for each class under train/ and test/
    classes = ["Food", "movie", "notes", "real_life", "shopping"]
    for split in ("train", "test"):
        for cls in classes:
            dirs.append(os.path.join(client_data, split, cls))

    for d in dirs:
        os.makedirs(d, exist_ok=True)

    files = {
        os.path.join(client_base, "client.py"): (
            "# Flower client entrypoint\n"
        ),
        os.path.join(client_base, "requirements.txt"): (
            "flower\ntorch\ntorchvision\n"
        ),
    }
    write_files(files)


def create_readme(base_dir):
    """
    Create a README.md at the base_dir.
    """
    content = (
        "# Federated Learning Project (Flower)\n\n"
        "A lightweight scaffold for Flower-based FL projects.\n\n"
        "## Directory Structure\n"
        
        "common/                   # shared models & utils\n"
        "server/                   # server code & requirements\n"
        "client/                   # client code, requirements, and data partitions\n"
        "```\n"
    )
    path = os.path.join(base_dir, "README.md")
    os.makedirs(base_dir, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate FL project directory structure"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--server", action="store_true",
        help="Only create shared 'common/' and 'server/' directories"
    )
    group.add_argument(
        "--full", action="store_true",
        help="Create 'common/', 'server/' and your client folder"
    )
    parser.add_argument(
        "-c", "--client-id", default="1",
        help="Client ID for naming the data folder (e.g. '1' → 'client_1')"
    )
    parser.add_argument(
        "-b", "--base-dir", default="flower-fl",
        help="Base directory to create"
    )
    args = parser.parse_args()

    # Always include README
    create_readme(args.base_dir)

    # Shared code
    create_common(args.base_dir)

    if args.server:
        create_server(args.base_dir)
        print(f"Server structure created under '{args.base_dir}/server'")
    elif args.full:
        create_server(args.base_dir)
        create_client(args.base_dir, args.client_id)
        print(f"Full structure created under '{args.base_dir}' (client_{args.client_id} included)")
    else:
        create_client(args.base_dir, args.client_id)
        print(f"Client structure created under '{args.base_dir}/client/data/client_{args.client_id}'")
