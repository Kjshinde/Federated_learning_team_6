#!/usr/bin/env python3
import os
import argparse

def write_files(file_dict):
    for path, content in file_dict.items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)

def create_common(base_dir):
    # Shared models & utils
    dirs = [
        os.path.join(base_dir, "common", "models"),
        os.path.join(base_dir, "common", "utils"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    files = {
        os.path.join(base_dir, "common", "models", "cnn.py"):
            "# Define build_model() here\n",
        os.path.join(base_dir, "common", "utils", "data.py"):
            "# Implement load_client_data(client_id)\n",
        os.path.join(base_dir, "common", "utils", "metrics.py"):
            "# Custom evaluation metrics\n",
    }
    write_files(files)

def create_server(base_dir):
    # Flower server code & deps
    dirs = [os.path.join(base_dir, "server")]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    files = {
        os.path.join(base_dir, "server", "server.py"):
            "# Flower server entrypoint\n",
        os.path.join(base_dir, "server", "requirements.txt"):
            "flower\n",
        os.path.join(base_dir, "server", "Dockerfile"):
            "# Optional Dockerfile for server\n",
    }
    write_files(files)

def create_client(base_dir, client_id):
    # Client code & its data partition
    client_data = os.path.join(base_dir, "client", "data", f"client_{client_id}")
    dirs = [
        os.path.join(client_data, "train"),
        os.path.join(client_data, "test"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    files = {
        os.path.join(base_dir, "client", "client.py"):
            "# Flower client entrypoint\n",
        os.path.join(base_dir, "client", "requirements.txt"):
            "flower\ntorch\ntorchvision\n",
    }
    write_files(files)

def create_readme(base_dir):
    content = (
        "# Federated Learning Project (Flower)\n\n"
        "## Directory Structure\n"
        "```\n"
        "common/\n"
        "  models/\n"
        "  utils/\n"
        "server/\n"
        "client/\n"
        "  data/\n"
        "    client_<ID>/\n"
        "      train/\n"
        "      test/\n"
        "README.md\n"
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
        "--client-id", "-c", default="1",
        help="Client ID for naming the data folder (e.g. '1' → 'client_1')"
    )
    parser.add_argument(
        "--base-dir", "-b", default="flower-fl",
        help="Base directory to create"
    )
    args = parser.parse_args()

    # Always create README
    create_readme(args.base_dir)

    # Shared common code
    create_common(args.base_dir)

    if args.server:
        create_server(args.base_dir)
        print(f"Server structure created under '{args.base_dir}/server'")
    elif args.full:
        create_server(args.base_dir)
        create_client(args.base_dir, args.client_id)
        print(f"Full structure created under '{args.base_dir}' (client_{args.client_id} included)")
    else:
        # Default: client-only (assumes common already there)
        create_client(args.base_dir, args.client_id)
        print(f"Client structure created under '{args.base_dir}/client/data/client_{args.client_id}'")
