- `common/` holds everything both server and clients need (models, data‑loading, metrics).

- `server/` contains only the code to start your FL server; devs who only run the server can git sparse-checkout set server/ common/.

- `client/` contains only client code plus its data/ folder; each teammate clones client/ + common/, drops in their own client_<ID> data, and runs client.py.

- Separate `requirements.txt` means you can pin different deps if, say, the client needs PyTorch while the server only needs Flower.
# Directory Structure Generator

A lightweight Python script (`create_structure.py`) to scaffold your Flower‑based federated learning project. Quickly generate shared libraries, server components, client data partitions, or the full stack with one command.

---

## Prerequisites

- **Python 3.6+** (no external dependencies beyond the standard library)  
- (Optional) A virtual environment for isolation  

---

## Installation

1. **Clone** this repository (or copy `create_structure.py` into your project root):  
   ```bash
   git clone <repo-url>
   cd <repo-dir>
   ```
2. **Make executable** (optional):  
   ```bash
   chmod +x create_structure.py
   ```

---

## Usage

```bash
python create_structure.py [OPTIONS]
```

| Flag                       | Description                                                                                          |
|----------------------------|------------------------------------------------------------------------------------------------------|
| `-c, --client-id <ID>`     | _(default: `1`)_ Use numeric ID to name your client folder (`client/data/client_<ID>/`).             |
| `-b, --base-dir <DIR>`     | _(default: `flower-fl`)_ Base directory under which to scaffold.                                     |
| `--server`                 | Create **only** the shared (`common/`) and `server/` directories.                                     |
| `--full`                   | Create **all**: `common/`, `server/`, **and** one client folder under `client/data/`.                |
| _no flag_                  | Default to **client‑only** (plus shared `common/`).                                                   |

---

## Generated Layout

### 1. Client‑only (default)

```text
flower-fl/
├── common/
│   ├── models/
│   └── utils/
└── client/
    └── data/
        └── client_<ID>/
            ├── train/
            └── test/
```

### 2. Server‑only (`--server`)

```text
flower-fl/
├── common/
│   ├── models/
│   └── utils/
└── server/
    ├── server.py
    ├── requirements.txt
    └── Dockerfile
```

### 3. Full stack (`--full`)

```text
flower-fl/
├── common/
│   ├── models/
│   └── utils/
├── server/
│   ├── server.py
│   ├── requirements.txt
│   └── Dockerfile
└── client/
    ├── client.py
    ├── requirements.txt
    └── data/
        └── client_<ID>/
            ├── train/
            └── test/
```

---

## Examples

- **Client only** (ID 2):  
  ```bash
  python create_structure.py --client-id 2
  ```

- **Server only**:  
  ```bash
  python create_structure.py --server
  ```

- **Full stack** (client 5):  
  ```bash
  python create_structure.py --full --client-id 5
  ```

---

## Contributing

If you need additional flags, custom boilerplate, or alternative layouts, please open an issue or submit a pull request.

## Git Ignore

To prevent dataset images (and other generated files) from being committed, add a `.gitignore` at your project root with:

```gitignore
# Ignore all client data images and labels
/client/data/**

# (Optional) keep an empty placeholder file
!/client/data/**/.gitkeep

# Ignore dataset archives
*.zip

# Python cache and bytecode
__pycache__/
*.py[cod]

# Virtual environment
.venv/

# Logs and temporary files
*.log
temp*/

# IDE/editor settings
.vscode/
.idea/

# Flower checkpoints or other model artifacts
*.pth
*.pt
```

## Contributing

If you need additional flags, custom boilerplate, or alternative layouts, please open an issue or submit a pull request.
