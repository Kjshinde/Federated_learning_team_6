# Federated Learning Project (Flower)

## Directory Structure
```
common/
  models/
  utils/
server/
client/
  data/
    client_<ID>/
      train/
      test/
README.md
```

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

## Split Dataset Script

To generate per-client train/test splits from a unified dataset zip, use `split_dataset.py`. This script extracts a fraction of images per class (e.g., Food, movie, notes, real_life, shopping), splits them into train/test by your specified ratio, and outputs them under:

```
flower-fl/client/data/client_<ID>/{train,test}/{class}/
```

**Configuration:**  
At the top of `split_dataset.py`, set:

```python
ZIP_FILE    = "/path/to/all_screenshots.zip"   # Path to your zipped dataset
CLIENT_ID   = 1                                # Unique numeric client ID
BASE_DIR    = "flower-fl/client/data"         # Base output directory
FRACTION    = 0.2                              # Fraction of images per class to sample
TRAIN_RATIO = 0.8                              # Fraction of sampled images for training
```

**Usage:**  
```bash
python split_dataset.py
```

Check your client directory after running:

```
flower-fl/client/data/client_<ID>/
├── train/
│   └── Food/, movie/, notes/, real_life/, shopping/
└── test/
    └── Food/, movie/, notes/, real_life/, shopping/
```

---

## Contributing

If you need additional flags, custom boilerplate, or alternative layouts, please open an issue or submit a pull request.

