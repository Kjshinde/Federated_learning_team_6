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
## 🔒 Windows Firewall & Port Forwarding

To allow external clients to reach your Flower server on port 8080, you need to:

1. **Open the port in Windows Defender Firewall**  
2. **Forward that port on your router**  

---

### 1. Windows Firewall Configuration

#### A. Using the GUI
1. Press ⊞ Win, type **“Windows Defender Firewall with Advanced Security”**, and press ENTER.  
2. In the left pane, click **Inbound Rules**.  
3. In the right pane, click **New Rule…**  
4. Select **Port**, click **Next**.  
5. Choose **TCP**, select **Specific local ports**, enter `8080`, click **Next**.  
6. Select **Allow the connection**, click **Next**.  
7. Check the network profiles you trust (e.g. **Domain**, **Private**), click **Next**.  
8. Give the rule a name (e.g. “Flower FL Server TCP 8080”), click **Finish**.

#### B. Using PowerShell (Admin)
Open PowerShell **as Administrator** and run:
```powershell
New-NetFirewallRule `
  -DisplayName "Flower FL Server TCP 8080" `
  -Direction Inbound `
  -Protocol TCP `
  -LocalPort 8080 `
  -Action Allow `
  -Profile Private,Domain
  ```

  
  
## 🔒 Ubuntu Firewall & Port Forwarding

To expose your Flower server on port 8080 from an Ubuntu machine, you’ll:

1. Open the port in the OS firewall (UFW)  
2. (If this Ubuntu box is acting as your network gateway) Enable IP forwarding and add an iptables NAT rule  

---

### 1. Allow TCP 8080 in UFW

```bash
# Check UFW status (enable if it’s inactive)
sudo ufw status verbose
sudo ufw enable   # only if ufw is inactive

# Allow Flower’s port
sudo ufw allow 8080/tcp

# Reload & verify
sudo ufw reload
sudo ufw status
```

# How to use Client and server sim script
python client_sim.py --server <SERVER_IP>:8080


cd flower-fl/server
python server_sim.py --port 8080 --rounds 3

# how to use client and server .py files

```
cd flower-fl/client
python client.py \
  --client-id 1 \
  --server-address 192.168.1.50:8080 \
  --num-classes 5
```

```
python server.py --port 8080 --num-rounds 20
```

# How to use new server.py
```
# Default FedAvg, 10 rounds
python server.py

# 25 rounds, use FedAdam
python server.py --num-rounds 25 --strategy FedAdam

# 50 rounds, try the adaptive Yogi optimizer
python server.py -r 50 -s FedYogi

```