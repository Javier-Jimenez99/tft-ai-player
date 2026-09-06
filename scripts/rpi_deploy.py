"""Deploy and sync TFT AI collection service to Raspberry Pi."""

import os
import sys
from pathlib import Path
import paramiko

# Load .env credentials
RPI_HOST = "raspberrypi.local"
RPI_USER = "javi"
RPI_PASSWORD = "sal739567"

env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip("'\"")
                if k == "RPI_HOST":
                    RPI_HOST = v
                elif k == "RPI_USER":
                    RPI_USER = v
                elif k == "RPI_PASSWORD":
                    RPI_PASSWORD = v

print(f"Connecting to {RPI_USER}@{RPI_HOST}...")

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(RPI_HOST, username=RPI_USER, password=RPI_PASSWORD, timeout=10)

# 1. Check system info
stdin, stdout, stderr = ssh.exec_command("whoami; uname -a; pwd")
print("Remote info:\n" + stdout.read().decode())

# 2. Authorize local SSH key
pub_key_path = Path.home() / ".ssh" / "id_ed25519.pub"
if pub_key_path.exists():
    pub_key = pub_key_path.read_text().strip()
    ssh.exec_command("mkdir -p ~/.ssh && chmod 700 ~/.ssh")
    cmd = f'grep -qxF "{pub_key}" ~/.ssh/authorized_keys 2>/dev/null || echo "{pub_key}" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys'
    ssh.exec_command(cmd)
    print("SSH public key registered in remote ~/.ssh/authorized_keys")

# 3. Check where repo is located
stdin, stdout, stderr = ssh.exec_command("find /home/javi -maxdepth 3 -name 'tft-ai-player' 2>/dev/null")
found_repos = stdout.read().decode().strip().splitlines()
print(f"Found repos on Pi: {found_repos}")

ssh.close()
