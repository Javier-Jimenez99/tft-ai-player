import os
import sys
from pathlib import Path
import paramiko

# Load .env
env_path = Path(__file__).resolve().parent.parent / ".env"
RPI_PASSWORD = os.environ.get("RPI_PASSWORD", "")
RPI_HOST = os.environ.get("RPI_HOST", "raspberrypi.local")
RPI_USER = os.environ.get("RPI_USER", "javi")
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            if k == "RPI_PASSWORD": RPI_PASSWORD = v.strip("'\"")
            elif k == "RPI_HOST": RPI_HOST = v.strip("'\"")
            elif k == "RPI_USER": RPI_USER = v.strip("'\"")

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
print(f"Connecting via SSH/SFTP to {RPI_HOST}...")
ssh.connect(RPI_HOST, username=RPI_USER, password=RPI_PASSWORD, timeout=10)

sftp = ssh.open_sftp()

local_root = Path(__file__).resolve().parent.parent
remote_root = "/home/javi/tft-ai-player"

files_to_sync = [
    "src/tft_ai_player/dataset/collector_service.py",
    "src/tft_ai_player/dataset/monitor_service.py",
    "src/tft_ai_player/cli.py",
    "scripts/tft-collector.service",
]

for rel_path in files_to_sync:
    local_file = local_root / rel_path
    remote_file = f"{remote_root}/{rel_path}"
    print(f"Uploading {rel_path} -> {remote_file}")
    sftp.put(str(local_file), remote_file)

sftp.close()

def run_remote(cmd, sudo=False):
    print(f"\n>>> {cmd}")
    if sudo:
        stdin, stdout, stderr = ssh.exec_command(f"echo {RPI_PASSWORD} | sudo -S {cmd}")
    else:
        stdin, stdout, stderr = ssh.exec_command(cmd)
    raw = stdout.read()
    print(raw.decode("utf-8", errors="replace").strip())
    err = stderr.read().decode("utf-8", errors="replace").strip()
    filtered = "\n".join([line for line in err.splitlines() if "password for" not in line])
    if filtered:
        print("ERR:", filtered)

# 1. Update systemd service
run_remote("cp /home/javi/tft-ai-player/scripts/tft-collector.service /etc/systemd/system/tft-collector.service", sudo=True)

# 2. Stop and disable tft-monitor
run_remote("systemctl stop tft-monitor", sudo=True)
run_remote("systemctl disable tft-monitor", sudo=True)

# 3. Reload and restart collector
run_remote("systemctl daemon-reload", sudo=True)
run_remote("systemctl restart tft-collector", sudo=True)

import time
print("\nWaiting 5s for collector service & API to initialize...")
time.sleep(5)

# 4. Check active services
run_remote("systemctl is-active tft-collector")
run_remote("systemctl is-active tft-monitor")

# 5. Check API responsiveness
print("\n--- Testing API Endpoints on Raspberry Pi localhost:8080 ---")
run_remote("curl -s http://127.0.0.1:8080/api/health")
run_remote("curl -s http://127.0.0.1:8080/api/status | head -c 200")
run_remote("curl -s -I http://127.0.0.1:8080/ | head -n 5")

ssh.close()
print("\nSFTP Sync and Pi Deployment Complete!")
