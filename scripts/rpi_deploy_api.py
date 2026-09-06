import time
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
print("Connecting to raspberrypi.local...")
ssh.connect("raspberrypi.local", username="javi", password="sal739567", timeout=10)

def run_remote(cmd, sudo=False):
    print(f"\n>>> {cmd}")
    if sudo:
        stdin, stdout, stderr = ssh.exec_command(f"echo sal739567 | sudo -S {cmd}")
    else:
        stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if out:
        print(out)
    filtered_err = "\n".join([line for line in err.splitlines() if "password for" not in line])
    if filtered_err:
        print("ERR:", filtered_err)
    return out

# 1. Update repo on Raspberry Pi
run_remote("cd /home/javi/tft-ai-player && git fetch origin && git checkout rl-models && git pull origin rl-models")

# 2. Update tft-collector service file with --api-port 8080
run_remote("cp /home/javi/tft-ai-player/scripts/tft-collector.service /etc/systemd/system/tft-collector.service", sudo=True)

# 3. Stop and disable tft-monitor (no longer needed, zero HTML files on Pi)
run_remote("systemctl stop tft-monitor", sudo=True)
run_remote("systemctl disable tft-monitor", sudo=True)

# 4. Reload daemon and restart collector
run_remote("systemctl daemon-reload", sudo=True)
run_remote("systemctl restart tft-collector", sudo=True)

print("Waiting 4s for collector & API initialization...")
time.sleep(4)

# 5. Check service status
run_remote("systemctl is-active tft-collector")
run_remote("systemctl is-active tft-monitor")

# 6. Test endpoints
print("\n--- Testing API endpoints on localhost:8080 ---")
run_remote("curl -s http://127.0.0.1:8080/api/health")
run_remote('curl -s -o /dev/null -w "HTTP Status: %{http_code}\n" http://127.0.0.1:8080/api/status')
run_remote('curl -s -I http://127.0.0.1:8080/ | head -n 5')

ssh.close()
print("\nDeployment completed successfully!")
