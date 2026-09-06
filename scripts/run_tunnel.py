#!/usr/bin/env python3
"""Runner for Cloudflare Quick Tunnel on Raspberry Pi.

Starts cloudflared, captures the generated *.trycloudflare.com URL,
and writes it to data/tunnel_url.txt for easy mobile access.
"""

import logging
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tft_tunnel")

URL_REGEX = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")


def main() -> int:
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    tunnel_file = data_dir / "tunnel_url.txt"

    cmd = ["cloudflared", "tunnel", "--url", "http://localhost:8080", "--no-autoupdate"]
    logger.info("Starting Cloudflare Quick Tunnel: %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def handle_signal(signum, frame):
        logger.info("Signal %d received, stopping tunnel...", signum)
        proc.terminate()
        if tunnel_file.exists():
            try:
                tunnel_file.unlink()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    url_found = False

    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

            if not url_found:
                match = URL_REGEX.search(line)
                if match:
                    url = match.group(0)
                    url_found = True
                    tunnel_file.write_text(url, encoding="utf-8")
                    logger.info("=" * 70)
                    logger.info(" 🎉 ACCESO DESDE EL MÓVIL (URL PÚBLICA HTTPS):")
                    logger.info(" 👉 %s", url)
                    logger.info("=" * 70)

        proc.wait()
        return proc.returncode or 0
    finally:
        if proc.poll() is None:
            proc.terminate()
        if tunnel_file.exists():
            try:
                tunnel_file.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
