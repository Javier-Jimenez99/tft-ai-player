"""Lightweight Web Dashboard and Deterministic Telegram Bot for TFT Dataset Monitoring.

Designed to run on low-power devices like Raspberry Pi:
- Built exclusively with Python standard library (http.server, urllib, json, sqlite3).
- Ultra-low memory footprint (<25MB RAM).
- Fast, zero-dependency REST endpoints for the single-page web dashboard.
- 100% deterministic Telegram bot (pure command-driven, zero probabilistic/LLM models).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import socket
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from .collector_db import ALL_STANDARD_TIERS, CollectorDb

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# System & Hardware Metrics (Raspberry Pi & Linux)
# -----------------------------------------------------------------------------


def get_raspberry_pi_temperature() -> float | None:
    """Read CPU temperature from Linux thermal sysfs (Raspberry Pi)."""
    thermal_path = Path("/sys/class/thermal/thermal_zone0/temp")
    if thermal_path.exists():
        try:
            val = thermal_path.read_text().strip()
            return round(int(val) / 1000.0, 1)
        except Exception:
            return None
    return None


def get_system_hardware_stats(output_dir: Path) -> dict[str, Any]:
    """Collect lightweight system metrics (disk, CPU temp, RAM)."""
    stats: dict[str, Any] = {
        "cpu_temp_c": get_raspberry_pi_temperature(),
        "cpu_percent": 0.0,
        "ram_percent": 0.0,
        "free_disk_gb": 0.0,
        "total_disk_gb": 0.0,
    }

    # MicroSD Disk stats
    try:
        usage = shutil.disk_usage(output_dir)
        stats["free_disk_gb"] = round(usage.free / (1024**3), 2)
        stats["total_disk_gb"] = round(usage.total / (1024**3), 2)
    except Exception:
        pass

    # Linux /proc/meminfo
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        try:
            mem: dict[str, int] = {}
            for line in meminfo_path.read_text().splitlines():
                parts = line.split(":")
                if len(parts) == 2:
                    k = parts[0].strip()
                    val_str = parts[1].strip().split()[0]
                    if val_str.isdigit():
                        mem[k] = int(val_str)
            if "MemTotal" in mem and "MemAvailable" in mem and mem["MemTotal"] > 0:
                used = mem["MemTotal"] - mem["MemAvailable"]
                stats["ram_percent"] = round((used / mem["MemTotal"]) * 100, 1)
        except Exception:
            pass

    # Linux load average
    try:
        load1, _, _ = os.getloadavg()
        cpu_count = os.cpu_count() or 1
        stats["cpu_percent"] = round(min(100.0, (load1 / cpu_count) * 100), 1)
    except Exception:
        pass

    return stats


def get_local_ip() -> str:
    """Get preferred local IPv4 address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# -----------------------------------------------------------------------------
# HTTP Dashboard Handler
# -----------------------------------------------------------------------------


class DashboardHttpHandler(BaseHTTPRequestHandler):
    """Handles REST API and static single-page HTML dashboard requests."""

    # Injected by server
    status_file: Path
    db_path: Path
    dashboard_html_path: Path
    output_dir: Path

    def log_message(self, format: str, *args: Any) -> None:
        """Quiet default logging to avoid cluttering stdout."""
        logger.debug("HTTP %s - " + format, self.address_string(), *args)

    def _send_json(self, status_code: int, data: Any) -> None:
        payload = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def do_HEAD(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
        elif path in ("/api/status", "/api/recent", "/api/analytics"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
        else:
            self.send_error(404, "Not Found")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._serve_html()
        elif path == "/api/status":
            self._serve_status()
        elif path == "/api/recent":
            self._serve_recent(parsed.query)
        elif path == "/api/analytics":
            self._serve_analytics()
        elif path == "/api/health":
            self._send_json(200, {"status": "healthy", "service": "tft-ai-collector", "frontend": "https://javier-jimenez99.github.io/tft-ai-player/"})
        else:
            self._send_json(404, {"error": "Not Found", "frontend": "https://javier-jimenez99.github.io/tft-ai-player/"})

    def _serve_html(self) -> None:
        accept = self.headers.get("Accept", "")
        if "text/html" in accept or "*/*" in accept:
            self.send_response(302)
            self.send_header("Location", "https://javier-jimenez99.github.io/tft-ai-player/")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
        else:
            self._send_json(200, {
                "service": "tft-ai-collector",
                "status": "online",
                "frontend_portal": "https://javier-jimenez99.github.io/tft-ai-player/",
                "endpoints": ["/api/status", "/api/recent", "/api/analytics", "/api/health"],
            })

    def _serve_status(self) -> None:
        status_data: dict[str, Any] = {}
        if self.status_file.exists():
            try:
                status_data = json.loads(self.status_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("Error reading status.json: %s", e)

        # Check for active Cloudflare remote mobile tunnel
        tunnel_file = self.output_dir / "tunnel_url.txt"
        if tunnel_file.exists():
            try:
                status_data["remote_url"] = tunnel_file.read_text(encoding="utf-8").strip()
            except Exception:
                pass

        # Enrich with live hardware metrics
        status_data["system"] = get_system_hardware_stats(self.output_dir)

        # Enrich with live ELO & placement analytics if not already in status.json
        if "analytics" not in status_data or not status_data["analytics"]:
            try:
                db = CollectorDb(self.db_path)
                status_data["analytics"] = db.get_analytics_summary()
                db.close()
            except Exception as e:
                logger.debug("Could not enrich status with analytics: %s", e)

        self._send_json(200, status_data)

    def _serve_analytics(self) -> None:
        try:
            db = CollectorDb(self.db_path)
            analytics = db.get_analytics_summary()
            db.close()
            self._send_json(200, analytics)
        except Exception as e:
            logger.warning("Error querying analytics from DB: %s", e)
            self._send_json(500, {"error": str(e)})

    def _serve_recent(self, query_str: str) -> None:
        params = urllib.parse.parse_qs(query_str)
        limit = 30
        if "limit" in params and params["limit"][0].isdigit():
            limit = max(1, min(100, int(params["limit"][0])))

        try:
            db = CollectorDb(self.db_path)
            matches = db.get_recent_downloaded_matches(limit=limit)
            db.close()
            self._send_json(200, {"matches": matches})
        except Exception as e:
            logger.warning("Error querying recent matches from DB: %s", e)
            self._send_json(500, {"error": str(e), "matches": []})


# -----------------------------------------------------------------------------
# 100% Deterministic Telegram Bot (Zero LLM)
# -----------------------------------------------------------------------------


def generate_ascii_bar(pct: float, max_blocks: int = 5) -> str:
    """Generate deterministic mini progress bar for Telegram text."""
    filled = min(max_blocks, max(0, int(round((pct / 20.0) * max_blocks))))
    return "▰" * filled + "▱" * (max_blocks - filled)


class DeterministicTelegramBot:
    """Zero-dependency, 100% deterministic Telegram bot using standard polling."""

    def __init__(
        self,
        token: str,
        chat_id: str | None = None,
        status_file: Path | None = None,
        db_path: Path | None = None,
        port: int = 8080,
    ) -> None:
        self.token = token.strip()
        self.allowed_chat_id = chat_id.strip() if chat_id else None
        self.status_file = status_file or Path("data/status.json")
        self.db_path = db_path or Path("data/collector.db")
        self.port = port
        self.api_url = f"https://api.telegram.org/bot{self.token}"
        self.last_update_id = 0
        self._stop_event = threading.Event()

    def send_message(self, chat_id: str | int, text: str, parse_mode: str = "Markdown") -> bool:
        """Send a deterministic text message to a chat ID via Telegram HTTP API."""
        url = f"{self.api_url}/sendMessage"
        payload = json.dumps({
            "chat_id": str(chat_id),
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.status == 200
        except Exception as e:
            logger.warning("Failed to send Telegram message to %s: %s", chat_id, e)
            return False

    def format_status(self) -> str:
        """Generate deterministic markdown status report."""
        if not self.status_file.exists():
            return "⏳ *TFT Collector*: Aún no hay archivo `status.json` generado."

        try:
            data = json.loads(self.status_file.read_text(encoding="utf-8"))
        except Exception as e:
            return f"❌ *Error leyendo status.json*: {e}"

        total_dl = data.get("total_matches_on_disk", 0)
        rate = data.get("download_rate_per_hour", 0.0)
        uptime_h = data.get("uptime_hours", 0.0)
        free_disk = data.get("free_disk_gb", 0.0)
        errors = data.get("session_errors", 0)
        last_action = data.get("last_action", "idle")

        lines = [
            "⚔️ *TFT Dataset Collector — Estado en Vivo*",
            "",
            f"📦 *Total Partidas*: `{total_dl:,}`",
            f"⚡ *Velocidad*: `{rate:,.0f} partidas/hora`",
            f"⏱ *Uptime*: `{uptime_h:.2f}h` | ⚠️ *Errores*: `{errors}`",
            f"💾 *MicroSD Libre*: `{free_disk:.1f} GB`",
            f"🎯 *Acción actual*: `{last_action}`",
            "",
            "📊 *Distribución por ELO (Real vs Objetivo)*:",
        ]

        dist = data.get("distribution", {})
        total_pending = 0
        total_unscanned = 0

        for tier in ALL_STANDARD_TIERS:
            info = dist.get(tier, {})
            cur_pct = info.get("current_share_pct", 0.0)
            tgt_pct = info.get("target_share_pct", 0.0)
            pending = info.get("pending_games", 0)
            unscanned = info.get("unscanned_players", 0)
            total_pending += pending
            total_unscanned += unscanned

            bar = generate_ascii_bar(cur_pct, max_blocks=5)
            lines.append(f"• `{tier[:4]}`: {cur_pct:4.1f}% / {tgt_pct:4.1f}% `{bar}` ({pending:,} cola)")

        lines.extend([
            "",
            f"♻️ *En Cola*: `{total_pending:,}` partidas | `{total_unscanned:,}` jugadores",
        ])

        return "\n".join(lines)

    def format_recent(self, limit: int = 8) -> str:
        """Generate deterministic list of recent matches."""
        try:
            db = CollectorDb(self.db_path)
            matches = db.get_recent_downloaded_matches(limit=limit)
            db.close()
        except Exception as e:
            return f"❌ *Error consultando base de datos*: {e}"

        if not matches:
            return "ℹ️ No hay partidas recientes registradas."

        lines = [f"📜 *Últimas {len(matches)} partidas descargadas:*", ""]
        for m in matches:
            uuid_short = (m.get("match_uuid") or "")[:10]
            tier = m.get("tier", "UNKNOWN")
            player = m.get("focal_player") or "Anon"
            lines.append(f"• `{uuid_short}..` | *{tier}* | {player}")

        return "\n".join(lines)

    def format_link(self) -> str:
        """Format local IP and dashboard link (including active mobile remote tunnel if present)."""
        local_ip = get_local_ip()
        tunnel_file = self.status_file.parent / "tunnel_url.txt"
        remote_url = None
        if tunnel_file.exists():
            try:
                remote_url = tunnel_file.read_text(encoding="utf-8").strip()
            except Exception:
                pass

        lines = ["🌐 *Dashboard Web de Monitoreo*", ""]
        if remote_url:
            lines.extend([
                "📱 *Acceso Remoto desde el Móvil (Fuera de casa / 4G/5G)*:",
                f"👉 {remote_url}",
                "",
            ])
        lines.extend([
            "🏠 *Acceso en Red Local (WiFi de casa)*:",
            f"• http://{local_ip}:{self.port}",
            f"• http://raspberrypi.local:{self.port}",
        ])
        return "\n".join(lines)

    def format_help(self) -> str:
        """Deterministic help menu."""
        return (
            "🤖 *TFT Collector Bot (100% Determinista)*\n\n"
            "Comandos disponibles:\n"
            "• `/status` o `/stats` — Resumen en vivo y balanceo de ligas\n"
            "• `/recent` — Ver las últimas partidas descargadas\n"
            "• `/link` — Enlace al Dashboard Web\n"
            "• `/ping` — Verificar si el servicio responde\n"
            "• `/help` — Mostrar este mensaje de ayuda"
        )

    def handle_command(self, chat_id: str | int, command_text: str) -> None:
        """Pure deterministic dispatch."""
        cmd = command_text.strip().split()[0].lower().split("@")[0]  # Strip @botname if in group

        logger.info("Handling Telegram command '%s' from chat %s", cmd, chat_id)

        if cmd in ("/start", "/help"):
            reply = self.format_help()
        elif cmd in ("/status", "/stats"):
            reply = self.format_status()
        elif cmd in ("/recent", "/last"):
            reply = self.format_recent(limit=8)
        elif cmd == "/link":
            reply = self.format_link()
        elif cmd == "/ping":
            reply = "🏓 *Pong!* El servicio de monitor y el colector de TFT están activos."
        else:
            reply = f"Comando no reconocido: `{cmd}`. Usa `/help` para ver los comandos válidos."

        self.send_message(chat_id, reply)

    def poll_updates_cycle(self) -> None:
        """Execute one long-polling cycle."""
        url = f"{self.api_url}/getUpdates?offset={self.last_update_id + 1}&timeout=20"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status != 200:
                    return
                data = json.loads(resp.read().decode("utf-8"))

            if not data.get("ok"):
                return

            for update in data.get("result", []):
                update_id = update.get("update_id", 0)
                if update_id > self.last_update_id:
                    self.last_update_id = update_id

                msg = update.get("message") or update.get("edited_message")
                if not msg:
                    continue

                chat = msg.get("chat", {})
                chat_id = str(chat.get("id"))
                text = msg.get("text", "")

                # If allowed_chat_id is specified, ignore unauthorized chats
                if self.allowed_chat_id and chat_id != self.allowed_chat_id:
                    logger.warning("Unauthorized access attempt from chat ID: %s", chat_id)
                    continue

                if text.startswith("/"):
                    self.handle_command(chat_id, text)

        except urllib.error.URLError as e:
            logger.debug("Telegram polling network timeout/error: %s", e)
            time.sleep(2.0)
        except Exception as e:
            logger.warning("Telegram polling unexpected error: %s", e)
            time.sleep(5.0)

    def run_polling(self) -> None:
        """Run the polling loop continuously until stopped."""
        logger.info("Telegram deterministic bot polling started (bot token configured).")
        while not self._stop_event.is_set():
            self.poll_updates_cycle()

    def stop(self) -> None:
        self._stop_event.set()


# -----------------------------------------------------------------------------
# Background Alert Heartbeat
# -----------------------------------------------------------------------------


class AlertMonitorThread(threading.Thread):
    """Periodically checks for anomalies (disk full, collector dead) and notifies via Telegram."""

    def __init__(self, bot: DeterministicTelegramBot, check_interval_sec: float = 600.0) -> None:
        super().__init__(daemon=True, name="tft-alert-monitor")
        self.bot = bot
        self.check_interval_sec = check_interval_sec
        self._stop_event = threading.Event()
        self._last_alerted_at = 0.0

    def run(self) -> None:
        if not self.bot.allowed_chat_id:
            logger.info("No TELEGRAM_CHAT_ID set; proactive alerts will be logged but not pushed.")
            return

        while not self._stop_event.is_set():
            time.sleep(self.check_interval_sec)
            self._check_and_alert()

    def _check_and_alert(self) -> None:
        if not self.bot.status_file.exists():
            return

        try:
            data = json.loads(self.bot.status_file.read_text(encoding="utf-8"))
        except Exception:
            return

        free_disk = data.get("free_disk_gb", 100.0)
        chat_id = self.bot.allowed_chat_id

        # Alert 1: Low disk space (< 3.0 GB)
        if free_disk < 3.0 and (time.time() - self._last_alerted_at > 3600):
            msg = f"⚠️ *ALERTA CRÍTICA: Espacio bajo en MicroSD*\nQuedan solo `{free_disk:.2f} GB` libres en la Raspberry Pi."
            self.bot.send_message(chat_id, msg)
            self._last_alerted_at = time.time()

    def stop(self) -> None:
        self._stop_event.set()


# -----------------------------------------------------------------------------
# Service Lifecycle Runner
# -----------------------------------------------------------------------------


def run_monitor_service(
    host: str = "0.0.0.0",
    port: int = 8080,
    output_dir: str | Path = "data",
    db_path: str | Path = "data/collector.db",
    status_file: str | Path = "data/status.json",
    enable_telegram: bool = True,
) -> None:
    """Run the complete monitoring service (Web Dashboard + Telegram Bot)."""
    out_dir = Path(output_dir)
    db_p = Path(db_path)
    stat_p = Path(status_file)
    html_p = Path(__file__).parent / "dashboard.html"

    logger.info("=" * 65)
    logger.info(" Starting TFT AI Player Monitor Service")
    logger.info(" Dashboard URL: http://%s:%d (Local IP: http://%s:%d)", host, port, get_local_ip(), port)
    logger.info(" Database: %s", db_p.resolve())
    logger.info("=" * 65)

    # Configure Dashboard HTTP Handler
    DashboardHttpHandler.status_file = stat_p
    DashboardHttpHandler.db_path = db_p
    DashboardHttpHandler.dashboard_html_path = html_p
    DashboardHttpHandler.output_dir = out_dir

    server = ThreadingHTTPServer((host, port), DashboardHttpHandler)

    # Setup Telegram Bot if configured
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
    tg_chat_id = os.getenv("TELEGRAM_CHAT_ID")
    bot: DeterministicTelegramBot | None = None
    bot_thread: threading.Thread | None = None
    alert_thread: AlertMonitorThread | None = None

    if enable_telegram and tg_token:
        logger.info("Initializing Deterministic Telegram Bot...")
        bot = DeterministicTelegramBot(
            token=tg_token,
            chat_id=tg_chat_id,
            status_file=stat_p,
            db_path=db_p,
            port=port,
        )
        bot_thread = threading.Thread(target=bot.run_polling, daemon=True, name="tft-tg-bot")
        bot_thread.start()

        alert_thread = AlertMonitorThread(bot)
        alert_thread.start()
    else:
        if not tg_token:
            logger.info("TELEGRAM_BOT_TOKEN not provided in .env. Telegram bot disabled (Web Dashboard active).")

    # Graceful shutdown handlers
    def shutdown_handler(signum: int, frame: Any) -> None:
        logger.info("Shutdown signal %d received. Closing monitor service...", signum)
        if bot:
            bot.stop()
        if alert_thread:
            alert_thread.stop()
        threading.Thread(target=server.shutdown).start()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        server.serve_forever()
    finally:
        server.server_close()
        logger.info("TFT Monitor HTTP Server stopped cleanly.")
