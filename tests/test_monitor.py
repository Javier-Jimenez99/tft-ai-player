"""Unit tests for the Web Dashboard and Deterministic Telegram Bot."""

import json
import tempfile
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tft_ai_player.dataset.collector_db import CollectorDb
from tft_ai_player.dataset.downloader import setup_network_proxy
from tft_ai_player.dataset.monitor_service import (
    DashboardHttpHandler,
    DeterministicTelegramBot,
    generate_ascii_bar,
    get_system_hardware_stats,
)


@pytest.fixture
def temp_monitor_env():
    setup_network_proxy(None)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        db_path = tmp / "collector.db"
        status_file = tmp / "status.json"
        html_file = tmp / "dashboard.html"

        # Create dummy db with a match
        db = CollectorDb(db_path)
        cursor = db.conn.cursor()
        cursor.execute(
            """
            INSERT INTO downloaded_matches (match_uuid, tier, focal_player, downloaded_at)
            VALUES ('test-match-1234', 'CHALLENGER', 'TestPlayer#NA1', 1700000000.0);
            """
        )
        cursor.close()
        db.close()

        # Create dummy status file
        status_data = {
            "service": "tft-ai-collector",
            "uptime_hours": 1.5,
            "total_matches_on_disk": 100,
            "session_downloaded": 25,
            "session_errors": 0,
            "download_rate_per_hour": 1500.0,
            "free_disk_gb": 42.5,
            "last_action": "downloading_CHALLENGER_test",
            "distribution": {
                "CHALLENGER": {
                    "downloaded": 10,
                    "pending_games": 5,
                    "unscanned_players": 2,
                    "current_share_pct": 10.0,
                    "target_share_pct": 10.0,
                },
                "MASTER": {
                    "downloaded": 20,
                    "pending_games": 10,
                    "unscanned_players": 4,
                    "current_share_pct": 20.0,
                    "target_share_pct": 20.0,
                },
            },
        }
        status_file.write_text(json.dumps(status_data), encoding="utf-8")
        html_file.write_text("<html><body><h1>TFT Dashboard</h1></body></html>", encoding="utf-8")

        yield tmp, db_path, status_file, html_file


def test_system_stats(temp_monitor_env):
    tmp, _, _, _ = temp_monitor_env
    stats = get_system_hardware_stats(tmp)
    assert "free_disk_gb" in stats
    assert "total_disk_gb" in stats
    assert stats["free_disk_gb"] >= 0.0


def test_ascii_bar():
    assert generate_ascii_bar(0.0) == "▱▱▱▱▱"
    assert generate_ascii_bar(10.0) == "▰▰▱▱▱"
    assert generate_ascii_bar(20.0) == "▰▰▰▰▰"


def test_dashboard_http_server(temp_monitor_env):
    tmp, db_path, status_file, html_file = temp_monitor_env

    DashboardHttpHandler.status_file = status_file
    DashboardHttpHandler.db_path = db_path
    DashboardHttpHandler.dashboard_html_path = html_file
    DashboardHttpHandler.output_dir = tmp

    server = ThreadingHTTPServer(("127.0.0.1", 0), DashboardHttpHandler)
    port = server.server_port

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        # Test GET /
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as resp:
            assert resp.status == 200
            content = resp.read().decode("utf-8")
            assert "TFT Dashboard" in content

        # Test GET /api/status
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["service"] == "tft-ai-collector"
            assert "system" in data
            assert data["total_matches_on_disk"] == 100

        # Test GET /api/recent
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/recent?limit=10") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert "matches" in data
            assert len(data["matches"]) == 1
            assert data["matches"][0]["match_uuid"] == "test-match-1234"
            assert data["matches"][0]["tier"] == "CHALLENGER"

        # Test 404
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/invalid")
        assert exc_info.value.code == 404

    finally:
        server.shutdown()
        server.server_close()


def test_deterministic_telegram_bot(temp_monitor_env):
    tmp, db_path, status_file, _ = temp_monitor_env

    bot = DeterministicTelegramBot(
        token="DUMMY_TOKEN_12345",
        chat_id="999999",
        status_file=status_file,
        db_path=db_path,
        port=8080,
    )

    # 1. Test status formatting
    status_text = bot.format_status()
    assert "TFT Dataset Collector" in status_text
    assert "100" in status_text
    assert "CHAL" in status_text

    # 2. Test recent formatting
    recent_text = bot.format_recent()
    assert "test-match" in recent_text
    assert "CHALLENGER" in recent_text

    # 3. Test link formatting
    link_text = bot.format_link()
    assert ":8080" in link_text

    # 4. Test deterministic dispatch
    with patch.object(bot, "send_message", return_value=True) as mock_send:
        # /help
        bot.handle_command("999999", "/help")
        mock_send.assert_called_once()
        assert "/status" in mock_send.call_args[0][1]
        mock_send.reset_mock()

        # /ping
        bot.handle_command("999999", "/ping")
        assert "Pong" in mock_send.call_args[0][1]
        mock_send.reset_mock()

        # /status
        bot.handle_command("999999", "/status")
        assert "Total Partidas" in mock_send.call_args[0][1]
        mock_send.reset_mock()

        # /recent
        bot.handle_command("999999", "/recent")
        assert "Últimas" in mock_send.call_args[0][1]
        mock_send.reset_mock()

        # Unknown command
        bot.handle_command("999999", "/unknown_cmd")
        assert "Comando no reconocido" in mock_send.call_args[0][1]
