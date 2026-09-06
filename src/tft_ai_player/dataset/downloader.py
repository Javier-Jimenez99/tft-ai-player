"""Unified match timeline downloader and observation extraction pipeline.

Provides shared network proxying (SOCKS5/HTTP), Tor IP circuit rotation on HTTP 429,
robust timeline fetching, PvP round observation extraction, and partitioned CSV storage.
Shared by both graph manifest downloading (`download-games`) and leaderboard crawling (`collect`).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

from tqdm import tqdm

from tft_ai_player.dataset.models import RoundObservation
from tft_ai_player.dataset.storage import PlayerCsvWriter
from tft_ai_player.dataset.timeline import TimelineValidationError, extract_pvp_rounds
import socket
from tft_ai_player.metatft.client import MetaTftClient, MetaTftRequestError

logger = logging.getLogger(__name__)

_original_socket = socket.socket


def setup_network_proxy(proxy_str: str | None) -> None:
    """Configure global socket or HTTP proxy for requests."""
    global _original_socket
    if not proxy_str or not proxy_str.strip():
        socket.socket = _original_socket
        try:
            import socks

            socks.set_default_proxy()
        except Exception:
            pass
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)
        return
    proxy_str = proxy_str.strip()
    parsed = urlparse(proxy_str if "://" in proxy_str else f"socks5://{proxy_str}")
    if parsed.scheme.startswith("socks"):
        try:
            import socks

            proxy_type = socks.SOCKS5 if "5" in parsed.scheme else socks.SOCKS4
            port = parsed.port or 1080
            socks.set_default_proxy(
                proxy_type,
                parsed.hostname,
                port,
                rdns=True,
                username=parsed.username,
                password=parsed.password,
            )
            socket.socket = socks.socksocket
            logger.info("Configured SOCKS proxy: %s:%s", parsed.hostname, port)
        except ImportError:
            raise RuntimeError(
                "PySocks is required for SOCKS proxy support. Install with: pip install pysocks"
            ) from None
    elif parsed.scheme.startswith("http"):
        os.environ["http_proxy"] = proxy_str
        os.environ["https_proxy"] = proxy_str
        logger.info("Configured HTTP proxy: %s", proxy_str)


def rotate_tor_identity(control_host: str = "127.0.0.1", control_port: int = 9051) -> bool:
    """Request a new Tor circuit / IP identity via Tor ControlPort."""
    try:
        import socket

        try:
            import socks
            raw_socket = socks.socksocket(socket.AF_INET, socket.SOCK_STREAM)
            raw_socket.set_proxy()  # Direct localhost connection bypassing SOCKS monkeypatch
        except Exception:
            raw_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        raw_socket.settimeout(5.0)
        raw_socket.connect((control_host, control_port))
        raw_socket.sendall(b'AUTHENTICATE ""\r\n')
        auth_resp = raw_socket.recv(1024)
        if b"250" not in auth_resp:
            raw_socket.close()
            return False
        raw_socket.sendall(b"SIGNAL NEWNYM\r\n")
        sig_resp = raw_socket.recv(1024)
        raw_socket.close()
        return b"250" in sig_resp
    except Exception as e:
        logger.debug("Tor rotation failed: %s", e)
        return False


def get_existing_match_ids(output_dir: Path | str) -> set[str]:
    """Scan existing CSV output directory and extract already downloaded match UUIDs."""
    p = Path(output_dir)
    if not p.exists():
        return set()
    return PlayerCsvWriter(p).existing_match_ids()


def add_common_download_arguments(parser: argparse.ArgumentParser) -> None:
    """Add standard networking, rate-limiting, and storage arguments to a parser."""
    parser.add_argument(
        "--proxy",
        type=str,
        default=os.environ.get("ALL_PROXY") or os.environ.get("SOCKS_PROXY") or os.environ.get("HTTPS_PROXY"),
        help="optional HTTP or SOCKS5 proxy URL (e.g. socks5://127.0.0.1:9050)",
    )
    parser.add_argument(
        "--tor-control-port",
        type=int,
        default=None,
        help="optional Tor ControlPort (e.g. 9051) to instantly rotate IP identity on HTTP 429 rate limit",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=1.2,
        help="minimum seconds between API requests (default: 1.2)",
    )
    parser.add_argument(
        "--no-wait-cooldown",
        dest="auto_wait_cooldown",
        action="store_false",
        default=True,
        help="do not automatically wait for CDN rate limit cooldown (default: auto-waits cooldown and resumes)",
    )
    parser.add_argument(
        "--tier-partitioned",
        action="store_true",
        help="partition output CSV storage by tier directory (e.g. data/tiers/gold/players/)",
    )


class MatchDownloadPipeline:
    """Unified pipeline for fetching timelines, extracting observations, and writing CSVs."""

    def __init__(
        self,
        output_dir: Path | str = "data",
        tier_partitioned: bool = False,
        request_interval: float = 1.2,
        proxy: str | None = None,
        tor_control_port: int | None = None,
        auto_wait_cooldown: bool = True,
        tft_set: str = "TFTSet18",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.tier_partitioned = tier_partitioned
        self.tft_set = tft_set
        self.auto_wait_cooldown = auto_wait_cooldown
        self.tor_control_port = tor_control_port

        if proxy:
            setup_network_proxy(proxy)

        self.client = MetaTftClient(
            minimum_request_interval_seconds=request_interval,
            retry_count=5,
        )
        self.writer = PlayerCsvWriter(self.output_dir, tier_partitioned=tier_partitioned)
        self.seen_game_ids = get_existing_match_ids(self.output_dir)
        self.written_count = 0
        self.skipped_count = 0
        self.error_count = 0

    def download_match(
        self,
        match_id: str,
        region: str,
        focal_player: str,
        focal_tier: str | None = None,
        tier_category: str | None = None,
        focal_rating_numeric: int | None = None,
        avg_match_rating_numeric: int | None = None,
        portal: str | None = None,
    ) -> list[RoundObservation]:
        """Fetch timeline, extract PvP observations, and write to disk with automatic 429 recovery."""
        if match_id in self.seen_game_ids:
            self.skipped_count += 1
            return []

        while True:
            try:
                timeline = self.client.fetch_game_timeline(
                    match_id=match_id,
                    region=region,
                    tft_set=self.tft_set,
                )
                break
            except MetaTftRequestError as err:
                if "429" in str(err) or (hasattr(err, "status_code") and err.status_code == 429):
                    if self.tor_control_port:
                        tqdm.write(f"\n[!] HTTP 429 rate-limited on {match_id}. Rotating Tor IP identity...")
                        rotated = rotate_tor_identity(control_port=self.tor_control_port)
                        if rotated:
                            tqdm.write("    --> Tor circuit rotated successfully. Retrying immediately.")
                            time.sleep(1.0)
                            continue
                    if self.auto_wait_cooldown:
                        tqdm.write(f"\n[!] Rate-limited on {match_id}. Waiting 30s cooldown...")
                        time.sleep(30.0)
                        continue
                self.error_count += 1
                return []
            except Exception as e:
                logger.debug("Error fetching match %s: %s", match_id, e)
                self.error_count += 1
                return []

        try:
            observations = extract_pvp_rounds(
                timeline=timeline,
                focal_player=focal_player,
                tft_set=self.tft_set,
                focal_tier=focal_tier,
                tier_category=tier_category,
                focal_rating_numeric=focal_rating_numeric,
                avg_match_rating_numeric=avg_match_rating_numeric,
                portal=portal,
            )
        except (TimelineValidationError, ValueError) as err:
            logger.debug("Validation error on match %s: %s", match_id, err)
            self.error_count += 1
            return []

        if not observations:
            self.skipped_count += 1
            return []

        for obs in observations:
            self.writer.write_round_observation(obs)

        self.seen_game_ids.add(match_id)
        self.written_count += 1
        return observations
