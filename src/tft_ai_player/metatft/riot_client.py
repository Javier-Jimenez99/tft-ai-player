"""Official Riot Games TFT League API client for direct ELO / tier player retrieval."""

from __future__ import annotations

import json
import logging
import time
from typing import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .client import LeaderboardPlayer

logger = logging.getLogger(__name__)

REGION_TO_CLUSTER: Mapping[str, str] = {
    "na1": "americas",
    "br1": "americas",
    "la1": "americas",
    "la2": "americas",
    "euw1": "europe",
    "eun1": "europe",
    "tr1": "europe",
    "ru": "europe",
    "kr": "asia",
    "jp1": "asia",
    "oc1": "sea",
    "ph2": "sea",
    "sg2": "sea",
    "th2": "sea",
    "tw2": "sea",
    "vn2": "sea",
}


class RiotTftClient:
    """Lightweight client to query Riot Games official TFT League & Account APIs."""

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 15.0,
        rate_limit_delay_seconds: float = 1.2,
    ) -> None:
        self.api_key = api_key.strip()
        self.timeout_seconds = timeout_seconds
        self.rate_limit_delay_seconds = rate_limit_delay_seconds
        self._last_request_at = 0.0

    def _get_json(self, url: str) -> object:
        now = time.monotonic()
        elapsed = now - self._last_request_at
        if elapsed < self.rate_limit_delay_seconds:
            time.sleep(self.rate_limit_delay_seconds - elapsed)
        self._last_request_at = time.monotonic()

        headers = {
            "X-Riot-Token": self.api_key,
            "User-Agent": "Mozilla/5.0 TFT-AI-Player",
            "Accept": "application/json",
        }
        req = Request(url, headers=headers)
        with urlopen(req, timeout=self.timeout_seconds) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def fetch_account_riot_id(self, puuid: str, *, region: str = "na1") -> tuple[str, str] | None:
        """Resolve PUUID to (game_name, tag_line) via Riot Account API."""
        cluster = REGION_TO_CLUSTER.get(region.lower(), "americas")
        url = f"https://{cluster}.api.riotgames.com/riot/account/v1/accounts/by-puuid/{puuid}"
        try:
            data = self._get_json(url)
            if isinstance(data, dict):
                gname = data.get("gameName")
                tag = data.get("tagLine")
                if isinstance(gname, str) and isinstance(tag, str):
                    return gname.strip(), tag.strip()
        except Exception as error:
            logger.warning("Failed to resolve PUUID %s on cluster %s: %s", puuid, cluster, error)
        return None

    def fetch_tier_players(
        self,
        tier: str,
        *,
        region: str = "na1",
        division: str = "I",
        count: int = 20,
    ) -> list[LeaderboardPlayer]:
        """Fetch active players in a specific tier (e.g. IRON, BRONZE, SILVER, GOLD, PLATINUM, EMERALD, DIAMOND, MASTER, GM, CHALLENGER)."""
        tier_upper = tier.strip().upper()
        reg = region.strip().lower()

        if tier_upper == "CHALLENGER":
            url = f"https://{reg}.api.riotgames.com/tft/league/v1/challenger"
        elif tier_upper == "GRANDMASTER":
            url = f"https://{reg}.api.riotgames.com/tft/league/v1/grandmaster"
        elif tier_upper == "MASTER":
            url = f"https://{reg}.api.riotgames.com/tft/league/v1/master"
        else:
            url = f"https://{reg}.api.riotgames.com/tft/league/v1/entries/{tier_upper}/{division}?page=1"

        try:
            payload = self._get_json(url)
        except Exception as error:
            logger.error("Failed to query Riot league for tier %s (%s): %s", tier_upper, reg, error)
            return []

        entries: list[dict] = []
        if isinstance(payload, dict) and "entries" in payload and isinstance(payload["entries"], list):
            entries = payload["entries"]
        elif isinstance(payload, list):
            entries = payload

        players: list[LeaderboardPlayer] = []
        for entry in entries[:count]:
            if not isinstance(entry, dict):
                continue
            puuid = entry.get("puuid")
            if puuid and isinstance(puuid, str):
                riot_id_pair = self.fetch_account_riot_id(puuid, region=reg)
                if riot_id_pair is not None:
                    gname, tag = riot_id_pair
                    players.append(
                        LeaderboardPlayer(
                            region=reg,
                            game_name=gname,
                            tag_line=tag,
                            player_id=None,
                        )
                    )
        return players
