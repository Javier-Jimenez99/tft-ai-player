"""Small, rate-limited client for the public MetaTFT endpoints."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .models import LeaderboardPlayer, MatchCandidate, TrackedTimelineCandidate, _optional_int

JsonFetcher = Callable[[str], Mapping[str, Any]]


class MetaTftRequestError(RuntimeError):
    """Raised when a MetaTFT request cannot be completed or decoded."""


class MetaTftClient:
    """Fetch MetaTFT API data with bounded retries and pacing."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 20.0,
        minimum_request_interval_seconds: float = 0.5,
        retry_count: int = 2,
        json_fetcher: JsonFetcher | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if minimum_request_interval_seconds < 0:
            raise ValueError("minimum_request_interval_seconds cannot be negative")
        if retry_count < 0:
            raise ValueError("retry_count cannot be negative")

        self.timeout_seconds = timeout_seconds
        self.minimum_request_interval_seconds = minimum_request_interval_seconds
        self.retry_count = retry_count
        self.json_fetcher = json_fetcher
        self._last_request_at = 0.0

    DEFAULT_LEADERBOARD_REGIONS: tuple[str, ...] = (
        "global",
        "kr",
        "na1",
        "euw1",
        "eun1",
        "vn2",
        "la1",
        "la2",
        "br1",
        "oc1",
        "jp1",
    )

    def fetch_leaderboard(
        self,
        *,
        region: str = "global",
        offset: int = 0,
        limit: int = 100,
    ) -> Mapping[str, Any]:
        """Return a page of app-user leaderboard data for a region."""

        if offset < 0:
            raise ValueError("offset cannot be negative")
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")

        query = urlencode(
            {
                "offset": offset,
                "limit": limit,
                "filter_app_user": "true",
                "stat_type": "recent",
            }
        )
        return self._get_json(f"https://api.metatft.com/tft-leaderboard/v2/{region}?{query}")

    def fetch_leaderboard_players(
        self,
        *,
        count: int,
        offset: int = 0,
        regions: Sequence[str] | None = None,
    ) -> list[LeaderboardPlayer]:
        """Return distinct tracked leaderboard players across regions, loading pages as needed."""

        if count <= 0:
            raise ValueError("count must be positive")
        if offset < 0:
            raise ValueError("offset cannot be negative")

        target_regions = regions or self.DEFAULT_LEADERBOARD_REGIONS
        players: list[LeaderboardPlayer] = []
        seen_riot_ids: set[str] = set()

        for region in target_regions:
            page_offset = offset
            while len(players) < count:
                page_limit = min(100, count - len(players))
                try:
                    payload = self.fetch_leaderboard(region=region, offset=page_offset, limit=page_limit)
                except MetaTftRequestError:
                    break
                raw_players = payload.get("data")
                if not isinstance(raw_players, list) or not raw_players:
                    break

                for record in raw_players:
                    if not isinstance(record, dict):
                        continue
                    player = LeaderboardPlayer.from_leaderboard_record(record)
                    if player is None or player.riot_id in seen_riot_ids:
                        continue
                    players.append(player)
                    seen_riot_ids.add(player.riot_id)
                    if len(players) == count:
                        return players

                page_offset += len(raw_players)
                if len(raw_players) < page_limit:
                    break

        return players

    def fetch_profile(
        self,
        *,
        region: str,
        game_name: str,
        tag_line: str,
        tft_set: str,
    ) -> Mapping[str, Any]:
        """Return the complete MetaTFT profile response for one Riot ID."""

        if not all((region, game_name, tag_line, tft_set)):
            raise ValueError("region, game_name, tag_line, and tft_set are required")

        encoded_path = "/".join(
            (
                quote(region.upper(), safe=""),
                quote(game_name, safe=""),
                quote(tag_line, safe=""),
            )
        )
        query = urlencode(
            {
                "source": "full_profile",
                "tft_set": tft_set,
                "include_revival_matches": "true",
            }
        )
        return self._get_json(
            f"https://api.metatft.com/public/profile/lookup_by_riotid/{encoded_path}?{query}"
        )

    def fetch_timeline(self, timeline_url: str) -> Mapping[str, Any]:
        """Fetch a known timeline URL without claiming to resolve match IDs."""

        if not timeline_url:
            raise ValueError("timeline_url is required")
        return self._get_json(timeline_url)

    @staticmethod
    def match_candidates(profile: Mapping[str, Any], *, tft_set: str) -> list[MatchCandidate]:
        """Return all valid profile matches for one TFT set, including incomplete ones."""

        raw_matches = profile.get("matches")
        if not isinstance(raw_matches, list):
            raise ValueError("profile response does not contain a matches list")

        candidates: list[MatchCandidate] = []
        for record in raw_matches:
            if not isinstance(record, dict) or record.get("tft_set") != tft_set:
                continue
            candidate = MatchCandidate.from_profile_record(record)
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    @staticmethod
    def tracked_timeline_candidates(
        profile: Mapping[str, Any],
        *,
        tft_set: str,
        max_timestamp_delta_ms: int = 3600 * 1000,
    ) -> list[TrackedTimelineCandidate]:
        """Return app-recorded matches joined with Riot match metadata by timestamp."""

        raw_app_matches = profile.get("app_matches")
        if raw_app_matches is None:
            return []
        if not isinstance(raw_app_matches, list):
            raise ValueError("profile response does not contain an app_matches list")

        profile_matches = MetaTftClient.match_candidates(profile, tft_set=tft_set)
        candidates: list[TrackedTimelineCandidate] = []
        for record in raw_app_matches:
            if not isinstance(record, dict):
                continue
            created_timestamp = _optional_int(record.get("created_timestamp"))
            best_match: MatchCandidate | None = None
            if created_timestamp is not None:
                best_diff = float("inf")
                for match in profile_matches:
                    if match.match_timestamp is not None:
                        diff = abs(match.match_timestamp - created_timestamp)
                        if diff < best_diff:
                            best_diff = diff
                            best_match = match
                if best_diff > max_timestamp_delta_ms:
                    best_match = None

            candidate = TrackedTimelineCandidate.from_app_match_record(
                record,
                tft_set=tft_set,
                profile_match=best_match,
            )
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def _get_json(self, url: str) -> Mapping[str, Any]:
        if self.json_fetcher is not None:
            return _require_mapping(self.json_fetcher(url), url)

        last_error: Exception | None = None
        for attempt in range(self.retry_count + 1):
            self._wait_for_rate_limit()
            request = Request(url, headers={"User-Agent": "tft-ai-player/0.1"})
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return _require_mapping(payload, url)
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
                last_error = error
                if attempt == self.retry_count:
                    break
                if isinstance(error, HTTPError) and error.code == 429:
                    time.sleep(max(3.0, 2 ** (attempt + 1)))
                else:
                    time.sleep(2**attempt)

        raise MetaTftRequestError(f"request failed for {url}: {last_error}") from last_error

    def _wait_for_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        wait_seconds = self.minimum_request_interval_seconds - elapsed
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        self._last_request_at = time.monotonic()


def _require_mapping(payload: object, url: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise MetaTftRequestError(f"response from {url} is not a JSON object")
    return payload