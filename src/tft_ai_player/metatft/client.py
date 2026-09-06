"""Small, rate-limited client for the public MetaTFT endpoints."""

from __future__ import annotations

import http.client
import json
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .models import LeaderboardPlayer, MatchCandidate, TrackedTimelineCandidate, _optional_int

import gzip
import socket

JsonFetcher = Callable[[str], Mapping[str, Any]]


class MetaTftRequestError(RuntimeError):
    """Raised when a MetaTFT request cannot be completed or decoded."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


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
        socket.setdefaulttimeout(timeout_seconds)

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
        allowed_queue_ids: Sequence[int] | None = (1100,),
        max_timestamp_delta_ms: int = 3600 * 1000,
    ) -> list[TrackedTimelineCandidate]:
        """Return app-recorded matches joined with Riot match metadata by timestamp.

        By default, only standard Ranked TFT matches (queue_id=1100) are returned,
        filtering out Double Up (1160), Normal (1090), and other modes. Pass
        `allowed_queue_ids=None` to keep all queues.
        """

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

            if allowed_queue_ids is not None:
                if best_match is None or best_match.queue_id not in allowed_queue_ids:
                    continue

            candidate = TrackedTimelineCandidate.from_app_match_record(
                record,
                tft_set=tft_set,
                profile_match=best_match,
            )
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    @staticmethod
    def extract_lobby_participants(timeline: Mapping[str, Any]) -> list[tuple[str, str]]:
        """Extract unique (summoner_name, tag_line) tuples of all players in this timeline lobby."""
        participants: dict[str, tuple[str, str]] = {}
        board_players = timeline.get("board_players")
        if isinstance(board_players, list):
            for entry in board_players:
                if not isinstance(entry, dict):
                    continue
                boards = entry.get("board")
                if isinstance(boards, list):
                    for b in boards:
                        if not isinstance(b, dict):
                            continue
                        summoner = b.get("summoner")
                        tag_line = b.get("tag_line")
                        if isinstance(summoner, str) and isinstance(tag_line, str):
                            summoner = summoner.strip()
                            tag_line = tag_line.strip()
                            if summoner and tag_line:
                                riot_id = f"{summoner}#{tag_line}"
                                if riot_id not in participants:
                                    participants[riot_id] = (summoner, tag_line)

        # Extract from stage_data -> roster -> player_status
        raw_stage = timeline.get("stage_data")
        stage_data: Any = None
        if isinstance(raw_stage, str):
            try:
                stage_data = json.loads(raw_stage)
            except Exception:
                stage_data = None
        elif isinstance(raw_stage, list):
            stage_data = raw_stage

        if isinstance(stage_data, list):
            for r_entry in stage_data:
                if not isinstance(r_entry, dict):
                    continue
                roster = r_entry.get("roster")
                if isinstance(roster, dict):
                    p_status = roster.get("player_status")
                    if isinstance(p_status, dict):
                        for s_name, info in p_status.items():
                            if isinstance(info, dict) and isinstance(s_name, str):
                                t_line = info.get("tag_line")
                                if t_line is not None:
                                    s_clean = s_name.strip()
                                    t_clean = str(t_line).strip()
                                    if s_clean and t_clean:
                                        riot_id = f"{s_clean}#{t_clean}"
                                        if riot_id not in participants:
                                            participants[riot_id] = (s_clean, t_clean)

        # Fallback to focal player if participants still empty
        if not participants:
            focal_name = timeline.get("summoner_name")
            focal_tag = timeline.get("tagline")
            if isinstance(focal_name, str) and isinstance(focal_tag, str):
                focal_name = focal_name.strip()
                focal_tag = focal_tag.strip()
                if focal_name and focal_tag:
                    participants[f"{focal_name}#{focal_tag}"] = (focal_name, focal_tag)

        return list(participants.values())

    @staticmethod
    def extract_player_tier(profile: Mapping[str, Any]) -> str | None:
        """Extract the current ranked tier string (e.g. 'CHALLENGER I 995 LP', 'GOLD IV') from a profile."""
        ranked = profile.get("ranked")
        if isinstance(ranked, dict):
            rating_text = ranked.get("rating_text")
            if isinstance(rating_text, str) and rating_text.strip():
                return rating_text.strip()
        summoner = profile.get("summoner")
        if isinstance(summoner, dict):
            rating = summoner.get("rating")
            if isinstance(rating, str) and rating.strip():
                return rating.strip()
        return None

    def _get_json(self, url: str) -> Mapping[str, Any]:
        if self.json_fetcher is not None:
            return _require_mapping(self.json_fetcher(url), url)

        last_error: Exception | None = None
        last_retry_after: float | None = None
        for attempt in range(self.retry_count + 1):
            self._wait_for_rate_limit()
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json, text/plain, */*",
                "Accept-Encoding": "gzip, deflate",
                "Referer": "https://www.metatft.com/",
                "Origin": "https://www.metatft.com",
            }
            request = Request(url, headers=headers)
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    raw_bytes = response.read()
                    content_encoding = (
                        response.headers.get("Content-Encoding", "").lower()
                        if hasattr(response, "headers") and response.headers
                        else ""
                    )
                    if "gzip" in content_encoding or raw_bytes[:2] == b"\x1f\x8b":
                        raw_bytes = gzip.decompress(raw_bytes)
                    payload = json.loads(raw_bytes.decode("utf-8"))
                return _require_mapping(payload, url)
            except (
                HTTPError,
                URLError,
                TimeoutError,
                ConnectionError,
                OSError,
                http.client.HTTPException,
                json.JSONDecodeError,
            ) as error:
                last_error = error
                if attempt == self.retry_count or (isinstance(error, HTTPError) and error.code in (400, 404)):
                    break
                if isinstance(error, HTTPError) and error.code == 429:
                    retry_after_hdr = (
                        error.headers.get("Retry-After")
                        if hasattr(error, "headers") and error.headers
                        else None
                    )
                    retry_after_val: float | None = None
                    if retry_after_hdr:
                        try:
                            retry_after_val = float(retry_after_hdr)
                        except (ValueError, TypeError):
                            pass
                    last_retry_after = retry_after_val
                    if retry_after_val is not None and retry_after_val > 60.0:
                        # Server or CDN has imposed a long rate limit / lockout; fail fast rather than stalling the process
                        break
                    sleep_seconds = retry_after_val if retry_after_val is not None and retry_after_val > 0 else min(30.0, 2.0 * (2**attempt))
                else:
                    sleep_seconds = min(30.0, 1.5 * (2**attempt))

                time.sleep(sleep_seconds)
                self._last_request_at = time.monotonic()

        raise MetaTftRequestError(f"request failed for {url}: {last_error}", retry_after=last_retry_after) from last_error

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