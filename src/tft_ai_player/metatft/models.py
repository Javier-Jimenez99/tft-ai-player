"""Models for MetaTFT match discovery and timeline availability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class MatchCandidate:
    """A match discovered through a player's MetaTFT profile."""

    riot_match_id: str
    tft_set: str
    game_version: str | None
    match_data_url: str | None
    match_timestamp: int | None
    queue_id: int | None
    focal_tier: str | None = None
    focal_rating_numeric: int | None = None
    avg_match_rating: str | None = None
    avg_match_rating_numeric: int | None = None

    @classmethod
    def from_profile_record(cls, record: dict[str, Any]) -> MatchCandidate | None:
        """Create a candidate when the profile record has a stable match ID."""

        match_id = record.get("riot_match_id")
        tft_set = record.get("tft_set")
        if not isinstance(match_id, str) or not match_id:
            return None
        if not isinstance(tft_set, str) or not tft_set:
            return None

        summary = record.get("summary")
        summary_dict = summary if isinstance(summary, dict) else {}

        focal_tier = _optional_text(summary_dict.get("player_rating")) or _optional_text(record.get("player_rating"))
        focal_rating_numeric = _optional_int(summary_dict.get("player_rating_numeric")) or _optional_int(record.get("player_rating_numeric"))
        avg_match_rating = _optional_text(record.get("avg_rating")) or _optional_text(summary_dict.get("avg_rating"))
        avg_match_rating_numeric = _optional_int(record.get("avg_rating_numeric")) or _optional_int(summary_dict.get("avg_rating_numeric"))

        return cls(
            riot_match_id=match_id,
            tft_set=tft_set,
            game_version=_optional_text(record.get("patch")),
            match_data_url=_optional_text(record.get("match_data_url")),
            match_timestamp=_optional_int(record.get("match_timestamp")),
            queue_id=_optional_int(record.get("queue_id")),
            focal_tier=focal_tier,
            focal_rating_numeric=focal_rating_numeric,
            avg_match_rating=avg_match_rating,
            avg_match_rating_numeric=avg_match_rating_numeric,
        )


@dataclass(frozen=True, slots=True)
class LeaderboardPlayer:
    """A leaderboard player whose profile can be used to discover tracked games."""

    region: str
    game_name: str
    tag_line: str
    player_id: str | None

    @property
    def riot_id(self) -> str:
        """Return the Riot ID used as the collection-source label."""

        return f"{self.game_name}#{self.tag_line}"

    @classmethod
    def from_leaderboard_record(cls, record: dict[str, Any]) -> LeaderboardPlayer | None:
        """Create a player only when MetaTFT reports tracked app matches."""

        region = _optional_text(record.get("summoner_region"))
        riot_id = _optional_text(record.get("riot_id"))
        stats = record.get("stats")
        if region is None or riot_id is None or not isinstance(stats, dict) or stats.get("appMatches") is not True:
            return None

        game_name, separator, tag_line = riot_id.rpartition("#")
        game_name = game_name.strip()
        tag_line = tag_line.strip()
        if not separator or not game_name or not tag_line:
            return None

        return cls(
            region=region,
            game_name=game_name,
            tag_line=tag_line,
            player_id=_optional_text(record.get("player_id")),
        )


@dataclass(frozen=True, slots=True)
class TrackedTimelineCandidate:
    """A MetaTFT app-recorded match with a directly usable timeline URL."""

    app_match_uuid: str
    match_id_ow: str
    timeline_url: str
    tft_set: str
    game_version: str | None
    riot_match_id: str | None
    created_timestamp: int | None
    player_id: int | None
    focal_tier: str | None = None
    focal_rating_numeric: int | None = None
    avg_match_rating: str | None = None
    avg_match_rating_numeric: int | None = None
    queue_id: int | None = None

    @classmethod
    def from_app_match_record(
        cls,
        record: dict[str, Any],
        *,
        tft_set: str,
        profile_match: MatchCandidate | None = None,
    ) -> TrackedTimelineCandidate | None:
        """Create a candidate from an app match and optional exact Riot-ID match."""

        app_match_uuid = _optional_text(record.get("uuid"))
        match_id_ow = _optional_text(record.get("match_id_ow"))
        timeline_url = _optional_text(record.get("match_data_url"))
        if app_match_uuid is None or match_id_ow is None or timeline_url is None:
            return None

        return cls(
            app_match_uuid=app_match_uuid,
            match_id_ow=match_id_ow,
            timeline_url=timeline_url,
            tft_set=tft_set,
            game_version=profile_match.game_version if profile_match else None,
            riot_match_id=profile_match.riot_match_id if profile_match else None,
            created_timestamp=_optional_int(record.get("created_timestamp")),
            player_id=_optional_int(record.get("player_id")),
            focal_tier=profile_match.focal_tier if profile_match else None,
            focal_rating_numeric=profile_match.focal_rating_numeric if profile_match else None,
            avg_match_rating=profile_match.avg_match_rating if profile_match else None,
            avg_match_rating_numeric=profile_match.avg_match_rating_numeric if profile_match else None,
            queue_id=profile_match.queue_id if profile_match else None,
        )


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None