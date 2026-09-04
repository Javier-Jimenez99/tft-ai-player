"""Data structures shared by the dataset pipeline."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

RoundOutcome = Literal["victory", "defeat"]

KNOWN_TIERS: tuple[str, ...] = (
    "IRON",
    "BRONZE",
    "SILVER",
    "GOLD",
    "PLATINUM",
    "EMERALD",
    "DIAMOND",
    "MASTER",
    "GRANDMASTER",
    "CHALLENGER",
)


def normalize_tier(tier_str: str | None) -> str:
    """Normalize a tier string (e.g. 'CHALLENGER I 500 LP', 'gold', 'Emerald IV') to a standard tier name."""
    if not tier_str or not isinstance(tier_str, str):
        return "UNKNOWN"
    upper = tier_str.strip().upper()
    for tier in KNOWN_TIERS:
        # Check word boundary or substring
        if tier in upper.split() or upper.startswith(tier):
            return tier
    for tier in KNOWN_TIERS:
        if tier in upper:
            return tier
    return "UNKNOWN"


@dataclass(frozen=True, slots=True)
class RoundObservation:
    """A labeled focal-player-versus-opponent PVP round."""

    match_id: str
    game_datetime: str | None
    round_stage: str
    round_type: str
    tft_set: str
    game_version: str
    game_client_version: str | None
    timeline_schema_version: str | None
    portal: str | None
    focal_player: str
    focal_tier: str | None
    focal_rating_numeric: int | None
    avg_match_rating: str | None
    avg_match_rating_numeric: int | None
    focal_health: int | None
    focal_level: int | None
    focal_gold: int | None
    focal_augments: Sequence[str]
    opponent: str
    opponent_health: int | None
    opponent_level: int | None
    opponent_augments: Sequence[str]
    outcome: RoundOutcome
    input_state: dict[str, Any]
    metatft_win_prob: float | None = None
    tier_category: str | None = None

    @property
    def label(self) -> int:
        """Return one for a victory and zero for a defeat."""

        return int(self.outcome == "victory")

    @property
    def observation_id(self) -> str:
        """Return the stable identifier used to deduplicate extracted rows."""

        return f"{self.match_id}:{self.round_stage}:{self.focal_player}"

    @property
    def focal_unit_count(self) -> int:
        """Return the total number of fielded champions for the focal player."""

        return len(self.input_state.get("focal_board", []))

    @property
    def focal_item_count(self) -> int:
        """Return the total number of items equipped across focal player champions."""

        return sum(len(unit.get("items", [])) for unit in self.input_state.get("focal_board", []))

    @property
    def opponent_unit_count(self) -> int:
        """Return the total number of fielded champions for the opponent."""

        return len(self.input_state.get("opponent_board", []))

    @property
    def opponent_item_count(self) -> int:
        """Return the total number of items equipped across opponent champions."""

        return sum(len(unit.get("items", [])) for unit in self.input_state.get("opponent_board", []))

    def to_record(self) -> dict[str, str | int | float | None]:
        """Return a CSV-friendly representation with nested inputs as JSON."""

        effective_tier_category = self.tier_category or normalize_tier(self.focal_tier or self.avg_match_rating)
        return {
            "observation_id": self.observation_id,
            "match_id": self.match_id,
            "game_datetime": self.game_datetime,
            "round_stage": self.round_stage,
            "round_type": self.round_type,
            "tft_set": self.tft_set,
            "game_version": self.game_version,
            "game_client_version": self.game_client_version,
            "timeline_schema_version": self.timeline_schema_version,
            "portal": self.portal,
            "focal_player": self.focal_player,
            "focal_tier": self.focal_tier,
            "tier_category": effective_tier_category,
            "focal_rating_numeric": self.focal_rating_numeric,
            "avg_match_rating": self.avg_match_rating,
            "avg_match_rating_numeric": self.avg_match_rating_numeric,
            "focal_health": self.focal_health,
            "focal_level": self.focal_level,
            "focal_gold": self.focal_gold,
            "focal_augments": ",".join(self.focal_augments),
            "focal_unit_count": self.focal_unit_count,
            "focal_item_count": self.focal_item_count,
            "opponent": self.opponent,
            "opponent_health": self.opponent_health,
            "opponent_level": self.opponent_level,
            "opponent_augments": ",".join(self.opponent_augments),
            "opponent_unit_count": self.opponent_unit_count,
            "opponent_item_count": self.opponent_item_count,
            "outcome": self.outcome,
            "label": self.label,
            "metatft_win_prob": self.metatft_win_prob,
            "input_state_json": json.dumps(
                self.input_state,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
        }