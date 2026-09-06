"""Extraction of leakage-safe PVP observations from MetaTFT timelines."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from .models import RoundObservation, RoundOutcome, normalize_tier


class TimelineValidationError(ValueError):
    """Raised when a timeline does not contain the expected stage data."""


def parse_stage_data(timeline: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Parse the stringified `stage_data` collection supplied by MetaTFT."""

    raw_stage_data = timeline.get("stage_data")
    if isinstance(raw_stage_data, str):
        try:
            stage_data = json.loads(raw_stage_data)
        except json.JSONDecodeError as error:
            raise TimelineValidationError("stage_data is not valid JSON") from error
    else:
        stage_data = raw_stage_data

    if not isinstance(stage_data, list):
        raise TimelineValidationError("stage_data must decode to a list of snapshots")
    if not all(isinstance(snapshot, Mapping) for snapshot in stage_data):
        raise TimelineValidationError("stage_data contains a non-object snapshot")

    return list(stage_data)


def extract_pvp_rounds(
    timeline: Mapping[str, Any],
    *,
    match_id: str,
    tft_set: str,
    game_version: str,
    focal_player: str | None = None,
    focal_tier: str | None = None,
    focal_rating_numeric: int | None = None,
    avg_match_rating: str | None = None,
    avg_match_rating_numeric: int | None = None,
    focal_augments: Sequence[str] | None = None,
) -> list[RoundObservation]:
    """Extract valid PVP rounds without copying post-combat data into features."""

    if not match_id:
        raise ValueError("match_id is required")
    if not tft_set:
        raise ValueError("tft_set is required")
    if not game_version:
        raise ValueError("game_version is required")

    observations: list[RoundObservation] = []
    game_datetime = _format_datetime(timeline.get("datetime"))
    client_version = _text_at(timeline, "gep_internal", "version_info", "local_version")
    schema_version = _text_at(timeline, "metadata", "manifest_version") or _text(
        timeline.get("data_version")
    )
    portal = _text(timeline.get("portal"))
    default_focal_player = focal_player or _text(timeline.get("summoner_name"))
    effective_focal_tier = focal_tier or _text(timeline.get("summoner_tier"))

    for snapshot in parse_stage_data(timeline):
        match_info = _mapping(snapshot.get("match_info"))
        round_type = _mapping(match_info.get("round_type"))
        round_type_val = _text(round_type.get("type")) or _text(round_type.get("name"))
        if not round_type_val or round_type_val.upper() != "PVP":
            continue

        stage = _text(round_type.get("stage"))
        snapshot_focal_player = default_focal_player or _text_at(snapshot, "me", "summoner_name")
        if stage is None or snapshot_focal_player is None:
            continue

        outcome = _outcome_for(snapshot_focal_player, match_info)
        opponent = _text_at(match_info, "opponent", "name")
        player_board, opponent_board = _matchup_boards(snapshot)
        if outcome is None or opponent is None or player_board is None or opponent_board is None:
            continue

        metatft_win_prob = _float_at(snapshot, "winrate_info", "model_data", "prediction")

        round_start_players = _mapping(_mapping(snapshot.get("round_start_health")).get("player_status"))
        focal_status = _lookup_player_map(round_start_players, snapshot_focal_player)
        opponent_status = _lookup_player_map(round_start_players, opponent)
        me = _mapping(snapshot.get("me"))

        focal_health = _int(focal_status.get("health"))
        focal_level = _int(focal_status.get("xp"))
        focal_gold = _int(me.get("gold"))
        if focal_gold is None:
            focal_gold = _int(focal_status.get("gold"))
        if focal_level is None:
            focal_level = _int(_mapping(me.get("xp")).get("level"))

        opponent_health = _int(opponent_status.get("health"))
        opponent_level = _int(opponent_status.get("xp"))

        extracted_focal_augments = _extract_augments(snapshot, snapshot_focal_player, is_focal=True)
        if not extracted_focal_augments and focal_augments:
            stage_tuple = _parse_stage_tuple(stage)
            if stage_tuple is not None:
                if stage_tuple < (2, 1):
                    active_count = 0
                elif stage_tuple < (3, 2):
                    active_count = 1
                elif stage_tuple < (4, 2):
                    active_count = 2
                else:
                    active_count = 3
                extracted_focal_augments = list(focal_augments[:active_count])
            else:
                extracted_focal_augments = list(focal_augments)

        opponent_augments = _extract_augments(snapshot, opponent, is_focal=False)

        observations.append(
            RoundObservation(
                match_id=match_id,
                game_datetime=game_datetime,
                round_stage=stage,
                round_type="PVP",
                tft_set=tft_set,
                game_version=game_version,
                game_client_version=client_version,
                timeline_schema_version=schema_version,
                portal=portal or _text(snapshot.get("portal")),
                focal_player=snapshot_focal_player,
                focal_tier=effective_focal_tier,
                focal_rating_numeric=focal_rating_numeric,
                avg_match_rating=avg_match_rating,
                avg_match_rating_numeric=avg_match_rating_numeric,
                focal_health=focal_health,
                focal_level=focal_level,
                focal_gold=focal_gold,
                focal_augments=extracted_focal_augments,
                opponent=opponent,
                opponent_health=opponent_health,
                opponent_level=opponent_level,
                opponent_augments=opponent_augments,
                outcome=outcome,
                metatft_win_prob=metatft_win_prob,
                tier_category=normalize_tier(effective_focal_tier or avg_match_rating),
                input_state=_precombat_state(
                    player_board=player_board,
                    opponent_board=opponent_board,
                ),
            )
        )

    return observations


def _parse_stage_tuple(stage: str | None) -> tuple[int, int] | None:
    if not stage or not isinstance(stage, str):
        return None
    parts = stage.strip().split("-")
    if len(parts) == 2:
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return None
    return None


def _format_datetime(value: object) -> str | None:
    if isinstance(value, int) and not isinstance(value, bool):
        try:
            dt = datetime.fromtimestamp(value / 1000, timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except (ValueError, OverflowError, OSError):
            return None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _precombat_state(
    *,
    player_board: Sequence[Any],
    opponent_board: Sequence[Any],
) -> dict[str, Any]:
    return {
        "focal_board": _clean_board(player_board),
        "opponent_board": _clean_board(opponent_board),
    }


def _lookup_player_map(mapping: Mapping[str, Any], player_name: str) -> Mapping[str, Any]:
    if not mapping or not player_name:
        return {}
    exact = mapping.get(player_name)
    if isinstance(exact, Mapping):
        return exact
    player_lower = player_name.lower()
    for key, value in mapping.items():
        if isinstance(key, str) and key.lower() == player_lower and isinstance(value, Mapping):
            return value
    return {}


def _extract_augments(snapshot: Mapping[str, Any], player_name: str, *, is_focal: bool) -> list[str]:
    augments_raw = None
    aug_dict = snapshot.get("augments")
    if isinstance(aug_dict, Mapping):
        augments_raw = aug_dict.get(player_name)
        if not augments_raw and player_name:
            player_lower = player_name.lower()
            for key, val in aug_dict.items():
                if isinstance(key, str) and key.lower() == player_lower:
                    augments_raw = val
                    break
    elif isinstance(aug_dict, Sequence) and not isinstance(aug_dict, (str, bytes)) and is_focal:
        augments_raw = aug_dict

    if not augments_raw and is_focal:
        me = snapshot.get("me")
        if isinstance(me, Mapping):
            augments_raw = me.get("augments")

    if not augments_raw:
        ow_aug = snapshot.get("overwolf_augments")
        if isinstance(ow_aug, Mapping):
            augments_raw = ow_aug.get(player_name)
            if not augments_raw and player_name:
                player_lower = player_name.lower()
                for key, val in ow_aug.items():
                    if isinstance(key, str) and key.lower() == player_lower:
                        augments_raw = val
                        break

    if not augments_raw:
        model_data = _mapping(_mapping(snapshot.get("winrate_info")).get("model_data"))
        p_data = _mapping(model_data.get("player" if is_focal else "opponent"))
        augments_raw = p_data.get("board_encoded_augments")

    result: list[str] = []
    if isinstance(augments_raw, Sequence) and not isinstance(augments_raw, (str, bytes)):
        for item in augments_raw:
            item_text = _text(item)
            if item_text:
                result.append(item_text)
            elif isinstance(item, Mapping):
                name = _text(item.get("name")) or _text(item.get("apiName")) or _text(item.get("id"))
                if name:
                    result.append(name)
    return result


def _clean_board(raw_units: Sequence[Any]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for raw_unit in raw_units:
        if not isinstance(raw_unit, Mapping):
            continue
        unit_id = _text(raw_unit.get("unit")) or _text(raw_unit.get("apiName"))
        if not unit_id:
            continue
        tier = _int(raw_unit.get("tier")) or 1
        loc = _text(raw_unit.get("loc")) or ""
        items_raw = raw_unit.get("items")
        if not isinstance(items_raw, Sequence) or isinstance(items_raw, (str, bytes)):
            items_raw = raw_unit.get("item_ids")

        items: list[str] = []
        if isinstance(items_raw, Sequence) and not isinstance(items_raw, (str, bytes)):
            for item in items_raw:
                item_text = _text(item)
                if item_text:
                    items.append(item_text)

        cleaned.append({
            "unit": unit_id,
            "tier": tier,
            "loc": loc,
            "items": items,
        })
    return cleaned


def _outcome_for(player_name: str, match_info: Mapping[str, Any]) -> RoundOutcome | None:
    outcomes = _mapping(match_info.get("round_outcome"))
    outcome_record = _mapping(outcomes.get(player_name))
    if not outcome_record and player_name:
        player_lower = player_name.lower()
        for key, value in outcomes.items():
            if isinstance(key, str) and key.lower() == player_lower and isinstance(value, Mapping):
                outcome_record = value
                break
    outcome = _text(outcome_record.get("outcome"))
    if outcome:
        outcome_lower = outcome.lower()
        if outcome_lower in {"victory", "defeat"}:
            return outcome_lower
    return None


def _matchup_boards(snapshot: Mapping[str, Any]) -> tuple[Sequence[Any] | None, Sequence[Any] | None]:
    matchups = _mapping(snapshot.get("matchup_boards"))
    player_board = matchups.get("player_board")
    opponent_board = matchups.get("opponent_board")
    if not _is_non_empty_sequence(player_board) or not _is_non_empty_sequence(opponent_board):
        return None, None
    return player_board, opponent_board


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _is_non_empty_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and bool(value)


def _text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    if isinstance(value, float):
        return int(value)
    return None


def _text_at(value: Mapping[str, Any], *keys: str) -> str | None:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return _text(current)


def _float_at(value: Mapping[str, Any], *keys: str) -> float | None:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    if isinstance(current, (float, int)) and not isinstance(current, bool):
        return round(float(current), 4)
    if isinstance(current, str):
        try:
            return round(float(current.strip()), 4)
        except ValueError:
            return None
    return None