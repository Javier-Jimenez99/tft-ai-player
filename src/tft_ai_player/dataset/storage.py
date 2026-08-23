"""CSV storage organized by player."""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from pathlib import Path

from .models import RoundObservation


class PlayerCsvWriter:
    """Append PVP observations for each player to one CSV file."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def player_csv_path(self, *, region: str, riot_id: str) -> Path:
        """Return the destination path for a given player's CSV."""

        slug = _safe_player_slug(region, riot_id)
        return self.root / "players" / f"{slug}.csv"

    def existing_match_ids(self, *, region: str, riot_id: str) -> set[str]:
        """Return set of match IDs already present in this player's CSV file."""

        path = self.player_csv_path(region=region, riot_id=riot_id)
        if not path.exists():
            return set()
        match_ids: set[str] = set()
        try:
            with path.open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    match_id = row.get("match_id")
                    if match_id:
                        match_ids.add(match_id)
        except Exception:
            return set()
        return match_ids

    def all_existing_match_ids(self) -> set[str]:
        """Return set of all match IDs across all player CSV files in data/players/."""

        players_dir = self.root / "players"
        if not players_dir.exists():
            return set()
        match_ids: set[str] = set()
        for path in players_dir.glob("*.csv"):
            try:
                with path.open(newline="", encoding="utf-8") as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        match_id = row.get("match_id")
                        if match_id:
                            match_ids.add(match_id)
            except Exception:
                continue
        return match_ids

    def write_player_game(
        self,
        observations: Sequence[RoundObservation],
        *,
        collected_from_riot_id: str,
        collected_from_region: str,
        match_id_ow: str | None = None,
    ) -> Path | None:
        """Append all valid rounds from one game to the player's CSV file."""

        if not observations:
            return None

        path = self.player_csv_path(
            region=collected_from_region,
            riot_id=collected_from_riot_id,
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        records = [
            {
                "collected_from_riot_id": collected_from_riot_id,
                "collected_from_region": collected_from_region,
                "match_id_ow": match_id_ow,
                **observation.to_record(),
            }
            for observation in observations
        ]

        file_exists = path.exists() and path.stat().st_size > 0
        with path.open("a", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(records[0].keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerows(records)

        return path

    # Compatibility alias for single game write
    def write_game(
        self,
        observations: Sequence[RoundObservation],
        *,
        collected_from_riot_id: str | None = None,
        collected_from_region: str | None = None,
        match_id_ow: str | None = None,
    ) -> Path | None:
        return self.write_player_game(
            observations,
            collected_from_riot_id=collected_from_riot_id or observations[0].focal_player,
            collected_from_region=collected_from_region or "unknown",
            match_id_ow=match_id_ow,
        )


GameCsvWriter = PlayerCsvWriter


def _safe_player_slug(region: str, riot_id: str) -> str:
    """Create a safe filesystem filename from region and Riot ID."""

    clean_region = _safe_segment(region.lower())
    clean_riot_id = _safe_segment(riot_id)
    return f"{clean_region}_{clean_riot_id}"


def _safe_segment(value: str) -> str:
    normalized = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )
    collapsed = re.sub(r"_+", "_", normalized).strip("_")
    return collapsed or "unknown"