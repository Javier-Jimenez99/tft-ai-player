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

    def blacklist_path(self) -> Path:
        """Return the destination path for the blacklist file."""

        return self.root / "blacklisted_games.txt"

    def player_blacklist_path(self) -> Path:
        """Return the destination path for the blacklisted players file."""

        return self.root / "blacklisted_players.txt"

    def load_blacklist(self) -> set[str]:
        """Return set of match IDs recorded in the blacklist."""

        blacklisted: set[str] = set()
        for filename in ("blacklisted_games.txt", "blacklist.txt"):
            path = self.root / filename
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as file:
                    for line in file:
                        stripped = line.strip()
                        if not stripped or stripped.startswith("#"):
                            continue
                        match_id = stripped.split()[0].strip()
                        if match_id:
                            blacklisted.add(match_id)
            except Exception:
                continue
        return blacklisted

    def load_player_blacklist(self) -> set[str]:
        """Return set of player Riot IDs recorded in the player blacklist."""

        blacklisted: set[str] = set()
        for filename in ("blacklisted_players.txt", "blacklist_players.txt"):
            path = self.root / filename
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as file:
                    for line in file:
                        stripped = line.strip()
                        if not stripped or stripped.startswith("#"):
                            continue
                        parts = stripped.split("\t", 1)
                        player_id = parts[0].strip()
                        if player_id:
                            blacklisted.add(player_id)
            except Exception:
                continue
        return blacklisted

    def add_to_blacklist(self, match_id: str, reason: str | None = None) -> None:
        """Add a match ID to the persistent blacklist if not already present."""

        cleaned_match_id = match_id.strip()
        if not cleaned_match_id:
            return

        if cleaned_match_id in self.load_blacklist():
            return

        path = self.blacklist_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = (
            f"{cleaned_match_id}\t# {reason}\n"
            if reason
            else f"{cleaned_match_id}\n"
        )
        with path.open("a", encoding="utf-8") as file:
            file.write(entry)

    def add_player_to_blacklist(self, riot_id: str, reason: str | None = None) -> None:
        """Add a player Riot ID to the persistent player blacklist if not already present."""

        cleaned_riot_id = riot_id.strip()
        if not cleaned_riot_id:
            return

        if cleaned_riot_id in self.load_player_blacklist():
            return

        path = self.player_blacklist_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = (
            f"{cleaned_riot_id}\t# {reason}\n"
            if reason
            else f"{cleaned_riot_id}\n"
        )
        with path.open("a", encoding="utf-8") as file:
            file.write(entry)

    def is_blacklisted(self, match_id: str) -> bool:
        """Check if a match ID is currently blacklisted."""

        return match_id.strip() in self.load_blacklist()

    def is_player_blacklisted(self, riot_id: str) -> bool:
        """Check if a player Riot ID is currently blacklisted."""

        return riot_id.strip() in self.load_player_blacklist()

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
        if not observations:
            return None
        return self.write_player_game(
            observations,
            collected_from_riot_id=collected_from_riot_id or observations[0].focal_player,
            collected_from_region=collected_from_region or "unknown",
            match_id_ow=match_id_ow,
        )


GameCsvWriter = PlayerCsvWriter


def _safe_player_slug(region: str, riot_id: str) -> str:
    """Create a safe filesystem filename from region and Riot ID."""

    clean_region = _safe_segment(region.lower() if region else "unknown")
    clean_riot_id = _safe_segment(riot_id)
    return f"{clean_region}_{clean_riot_id}"


def _safe_segment(value: str | None) -> str:
    if value is None:
        return "unknown"
    normalized = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in str(value)
    )
    collapsed = re.sub(r"_+", "_", normalized).strip("_")
    return collapsed or "unknown"