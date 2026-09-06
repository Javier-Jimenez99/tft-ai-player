"""SQLite persistence and state management for the 24/7 continuous TFT collector service.

Designed for long-running execution on low-power devices like Raspberry Pi:
- Uses SQLite Write-Ahead Logging (WAL) for minimal SD card wear and crash-safe ACID transactions.
- Provides fast, indexed lookups for deficit-based tier scheduling and queue management.
- Handles seamless zero-data-loss migration from existing CSV graph and on-disk tier datasets.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .models import normalize_tier

logger = logging.getLogger(__name__)

ALL_STANDARD_TIERS: tuple[str, ...] = (
    "CHALLENGER",
    "GRANDMASTER",
    "MASTER",
    "DIAMOND",
    "EMERALD",
    "PLATINUM",
    "GOLD",
    "SILVER",
    "BRONZE",
    "IRON",
)

TIER_BASE_ELO: dict[str, int] = {
    "IRON": 200,
    "BRONZE": 600,
    "SILVER": 1000,
    "GOLD": 1400,
    "PLATINUM": 1800,
    "EMERALD": 2200,
    "DIAMOND": 2600,
    "MASTER": 3000,
    "GRANDMASTER": 3600,
    "CHALLENGER": 4200,
}

DIV_OFFSET: dict[str, int] = {
    "IV": 0,
    "III": 100,
    "II": 200,
    "I": 300,
}


def parse_elo_to_numeric(tier_str: str | None, tier_fallback: str | None = None) -> int | None:
    """Extract continuous numerical ELO points from a rank string or tier name."""
    s = str(tier_str or tier_fallback or "").upper().strip()
    if not s or s in ("UNKNOWN", "UNRANKED", "NONE"):
        return None

    import re

    # Sort descending by name length so GRANDMASTER is evaluated before MASTER
    for t_name, base in sorted(TIER_BASE_ELO.items(), key=lambda x: len(x[0]), reverse=True):
        if re.search(rf"\b{t_name}\b", s) or t_name in s.split():
            lp_match = re.search(r"(\d+)\s*LP", s)
            lp = int(lp_match.group(1)) if lp_match else 50

            if t_name in ("MASTER", "GRANDMASTER", "CHALLENGER"):
                return base + lp

            div_off = 150
            for div, off in DIV_OFFSET.items():
                if f" {div} " in s or s.endswith(f" {div}") or f" {div}" in s:
                    div_off = off
                    break
            return base + div_off + min(99, lp)
    return None


@dataclass(slots=True)
class DbPlayer:
    """A tracked player in the collector database."""

    riot_id: str
    region: str
    game_name: str
    tag_line: str
    tier: str = "UNKNOWN"
    rank_text: str = "UNRANKED"
    app_matches: int = 0
    depth: int = 0
    is_app_user: bool = False
    scanned: bool = False
    discovered_from: str | None = None
    last_scanned_at: float | None = None


@dataclass(slots=True)
class DbGame:
    """A match candidate in the collector database."""

    match_uuid: str
    timeline_url: str
    focal_player_riot_id: str
    tier: str = "UNKNOWN"
    avg_rating: str | None = None
    tft_set: str = "TFTSet18"
    status: str = "pending"  # pending, downloaded, failed, blacklisted
    downloaded_at: float | None = None
    error_message: str | None = None


class CollectorDb:
    """Thread-safe SQLite manager for continuous collection state."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None,  # Autocommit mode by default, manual transactions when needed
        )
        self.conn.row_factory = sqlite3.Row
        self._setup_pragmas()
        self._create_tables()

    def _setup_pragmas(self) -> None:
        cursor = self.conn.cursor()
        # WAL mode is dramatically faster, reduces SD card writes, and allows concurrent reads
        cursor.execute("PRAGMA journal_mode = WAL;")
        cursor.execute("PRAGMA synchronous = NORMAL;")
        cursor.execute("PRAGMA foreign_keys = ON;")
        cursor.execute("PRAGMA temp_store = MEMORY;")
        cursor.execute("PRAGMA cache_size = -16000;")  # ~16MB cache
        cursor.close()

    def _create_tables(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute("BEGIN IMMEDIATE;")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS players (
                riot_id TEXT PRIMARY KEY,
                region TEXT NOT NULL,
                game_name TEXT NOT NULL,
                tag_line TEXT NOT NULL,
                tier TEXT NOT NULL DEFAULT 'UNKNOWN',
                rank_text TEXT NOT NULL DEFAULT 'UNRANKED',
                app_matches INTEGER NOT NULL DEFAULT 0,
                depth INTEGER NOT NULL DEFAULT 0,
                is_app_user INTEGER NOT NULL DEFAULT 0,
                scanned INTEGER NOT NULL DEFAULT 0,
                discovered_from TEXT,
                last_scanned_at REAL
            );
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_players_tier_scanned
            ON players (tier, scanned, is_app_user, depth);
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_players_aging
            ON players (is_app_user, last_scanned_at)
            WHERE is_app_user = 1;
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS games (
                match_uuid TEXT PRIMARY KEY,
                timeline_url TEXT NOT NULL,
                focal_player_riot_id TEXT NOT NULL,
                tier TEXT NOT NULL DEFAULT 'UNKNOWN',
                avg_rating TEXT,
                tft_set TEXT NOT NULL DEFAULT 'TFTSet18',
                status TEXT NOT NULL DEFAULT 'pending',
                downloaded_at REAL,
                error_message TEXT
            );
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_games_tier_status
            ON games (tier, status);
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS downloaded_matches (
                match_uuid TEXT PRIMARY KEY,
                tier TEXT NOT NULL DEFAULT 'UNKNOWN',
                focal_player TEXT,
                downloaded_at REAL NOT NULL,
                elo_rating INTEGER,
                placement INTEGER,
                is_top1 INTEGER DEFAULT 0
            );
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_downloaded_tier
            ON downloaded_matches (tier);
            """
        )
        # Migrate older tables if columns missing
        for col_name, col_type in (
            ("elo_rating", "INTEGER"),
            ("placement", "INTEGER"),
            ("is_top1", "INTEGER DEFAULT 0"),
        ):
            try:
                cursor.execute(f"ALTER TABLE downloaded_matches ADD COLUMN {col_name} {col_type};")
            except Exception:
                pass

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_downloaded_elo
            ON downloaded_matches (elo_rating);
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_downloaded_top1
            ON downloaded_matches (is_top1);
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            """
        )
        cursor.execute("COMMIT;")
        cursor.close()

    def close(self) -> None:
        """Close database connection cleanly."""
        try:
            self.conn.execute("PRAGMA optimize;")
            self.conn.close()
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Initial Import & Zero-Data-Loss Migration
    # -------------------------------------------------------------------------

    def import_existing_disk_matches(self, root_dir: str | Path) -> int:
        """Scan data/tiers/ and data/players/ and record all already-downloaded match UUIDs."""
        from .storage import PlayerCsvWriter

        writer = PlayerCsvWriter(root_dir, tier_partitioned=True)
        tier_matches = writer.existing_match_ids_by_tier()
        now = time.time()

        records: list[tuple[str, str, float]] = []
        for tier_norm, match_ids in tier_matches.items():
            for mid in match_ids:
                if mid and mid.strip():
                    records.append((mid.strip(), tier_norm, now))

        if not records:
            return 0

        cursor = self.conn.cursor()
        cursor.execute("BEGIN IMMEDIATE;")
        cursor.executemany(
            """
            INSERT OR IGNORE INTO downloaded_matches (match_uuid, tier, focal_player, downloaded_at)
            VALUES (?, ?, NULL, ?);
            """,
            records,
        )
        # Update any corresponding entries in games table to 'downloaded'
        cursor.executemany(
            """
            UPDATE games
            SET status = 'downloaded', downloaded_at = ?
            WHERE match_uuid = ? AND status != 'downloaded';
            """,
            [(r[2], r[0]) for r in records],
        )
        cursor.execute("COMMIT;")
        cursor.close()

        logger.info("Imported %d existing on-disk match IDs into database", len(records))
        return len(records)

    def import_existing_graph(self, manifest_dir: str | Path) -> tuple[int, int]:
        """Import legacy players.csv and games.csv without overwriting existing progress."""
        m_dir = Path(manifest_dir)
        if not m_dir.exists():
            return 0, 0

        players_path = m_dir / "players.csv"
        games_path = m_dir / "games.csv"

        imported_players = 0
        imported_games = 0

        cursor = self.conn.cursor()
        cursor.execute("BEGIN IMMEDIATE;")

        if players_path.exists():
            with players_path.open("r", newline="", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                player_rows: list[tuple] = []
                for row in reader:
                    rid = row.get("riot_id")
                    if not rid:
                        continue
                    raw_rank = row.get("rank_text", "UNRANKED")
                    raw_tier = row.get("tier", "UNKNOWN")
                    tier_norm = (
                        normalize_tier(raw_rank)
                        if raw_rank and raw_rank != "UNRANKED"
                        else normalize_tier(raw_tier)
                    )
                    scanned = 1 if row.get("scanned") in ("1", "True", "true") else 0
                    app_matches = int(row.get("app_matches", 0))
                    is_app_user = 1 if row.get("is_app_user") in ("1", "True", "true") or app_matches > 0 else 0
                    player_rows.append(
                        (
                            rid,
                            row.get("region", "na1").lower(),
                            row.get("game_name", ""),
                            row.get("tag_line", ""),
                            tier_norm,
                            raw_rank,
                            app_matches,
                            int(row.get("depth", 0)),
                            is_app_user,
                            scanned,
                            row.get("discovered_from") or None,
                            time.time() if scanned else None,
                        )
                    )
                if player_rows:
                    cursor.executemany(
                        """
                        INSERT OR IGNORE INTO players (
                            riot_id, region, game_name, tag_line, tier, rank_text,
                            app_matches, depth, is_app_user, scanned, discovered_from, last_scanned_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        player_rows,
                    )
                    imported_players = len(player_rows)

        if games_path.exists():
            with games_path.open("r", newline="", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                game_rows: list[tuple] = []
                for row in reader:
                    muuid = row.get("match_uuid")
                    t_url = row.get("timeline_url")
                    if not muuid or not t_url:
                        continue
                    tier_norm = normalize_tier(row.get("avg_rating") or row.get("tier") or "UNKNOWN")
                    game_rows.append(
                        (
                            muuid,
                            t_url,
                            row.get("focal_player_riot_id", ""),
                            tier_norm,
                            row.get("avg_rating") or None,
                            row.get("tft_set", "TFTSet18"),
                            "pending",
                        )
                    )
                if game_rows:
                    cursor.executemany(
                        """
                        INSERT OR IGNORE INTO games (
                            match_uuid, timeline_url, focal_player_riot_id, tier, avg_rating, tft_set, status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?);
                        """,
                        game_rows,
                    )
                    imported_games = len(game_rows)

        cursor.execute("COMMIT;")
        cursor.close()

        logger.info(
            "Imported legacy graph: %d players, %d games",
            imported_players,
            imported_games,
        )
        return imported_players, imported_games

    # -------------------------------------------------------------------------
    # Distribution Metrics & Deficit Calculations
    # -------------------------------------------------------------------------

    def get_tier_counts(self) -> dict[str, dict[str, int]]:
        """Return match counts grouped by tier.

        Returns:
            dict with:
                "downloaded": {tier: count}
                "pending": {tier: count}
                "unscanned_players": {tier: count}
                "app_users": {tier: count}
        """
        cursor = self.conn.cursor()

        # 1. Downloaded matches
        cursor.execute("SELECT tier, COUNT(*) as cnt FROM downloaded_matches GROUP BY tier;")
        downloaded: dict[str, int] = {row["tier"]: row["cnt"] for row in cursor.fetchall()}

        # 2. Pending games
        cursor.execute("SELECT tier, COUNT(*) as cnt FROM games WHERE status = 'pending' GROUP BY tier;")
        pending: dict[str, int] = {row["tier"]: row["cnt"] for row in cursor.fetchall()}

        # 3. Unscanned players
        cursor.execute("SELECT tier, COUNT(*) as cnt FROM players WHERE scanned = 0 GROUP BY tier;")
        unscanned: dict[str, int] = {row["tier"]: row["cnt"] for row in cursor.fetchall()}

        # 4. App users
        cursor.execute("SELECT tier, COUNT(*) as cnt FROM players WHERE is_app_user = 1 GROUP BY tier;")
        app_users: dict[str, int] = {row["tier"]: row["cnt"] for row in cursor.fetchall()}

        cursor.close()
        return {
            "downloaded": downloaded,
            "pending": pending,
            "unscanned_players": unscanned,
            "app_users": app_users,
        }

    def total_downloaded_count(self) -> int:
        """Return total number of downloaded matches in database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM downloaded_matches;")
        val = cursor.fetchone()[0]
        cursor.close()
        return int(val)

    # -------------------------------------------------------------------------
    # Player Management & Discovery
    # -------------------------------------------------------------------------

    def add_player(
        self,
        *,
        riot_id: str,
        region: str,
        game_name: str,
        tag_line: str,
        tier: str = "UNKNOWN",
        rank_text: str = "UNRANKED",
        app_matches: int = 0,
        depth: int = 0,
        is_app_user: bool = False,
        scanned: bool = False,
        discovered_from: str | None = None,
    ) -> bool:
        """Add or update a player node."""
        tier_norm = (
            normalize_tier(rank_text)
            if rank_text and rank_text != "UNRANKED"
            else normalize_tier(tier)
        )
        is_app = 1 if (is_app_user or app_matches > 0) else 0
        scanned_int = 1 if scanned else 0
        now = time.time() if scanned else None

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO players (
                riot_id, region, game_name, tag_line, tier, rank_text,
                app_matches, depth, is_app_user, scanned, discovered_from, last_scanned_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(riot_id) DO UPDATE SET
                tier = CASE WHEN excluded.tier != 'UNKNOWN' THEN excluded.tier ELSE players.tier END,
                rank_text = CASE WHEN excluded.rank_text != 'UNRANKED' THEN excluded.rank_text ELSE players.rank_text END,
                app_matches = MAX(players.app_matches, excluded.app_matches),
                is_app_user = MAX(players.is_app_user, excluded.is_app_user),
                scanned = MAX(players.scanned, excluded.scanned),
                last_scanned_at = COALESCE(excluded.last_scanned_at, players.last_scanned_at);
            """,
            (
                riot_id,
                region.lower(),
                game_name,
                tag_line,
                tier_norm,
                rank_text,
                app_matches,
                depth,
                is_app,
                scanned_int,
                discovered_from,
                now,
            ),
        )
        changed = cursor.rowcount > 0
        cursor.close()
        return changed

    def add_players_batch(self, players: Sequence[DbPlayer | Mapping[str, Any]]) -> int:
        """Batch insert or update player records."""
        if not players:
            return 0

        rows: list[tuple] = []
        for p in players:
            if isinstance(p, DbPlayer):
                tier_norm = normalize_tier(p.rank_text) if p.rank_text != "UNRANKED" else normalize_tier(p.tier)
                rows.append((
                    p.riot_id,
                    p.region.lower(),
                    p.game_name,
                    p.tag_line,
                    tier_norm,
                    p.rank_text,
                    p.app_matches,
                    p.depth,
                    1 if p.is_app_user or p.app_matches > 0 else 0,
                    1 if p.scanned else 0,
                    p.discovered_from,
                    p.last_scanned_at,
                ))
            elif isinstance(p, Mapping):
                raw_rank = p.get("rank_text", "UNRANKED")
                raw_tier = p.get("tier", "UNKNOWN")
                tier_norm = normalize_tier(raw_rank) if raw_rank != "UNRANKED" else normalize_tier(raw_tier)
                scanned = 1 if p.get("scanned") else 0
                app_m = int(p.get("app_matches", 0))
                rows.append((
                    p["riot_id"],
                    p.get("region", "na1").lower(),
                    p["game_name"],
                    p["tag_line"],
                    tier_norm,
                    raw_rank,
                    app_m,
                    int(p.get("depth", 0)),
                    1 if p.get("is_app_user") or app_m > 0 else 0,
                    scanned,
                    p.get("discovered_from"),
                    p.get("last_scanned_at"),
                ))

        cursor = self.conn.cursor()
        cursor.execute("BEGIN IMMEDIATE;")
        cursor.executemany(
            """
            INSERT INTO players (
                riot_id, region, game_name, tag_line, tier, rank_text,
                app_matches, depth, is_app_user, scanned, discovered_from, last_scanned_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(riot_id) DO UPDATE SET
                tier = CASE WHEN excluded.tier != 'UNKNOWN' THEN excluded.tier ELSE players.tier END,
                rank_text = CASE WHEN excluded.rank_text != 'UNRANKED' THEN excluded.rank_text ELSE players.rank_text END,
                app_matches = MAX(players.app_matches, excluded.app_matches),
                is_app_user = MAX(players.is_app_user, excluded.is_app_user),
                scanned = CASE
                    WHEN players.last_scanned_at IS NOT NULL AND (strftime('%s', 'now') - players.last_scanned_at) > 43200 THEN 0
                    ELSE MAX(players.scanned, excluded.scanned)
                END;
            """,
            rows,
        )
        placeholders = ",".join("?" for _ in rows)
        cursor.execute(
            f"SELECT COUNT(*) FROM players WHERE riot_id IN ({placeholders}) AND scanned = 0;",
            [r[0] for r in rows],
        )
        unscanned_count = cursor.fetchone()[0]
        cursor.execute("COMMIT;")
        cursor.close()
        return unscanned_count

    def get_next_unscanned_player(
        self,
        tier: str | None = None,
        max_depth: int | None = None,
    ) -> DbPlayer | None:
        """Retrieve the best unscanned candidate player for discovery."""
        cursor = self.conn.cursor()
        query = ["SELECT * FROM players WHERE scanned = 0"]
        params: list[Any] = []

        if tier:
            query.append("AND (tier = ? OR tier = 'UNKNOWN')")
            params.append(tier.upper())
        if max_depth is not None:
            query.append("AND depth <= ?")
            params.append(max_depth)

        # Prioritize known target tier, then app-user flag, then smaller depth
        query.append("ORDER BY (tier != 'UNKNOWN') DESC, is_app_user DESC, depth ASC, rowid ASC LIMIT 1;")
        cursor.execute(" ".join(query), params)
        row = cursor.fetchone()
        cursor.close()

        if not row:
            return None
        return DbPlayer(
            riot_id=row["riot_id"],
            region=row["region"],
            game_name=row["game_name"],
            tag_line=row["tag_line"],
            tier=row["tier"],
            rank_text=row["rank_text"],
            app_matches=row["app_matches"],
            depth=row["depth"],
            is_app_user=bool(row["is_app_user"]),
            scanned=bool(row["scanned"]),
            discovered_from=row["discovered_from"],
            last_scanned_at=row["last_scanned_at"],
        )

    def mark_player_scanned(
        self,
        riot_id: str,
        *,
        app_matches: int,
        tier: str | None = None,
        rank_text: str | None = None,
    ) -> None:
        """Mark a player profile as scanned with updated app match count and rating."""
        now = time.time()
        is_app = 1 if app_matches > 0 else 0
        cursor = self.conn.cursor()

        updates = ["scanned = 1", "last_scanned_at = ?", "app_matches = ?", "is_app_user = ?"]
        params: list[Any] = [now, app_matches, is_app]

        if tier and tier != "UNKNOWN":
            updates.append("tier = ?")
            params.append(normalize_tier(tier))
        if rank_text and rank_text != "UNRANKED":
            updates.append("rank_text = ?")
            params.append(rank_text)

        params.append(riot_id)
        cursor.execute(
            f"UPDATE players SET {', '.join(updates)} WHERE riot_id = ?;",
            params,
        )
        cursor.close()

    def get_aging_app_users(
        self,
        *,
        min_age_seconds: float = 3 * 86400.0,
        tier: str | None = None,
        limit: int = 10,
    ) -> list[DbPlayer]:
        """Fetch verified app users whose profile hasn't been scanned recently to discover fresh matches."""
        cutoff = time.time() - min_age_seconds
        cursor = self.conn.cursor()
        query = [
            "SELECT * FROM players",
            "WHERE is_app_user = 1 AND (last_scanned_at IS NULL OR last_scanned_at < ?)",
        ]
        params: list[Any] = [cutoff]

        if tier:
            query.append("AND tier = ?")
            params.append(tier.upper())

        query.append("ORDER BY last_scanned_at ASC NULLS FIRST LIMIT ?;")
        params.append(limit)

        cursor.execute(" ".join(query), params)
        rows = cursor.fetchall()
        cursor.close()

        return [
            DbPlayer(
                riot_id=r["riot_id"],
                region=r["region"],
                game_name=r["game_name"],
                tag_line=r["tag_line"],
                tier=r["tier"],
                rank_text=r["rank_text"],
                app_matches=r["app_matches"],
                depth=r["depth"],
                is_app_user=bool(r["is_app_user"]),
                scanned=bool(r["scanned"]),
                discovered_from=r["discovered_from"],
                last_scanned_at=r["last_scanned_at"],
            )
            for r in rows
        ]

    def reopen_stale_players(self, tier: str, min_hours: float = 24.0) -> int:
        """Reset scanned = 0 for players in this tier whose last scan is older than min_hours."""
        cutoff = time.time() - (min_hours * 3600.0)
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE players
            SET scanned = 0
            WHERE tier = ? AND (last_scanned_at IS NULL OR last_scanned_at < ?);
            """,
            (tier.upper(), cutoff),
        )
        count = cursor.rowcount
        cursor.close()
        return count

    # -------------------------------------------------------------------------
    # Game Candidate Management & Queue
    # -------------------------------------------------------------------------

    def add_games_batch(self, games: Sequence[DbGame | Mapping[str, Any]]) -> int:
        """Batch insert candidate match replay URLs into games table."""
        if not games:
            return 0

        rows: list[tuple] = []
        for g in games:
            if isinstance(g, DbGame):
                tier_norm = normalize_tier(g.avg_rating or g.tier)
                rows.append((
                    g.match_uuid,
                    g.timeline_url,
                    g.focal_player_riot_id,
                    tier_norm,
                    g.avg_rating,
                    g.tft_set,
                    g.status,
                ))
            elif isinstance(g, Mapping):
                tier_norm = normalize_tier(g.get("avg_rating") or g.get("tier") or "UNKNOWN")
                rows.append((
                    g["match_uuid"],
                    g["timeline_url"],
                    g.get("focal_player_riot_id", ""),
                    tier_norm,
                    g.get("avg_rating"),
                    g.get("tft_set", "TFTSet18"),
                    g.get("status", "pending"),
                ))

        cursor = self.conn.cursor()
        cursor.execute("BEGIN IMMEDIATE;")
        cursor.executemany(
            """
            INSERT OR IGNORE INTO games (
                match_uuid, timeline_url, focal_player_riot_id, tier, avg_rating, tft_set, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            rows,
        )
        cursor.execute("COMMIT;")
        cursor.close()
        return len(rows)

    def get_next_pending_game(self, tier: str | None = None) -> DbGame | None:
        """Retrieve the next pending match replay candidate for download."""
        cursor = self.conn.cursor()
        query = ["SELECT * FROM games WHERE status = 'pending'"]
        params: list[Any] = []

        if tier:
            query.append("AND tier = ?")
            params.append(tier.upper())

        query.append("ORDER BY rowid ASC LIMIT 1;")
        cursor.execute(" ".join(query), params)
        row = cursor.fetchone()
        cursor.close()

        if not row:
            return None
        return DbGame(
            match_uuid=row["match_uuid"],
            timeline_url=row["timeline_url"],
            focal_player_riot_id=row["focal_player_riot_id"],
            tier=row["tier"],
            avg_rating=row["avg_rating"],
            tft_set=row["tft_set"],
            status=row["status"],
            downloaded_at=row["downloaded_at"],
            error_message=row["error_message"],
        )

    def record_match_downloaded(
        self,
        *,
        match_uuid: str,
        tier: str,
        focal_player: str,
        elo_rating: int | None = None,
        placement: int | None = None,
        is_top1: int = 0,
    ) -> None:
        """Atomically mark a match as downloaded and record in downloaded_matches table."""
        now = time.time()
        tier_norm = normalize_tier(tier)

        cursor = self.conn.cursor()
        cursor.execute("BEGIN IMMEDIATE;")
        cursor.execute(
            """
            INSERT INTO downloaded_matches (match_uuid, tier, focal_player, downloaded_at, elo_rating, placement, is_top1)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(match_uuid) DO UPDATE SET
                tier = excluded.tier,
                downloaded_at = excluded.downloaded_at,
                elo_rating = COALESCE(excluded.elo_rating, downloaded_matches.elo_rating),
                placement = COALESCE(excluded.placement, downloaded_matches.placement),
                is_top1 = MAX(excluded.is_top1, downloaded_matches.is_top1);
            """,
            (match_uuid, tier_norm, focal_player, now, elo_rating, placement, is_top1),
        )
        cursor.execute(
            """
            UPDATE games
            SET status = 'downloaded', downloaded_at = ?, tier = ?
            WHERE match_uuid = ?;
            """,
            (now, tier_norm, match_uuid),
        )
        cursor.execute("COMMIT;")
        cursor.close()

    def record_match_failed(self, match_uuid: str, error_message: str) -> None:
        """Mark a match as failed with an error message."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE games
            SET status = 'failed', error_message = ?
            WHERE match_uuid = ?;
            """,
            (error_message[:255], match_uuid),
        )
        cursor.close()

    # -------------------------------------------------------------------------
    # Backward Compatibility: Export to CSV
    # -------------------------------------------------------------------------

    def export_csv_manifests(self, output_dir: str | Path) -> None:
        """Export current database tables to legacy games.csv and players.csv."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        cursor = self.conn.cursor()

        # Export players
        players_tmp = out / "players.csv.tmp"
        players_path = out / "players.csv"
        cursor.execute("SELECT * FROM players ORDER BY rowid ASC;")
        with players_tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "riot_id", "region", "game_name", "tag_line", "tier",
                "rank_text", "app_matches", "depth", "is_app_user", "scanned", "discovered_from"
            ])
            for r in cursor:
                writer.writerow([
                    r["riot_id"], r["region"], r["game_name"], r["tag_line"], r["tier"],
                    r["rank_text"], r["app_matches"], r["depth"], r["is_app_user"], r["scanned"],
                    r["discovered_from"] or ""
                ])
        players_tmp.replace(players_path)

        # Export games
        games_tmp = out / "games.csv.tmp"
        games_path = out / "games.csv"
        cursor.execute("SELECT * FROM games ORDER BY rowid ASC;")
        with games_tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "match_uuid", "timeline_url", "focal_player_riot_id", "tier", "avg_rating", "tft_set"
            ])
            for r in cursor:
                writer.writerow([
                    r["match_uuid"], r["timeline_url"], r["focal_player_riot_id"],
                    r["tier"], r["avg_rating"] or "", r["tft_set"]
                ])
        games_tmp.replace(games_path)
        cursor.close()
        logger.info("Exported SQLite state to %s/players.csv and games.csv", out)

    def get_recent_downloaded_matches(self, limit: int = 50) -> list[dict[str, Any]]:
        """Retrieve the most recently downloaded matches."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT match_uuid, tier, focal_player, downloaded_at, elo_rating, placement, is_top1
            FROM downloaded_matches
            ORDER BY downloaded_at DESC
            LIMIT ?;
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        cursor.close()
        return [
            {
                "match_uuid": r["match_uuid"],
                "tier": r["tier"],
                "focal_player": r["focal_player"],
                "elo_rating": r["elo_rating"],
                "placement": r["placement"],
                "is_top1": bool(r["is_top1"]),
                "downloaded_at": datetime.fromtimestamp(r["downloaded_at"], tz=timezone.utc).isoformat()
                if r["downloaded_at"]
                else None,
            }
            for r in rows
        ]

    def get_analytics_summary(self) -> dict[str, Any]:
        """Compute comprehensive ELO distribution and placement / Top 1 analytics."""
        cursor = self.conn.cursor()

        # 1. Total matches and Top 1 count
        cursor.execute(
            """
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN is_top1 = 1 OR placement = 1 THEN 1 ELSE 0 END) as top1_count,
                AVG(elo_rating) as avg_elo,
                MIN(elo_rating) as min_elo,
                MAX(elo_rating) as max_elo
            FROM downloaded_matches;
            """
        )
        row = cursor.fetchone()
        total_matches = row["total"] or 0
        top1_count = row["top1_count"] or 0
        avg_elo = round(row["avg_elo"], 1) if row["avg_elo"] is not None else 0.0
        min_elo = row["min_elo"] if row["min_elo"] is not None else 0
        max_elo = row["max_elo"] if row["max_elo"] is not None else 0

        # 2. Placement histogram (1 to 8)
        cursor.execute(
            """
            SELECT placement, COUNT(*) as cnt
            FROM downloaded_matches
            WHERE placement IS NOT NULL
            GROUP BY placement
            ORDER BY placement ASC;
            """
        )
        placements_dict: dict[str, int] = {str(i): 0 for i in range(1, 9)}
        total_with_placement = 0
        for r in cursor.fetchall():
            p_val = r["placement"]
            if p_val and 1 <= p_val <= 8:
                placements_dict[str(p_val)] = r["cnt"]
                total_with_placement += r["cnt"]

        # 3. Detailed ELO Histogram (Bins of 200 ELO points)
        cursor.execute(
            """
            SELECT 
                ((elo_rating / 200) * 200) as bin_start,
                COUNT(*) as cnt
            FROM downloaded_matches
            WHERE elo_rating IS NOT NULL
            GROUP BY bin_start
            ORDER BY bin_start ASC;
            """
        )
        elo_bins_raw = cursor.fetchall()
        cursor.close()

        bin_counts_map = {r["bin_start"]: r["cnt"] for r in elo_bins_raw}
        elo_labels: list[str] = []
        elo_counts: list[int] = []

        for b_start in range(200, 5000, 200):
            b_end = b_start + 200
            if b_start < 600:
                tier_hint = "Hierro"
            elif b_start < 1000:
                tier_hint = "Bronce"
            elif b_start < 1400:
                tier_hint = "Plata"
            elif b_start < 1800:
                tier_hint = "Oro"
            elif b_start < 2200:
                tier_hint = "Platino"
            elif b_start < 2600:
                tier_hint = "Esmeralda"
            elif b_start < 3000:
                tier_hint = "Diamante"
            elif b_start < 3600:
                tier_hint = "Master"
            elif b_start < 4200:
                tier_hint = "GM"
            else:
                tier_hint = "Challenger"

            label = f"{b_start}-{b_end} ({tier_hint})"
            elo_labels.append(label)
            elo_counts.append(bin_counts_map.get(b_start, 0))

        top1_pct = round((top1_count / max(1, total_matches)) * 100, 1)
        top4_count = sum(placements_dict[str(i)] for i in range(1, 5))
        top4_pct = round((top4_count / max(1, total_with_placement or total_matches)) * 100, 1)

        return {
            "total_matches": total_matches,
            "top1_count": top1_count,
            "top1_pct": top1_pct,
            "top4_count": top4_count,
            "top4_pct": top4_pct,
            "avg_elo": avg_elo,
            "min_elo": min_elo,
            "max_elo": max_elo,
            "placement_histogram": {
                "labels": ["Top 1 🏆", "Top 2 🥈", "Top 3 🥉", "Top 4 🎖️", "Top 5", "Top 6", "Top 7", "Top 8"],
                "counts": [placements_dict[str(i)] for i in range(1, 9)],
            },
            "elo_histogram": {
                "labels": elo_labels,
                "counts": elo_counts,
            },
        }

