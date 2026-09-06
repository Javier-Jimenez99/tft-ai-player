"""24/7 continuous balanced TFT match collector service.

Designed for unattended execution on Raspberry Pi:
- Proportional fair scheduling across all ELOs (Iron to Challenger).
- Self-sustaining closed loop: every downloaded match feeds 7 lobby participants back into discovery.
- Multi-channel anti-starvation for low elos (boundary mining, aging app-user rescan, multi-region seeding).
- Resilient to network drops, HTTP 429 rate limits, and Tor IP circuit rotation.
- MicroSD card safety (free disk space monitoring, atomic SQLite WAL writes).
- Health monitoring via periodic status JSON heartbeat and clean SIGINT/SIGTERM daemon handling.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .collector_db import ALL_STANDARD_TIERS, CollectorDb, DbGame, DbPlayer
from .downloader import rotate_tor_identity, setup_network_proxy
from .models import normalize_tier
from .storage import PlayerCsvWriter
from .timeline import TimelineValidationError, extract_pvp_rounds
from ..metatft.client import MetaTftClient, MetaTftRequestError

logger = logging.getLogger(__name__)

# Default distribution: ~45% High-Elo, ~35% Mid-Elo, ~20% Low-Elo
DEFAULT_TARGET_WEIGHTS: dict[str, float] = {
    "CHALLENGER": 0.10,
    "GRANDMASTER": 0.15,
    "MASTER": 0.20,
    "DIAMOND": 0.18,
    "EMERALD": 0.14,
    "PLATINUM": 0.10,
    "GOLD": 0.06,
    "SILVER": 0.04,
    "BRONZE": 0.02,
    "IRON": 0.01,
}

ADJACENT_TIERS: dict[str, tuple[str, ...]] = {
    "IRON": ("BRONZE", "SILVER", "GOLD"),
    "BRONZE": ("IRON", "SILVER", "GOLD"),
    "SILVER": ("BRONZE", "GOLD", "PLATINUM", "IRON"),
    "GOLD": ("SILVER", "PLATINUM", "EMERALD", "BRONZE"),
    "PLATINUM": ("GOLD", "EMERALD", "DIAMOND"),
    "EMERALD": ("PLATINUM", "DIAMOND", "MASTER"),
    "DIAMOND": ("EMERALD", "MASTER", "GRANDMASTER"),
    "MASTER": ("DIAMOND", "GRANDMASTER", "CHALLENGER"),
    "GRANDMASTER": ("MASTER", "CHALLENGER"),
    "CHALLENGER": ("GRANDMASTER", "MASTER"),
}


@dataclass(slots=True)
class CollectorStats:
    """Live runtime stats for health monitoring."""

    started_at: float
    total_downloaded_session: int = 0
    total_skipped_session: int = 0
    total_errors_session: int = 0
    consecutive_errors: int = 0
    last_action: str = "starting"
    last_action_at: float = 0.0
    last_successful_download_at: float = 0.0
    rate_limited_until: float = 0.0


class ContinuousCollectorService:
    """Autonomous 24/7 service managing continuous collection without ELO starvation."""

    def __init__(
        self,
        *,
        db_path: str | Path = "data/collector.db",
        output_dir: str | Path = "data",
        target_weights: dict[str, float] | None = None,
        request_interval: float = 1.2,
        tft_set: str = "TFTSet18",
        status_file: str | Path | None = "data/status.json",
        proxy: str | None = None,
        tor_control_port: int | None = None,
        auto_wait_cooldown: bool = True,
        riot_api_key: str | None = None,
        min_disk_free_gb: float = 2.0,
        export_csv_interval: float = 3600.0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.status_file = Path(status_file) if status_file else None
        self.tft_set = tft_set
        self.request_interval = request_interval
        self.auto_wait_cooldown = auto_wait_cooldown
        self.tor_control_port = tor_control_port
        self.riot_api_key = riot_api_key
        self.min_disk_free_gb = min_disk_free_gb
        self.export_csv_interval = export_csv_interval
        self._last_csv_export_at = time.time()
        self._last_status_write_at = 0.0
        self._last_leaderboard_fetch_at = 0.0

        # Normalize weights
        raw_weights = target_weights or DEFAULT_TARGET_WEIGHTS
        total_w = sum(raw_weights.values())
        self.target_weights = {k.upper(): v / total_w for k, v in raw_weights.items()}

        if proxy:
            setup_network_proxy(proxy)

        self.db = CollectorDb(db_path)
        self.client = MetaTftClient(
            minimum_request_interval_seconds=request_interval,
            retry_count=3,
        )
        self.writer = PlayerCsvWriter(self.output_dir, tier_partitioned=True)
        self.stats = CollectorStats(started_at=time.time())
        self._last_seed_injection: dict[str, float] = {}
        self._stop_requested = False

        # Register signal handlers for clean daemon shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        logger.info("Received termination signal %d. Shutting down gracefully...", signum)
        self._stop_requested = True

    def initialize_data(self) -> None:
        """Scan existing data directories and migrate legacy CSVs if necessary."""
        logger.info("Initializing collector state...")
        # 1. Import on-disk matches (ensures zero data loss and prevents re-downloading)
        disk_count = self.db.import_existing_disk_matches(self.output_dir)
        logger.info("Indexed %d existing matches from disk", disk_count)

        # 2. Import legacy graph CSVs if present
        graph_dir = self.output_dir / "graph"
        if graph_dir.exists():
            p_cnt, g_cnt = self.db.import_existing_graph(graph_dir)
            if p_cnt > 0 or g_cnt > 0:
                logger.info("Imported legacy graph manifest: %d players, %d games", p_cnt, g_cnt)

        # 3. Seed default players if DB has very few players
        counts = self.db.get_tier_counts()
        total_players = sum(counts["unscanned_players"].values()) + sum(counts["app_users"].values())
        if total_players < 20:
            logger.info("Database has few players (%d). Injecting multi-tier seed players...", total_players)
            self._inject_default_seeds()

    def _inject_default_seeds(self, target_tier: str | None = None) -> int:
        """Inject curated seeds from DEFAULT_MULTI_TIER_SEEDS into the database."""
        now = time.time()
        tier_key = target_tier.upper() if target_tier else "ALL"
        last_inj = self._last_seed_injection.get(tier_key, 0.0)
        if now - last_inj < 600.0:
            return 0
        self._last_seed_injection[tier_key] = now

        from .seeds import DEFAULT_MULTI_TIER_SEEDS

        added = 0
        for reg, gname, tag, tier in DEFAULT_MULTI_TIER_SEEDS:
            if target_tier and tier.upper() != target_tier.upper():
                continue
            p_rid = f"{gname}#{tag}"
            if self.db.add_player(
                riot_id=p_rid,
                region=reg.lower(),
                game_name=gname,
                tag_line=tag,
                tier=tier.upper(),
                depth=0,
            ):
                added += 1
        return added

    # -------------------------------------------------------------------------
    # Proportional Fair Scheduling Engine
    # -------------------------------------------------------------------------

    def compute_deficits(self) -> list[tuple[str, float]]:
        """Compute normalized deficit for each tier:
        D(tier) = (target_weight - current_share) / sqrt(target_weight).

        Using sqrt(target_weight) balances absolute shortage with relative starvation:
        - Prevents small-target tiers (like IRON at 1%) from being starved out by large-target tiers (like MASTER at 20%).
        - When all tiers have 0 downloads, preserves the natural hierarchy (MASTER > DIAMOND > ... > IRON).
        - Correctly prioritizes high-weight targets when custom quotas are set.

        Returns:
            list of (tier_name, deficit) sorted descending by deficit (largest normalized shortage first).
        """
        import math
        counts = self.db.get_tier_counts()
        downloaded = counts["downloaded"]
        total_downloaded = sum(downloaded.values())

        deficits: list[tuple[str, float]] = []
        for tier in ALL_STANDARD_TIERS:
            target_w = self.target_weights.get(tier, 0.01)
            if total_downloaded == 0:
                current_share = 0.0
            else:
                current_share = downloaded.get(tier, 0) / total_downloaded
            # Scale-balanced deficit
            deficit = (target_w - current_share) / math.sqrt(max(1e-6, target_w))
            deficits.append((tier, deficit))

        deficits.sort(key=lambda item: item[1], reverse=True)
        return deficits

    # -------------------------------------------------------------------------
    # Single Collection Cycle (Step)
    # -------------------------------------------------------------------------

    def step(self) -> bool:
        """Execute one autonomous collection step. Returns True if work was done."""
        # 1. MicroSD safety check
        if not self._check_disk_space():
            logger.warning("Low disk space (< %.1f GB). Pausing downloads for 60s...", self.min_disk_free_gb)
            time.sleep(60.0)
            return False

        # 2. Rate limit cooldown check
        now = time.time()
        if now < self.stats.rate_limited_until:
            rem = int(self.stats.rate_limited_until - now)
            logger.debug("In rate-limit cooldown: %ds remaining", rem)
            time.sleep(min(5.0, max(1.0, rem)))
            return False

        # 3. Compute deficits across tiers
        deficits = self.compute_deficits()
        logger.debug("Current top deficit tiers: %s", deficits[:3])

        # 4. Attempt to satisfy deficit tiers in order
        for tier, deficit in deficits:
            try:
                # Check if this tier has pending games ready to download
                pending_game = self.db.get_next_pending_game(tier=tier)
                if pending_game:
                    success = self._download_game(pending_game)
                    if success:
                        self.stats.consecutive_errors = 0
                        self.stats.last_successful_download_at = time.time()
                    return True

                # If no pending game, try targeted discovery for this starving tier
                discovered = self._discover_for_tier(tier)
                if discovered > 0:
                    logger.info("Discovered %d candidate items for starved tier %s", discovered, tier)
                    # Check if discovery yielded a pending game for this tier right now
                    pending_game = self.db.get_next_pending_game(tier=tier)
                    if pending_game:
                        success = self._download_game(pending_game)
                        if success:
                            self.stats.consecutive_errors = 0
                            self.stats.last_successful_download_at = time.time()
                        return True

                    # Check if discovery yielded an unscanned player for this tier to profile
                    unscanned_player = self.db.get_next_unscanned_player(tier=tier)
                    if unscanned_player:
                        added_games = self._scan_player_profile(unscanned_player)
                        if added_games > 0:
                            return True
            except Exception as tier_err:
                logger.warning("Error during deficit processing for tier %s: %s", tier, tier_err, exc_info=True)
                continue

        # 5. If no targeted work could be completed for top deficit tiers,
        # fallback to downloading ANY available pending game across all tiers!
        fallback_game = self.db.get_next_pending_game(tier=None)
        if fallback_game:
            success = self._download_game(fallback_game)
            if success:
                self.stats.consecutive_errors = 0
                self.stats.last_successful_download_at = time.time()
            return True

        # 6. Fallback to scanning ANY available unscanned player across any tier
        fallback_player = self.db.get_next_unscanned_player(tier=None)
        if fallback_player:
            return self._scan_player_profile(fallback_player) > 0

        # 7. If completely dry across all queues and players, perform broad replenishment
        logger.info("All queues dry. Performing broad replenishment across all tiers...")
        return self._broad_replenish() > 0

    # -------------------------------------------------------------------------
    # Download & Lobby Recycling
    # -------------------------------------------------------------------------

    def _download_game(self, game: DbGame) -> bool:
        """Download timeline, extract PvP observations, write CSV, and recycle lobby participants."""
        self.stats.last_action = f"downloading_{game.tier}_{game.match_uuid[:8]}"
        self.stats.last_action_at = time.time()

        try:
            timeline = self.client.fetch_timeline(game.timeline_url)
        except MetaTftRequestError as err:
            return self._handle_network_or_rate_limit_error(err, game.match_uuid)
        except Exception as err:
            logger.error("Error fetching match %s: %s", game.match_uuid, err)
            self.db.record_match_failed(game.match_uuid, str(err))
            self.stats.total_errors_session += 1
            return False

        # Extract observations
        try:
            focal_p = game.focal_player_riot_id
            focal_summoner = focal_p.partition("#")[0] if focal_p else None
            observations = extract_pvp_rounds(
                timeline,
                match_id=game.match_uuid,
                tft_set=game.tft_set or self.tft_set,
                game_version="unknown",
                focal_player=focal_summoner,
                focal_tier=game.tier,
                avg_match_rating=game.avg_rating,
            )
        except (TimelineValidationError, ValueError) as err:
            logger.debug("Validation error for match %s: %s", game.match_uuid, err)
            self.db.record_match_failed(game.match_uuid, f"validation_error: {err}")
            self.stats.total_skipped_session += 1
            return False

        if not observations:
            self.db.record_match_failed(game.match_uuid, "no_pvp_observations")
            self.stats.total_skipped_session += 1
            return False

        # Write observations to tier-partitioned CSV
        effective_tier = observations[0].tier_category or game.tier
        focal_player = observations[0].focal_player or game.focal_player_riot_id
        region = "unknown"
        if "#" in focal_player:
            focal_record = self.db.conn.execute(
                "SELECT region FROM players WHERE riot_id = ? LIMIT 1;",
                (focal_player,),
            ).fetchone()
            if focal_record:
                region = focal_record["region"]

        # Determine Top 1, placement, and ELO points
        is_top1 = 0
        placement = None
        if observations:
            last_obs = observations[-1]
            last_stage = last_obs.round_stage or "2-1"
            focal_h = last_obs.focal_health if last_obs.focal_health is not None else 0
            if last_obs.outcome == "victory" and focal_h > 0:
                is_top1 = 1
                placement = 1
            else:
                try:
                    parts = [int(p) for p in str(last_stage).split("-")]
                    st = (parts[0], parts[1]) if len(parts) >= 2 else (2, 1)
                except Exception:
                    st = (2, 1)
                if st >= (6, 1):
                    placement = 2
                elif st >= (5, 5):
                    placement = 3
                elif st >= (5, 2):
                    placement = 4
                elif st >= (4, 5):
                    placement = 5
                elif st >= (4, 2):
                    placement = 6
                elif st >= (3, 5):
                    placement = 7
                else:
                    placement = 8

        from .collector_db import parse_elo_to_numeric
        rating_candidate = game.avg_rating or (observations[0].avg_match_rating if observations else None) or (observations[0].focal_tier if observations else None)
        elo_val = parse_elo_to_numeric(rating_candidate, tier_fallback=effective_tier)

        path = self.writer.write_player_game(
            observations,
            collected_from_riot_id=focal_player,
            collected_from_region=region,
            tier=effective_tier,
        )

        if path:
            self.db.record_match_downloaded(
                match_uuid=game.match_uuid,
                tier=effective_tier,
                focal_player=focal_player,
                elo_rating=elo_val,
                placement=placement,
                is_top1=is_top1,
            )
            self.stats.total_downloaded_session += 1
            logger.info(
                "[%s] Downloaded match %s (%d rounds, place #%s) -> %s",
                effective_tier,
                game.match_uuid[:12],
                len(observations),
                placement or "?",
                path.name,
            )

        # ---------------------------------------------------------------------
        # LOBBY RECYCLING: Feed the other 7 players back into discovery
        # ---------------------------------------------------------------------
        self._recycle_lobby_participants(timeline, game)
        return True

    def _recycle_lobby_participants(self, timeline: Mapping[str, Any], game: DbGame) -> int:
        """Extract all players from the match timeline and add them as unscanned candidates."""
        participants = self.client.extract_lobby_participants(timeline)
        focal_record = self.db.conn.execute(
            "SELECT region FROM players WHERE riot_id = ? LIMIT 1;",
            (game.focal_player_riot_id,),
        ).fetchone()
        region = focal_record["region"] if focal_record else "na1"

        new_players: list[DbPlayer] = []
        for gname, tag in participants:
            riot_id = f"{gname}#{tag}"
            if riot_id != game.focal_player_riot_id:
                new_players.append(
                    DbPlayer(
                        riot_id=riot_id,
                        region=region,
                        game_name=gname,
                        tag_line=tag,
                        tier=game.tier,  # Initial hint: match tier
                        depth=1,
                        discovered_from=game.match_uuid,
                    )
                )

        if new_players:
            added = self.db.add_players_batch(new_players)
            logger.debug(
                "Recycled lobby from match %s: +%d player candidates for %s",
                game.match_uuid[:8],
                added,
                game.tier,
            )
            return added
        return 0

    # -------------------------------------------------------------------------
    # Targeted Discovery Engine (Anti-Starvation)
    # -------------------------------------------------------------------------

    def _discover_for_tier(self, tier: str) -> int:
        """Execute anti-starvation discovery mechanisms for a specific tier."""
        tier_upper = tier.upper()
        self.stats.last_action = f"discovering_{tier_upper}"
        self.stats.last_action_at = time.time()

        # Channel 1: Unscanned player in DB matching this tier
        try:
            player = self.db.get_next_unscanned_player(tier=tier_upper)
            if player:
                return self._scan_player_profile(player)
        except Exception as e:
            logger.warning("Discovery Channel 1 failed for %s: %s", tier_upper, e)

        # Channel 2: Boundary mining from adjacent tiers
        try:
            adjacent_tiers = ADJACENT_TIERS.get(tier_upper, ())
            for adj in adjacent_tiers:
                adj_player = self.db.get_next_unscanned_player(tier=adj)
                if adj_player:
                    logger.info("Boundary mining: checking %s player %s to uncover %s matches", adj, adj_player.riot_id, tier_upper)
                    return self._scan_player_profile(adj_player)
        except Exception as e:
            logger.warning("Discovery Channel 2 failed for %s: %s", tier_upper, e)

        # Channel 3: Aging app-user rescan (re-scan known app users from > 12h / 24h ago)
        try:
            min_age = 43200.0 if tier_upper in ("IRON", "BRONZE", "SILVER") else 86400.0
            aging_players = self.db.get_aging_app_users(min_age_seconds=min_age, tier=tier_upper, limit=1)
            if not aging_players and tier_upper in ("IRON", "BRONZE"):
                # Check adjacent tier app-users (e.g. Bronze/Silver app users whose lobbies cross into Iron)
                for adj in adjacent_tiers:
                    aging_players = self.db.get_aging_app_users(min_age_seconds=min_age, tier=adj, limit=1)
                    if aging_players:
                        break
            if aging_players:
                logger.info("Re-scanning aging app user %s (%s) for fresh matches", aging_players[0].riot_id, aging_players[0].tier)
                return self._scan_player_profile(aging_players[0])
        except Exception as e:
            logger.warning("Discovery Channel 3 failed for %s: %s", tier_upper, e)

        # Channel 4: Riot League API (if key available)
        try:
            if self.riot_api_key:
                count = self._query_riot_api_for_tier(tier_upper)
                if count > 0:
                    return count
        except Exception as e:
            logger.warning("Discovery Channel 4 failed for %s: %s", tier_upper, e)

        # Channel 5: MetaTFT Leaderboard for high-tier
        try:
            if tier_upper in ("CHALLENGER", "GRANDMASTER", "MASTER"):
                count = self._replenish_from_leaderboard()
                if count > 0:
                    return count
        except Exception as e:
            logger.warning("Discovery Channel 5 failed for %s: %s", tier_upper, e)

        # Channel 6: Inject default seed players for this tier
        try:
            seed_count = self._inject_default_seeds(target_tier=tier_upper)
            if seed_count > 0:
                logger.info("Injected %d seed players for %s", seed_count, tier_upper)
                return seed_count
        except Exception as e:
            logger.warning("Discovery Channel 6 failed for %s: %s", tier_upper, e)

        # Channel 7: Low-ELO Player Re-Opening (for Iron, Bronze, Silver)
        try:
            if tier_upper in ("IRON", "BRONZE", "SILVER"):
                reopened = self.db.reopen_stale_players(tier=tier_upper, min_hours=12.0)
                if reopened == 0 and tier_upper in ("IRON", "BRONZE"):
                    # Also try reopening stale players from adjacent tiers (Bronze/Silver)
                    for adj in adjacent_tiers:
                        reopened = self.db.reopen_stale_players(tier=adj, min_hours=12.0)
                        if reopened > 0:
                            break
                if reopened > 0:
                    logger.info("Re-opened %d stale players for %s discovery", reopened, tier_upper)
                    reopened_player = self.db.get_next_unscanned_player(tier=tier_upper) or (
                        self.db.get_next_unscanned_player(tier=adjacent_tiers[0]) if adjacent_tiers else None
                    )
                    if reopened_player:
                        return self._scan_player_profile(reopened_player)
        except Exception as e:
            logger.warning("Discovery Channel 7 failed for %s: %s", tier_upper, e)

        return 0

    def _scan_player_profile(self, player: DbPlayer) -> int:
        """Fetch profile for a player, catalog candidate games, and mark as scanned."""
        try:
            profile = self.client.fetch_profile(
                region=player.region,
                game_name=player.game_name,
                tag_line=player.tag_line,
                tft_set=self.tft_set,
            )
        except MetaTftRequestError as err:
            self.db.mark_player_scanned(player.riot_id, app_matches=0)
            self._handle_network_or_rate_limit_error(err, player.riot_id)
            return 0
        except Exception as err:
            logger.debug("Error fetching profile for %s: %s", player.riot_id, err)
            self.db.mark_player_scanned(player.riot_id, app_matches=0)
            return 0

        # Extract verified rank and tier
        rank_text = self.client.extract_player_tier(profile) or player.rank_text
        tier_norm = normalize_tier(rank_text) if rank_text != "UNRANKED" else player.tier

        # Extract app-tracked timeline matches
        candidates = self.client.tracked_timeline_candidates(
            profile,
            tft_set=self.tft_set,
            allowed_queue_ids=(1100,),  # Ranked TFT
        )

        self.db.mark_player_scanned(
            player.riot_id,
            app_matches=len(candidates),
            tier=tier_norm,
            rank_text=rank_text,
        )

        if not candidates:
            return 0

        # Insert discovered candidate games
        game_records: list[DbGame] = []
        for c in candidates:
            # Use match's own average rating tier if known (crucial for capturing Iron/Bronze lobbies)
            match_rating_tier = normalize_tier(c.avg_match_rating) if c.avg_match_rating else "UNKNOWN"
            effective_game_tier = (
                match_rating_tier if match_rating_tier != "UNKNOWN"
                else (c.focal_tier or tier_norm)
            )
            game_records.append(
                DbGame(
                    match_uuid=c.app_match_uuid,
                    timeline_url=c.timeline_url,
                    focal_player_riot_id=player.riot_id,
                    tier=effective_game_tier,
                    avg_rating=c.avg_match_rating,
                    tft_set=self.tft_set,
                )
            )

        added_games = self.db.add_games_batch(game_records)
        logger.info(
            "Found App User %s (%s) -> +%d candidate games",
            player.riot_id,
            tier_norm,
            added_games,
        )
        return added_games

    def _replenish_from_leaderboard(self, count: int = 50) -> int:
        """Fetch fresh Challenger/GM/Master players from MetaTFT leaderboard (rate-limited)."""
        now = time.time()
        if now - getattr(self, "_last_leaderboard_fetch_at", 0.0) < 600.0:
            return 0
        self._last_leaderboard_fetch_at = now

        try:
            players = self.client.fetch_leaderboard_players(count=count)
            db_players = [
                DbPlayer(
                    riot_id=p.riot_id,
                    region=p.region.lower(),
                    game_name=p.game_name,
                    tag_line=p.tag_line,
                    tier="CHALLENGER",
                    depth=0,
                )
                for p in players
            ]
            added = self.db.add_players_batch(db_players)
            logger.info("Replenished +%d high-elo players from leaderboard", added)
            return added
        except Exception as e:
            logger.warning("Failed to replenish from leaderboard: %s", e)
            return 0

    def _query_riot_api_for_tier(self, tier: str) -> int:
        """Query official Riot League API if key is present."""
        if not self.riot_api_key:
            return 0
        try:
            from ..metatft.riot_client import RiotTftClient

            riot_client = RiotTftClient(self.riot_api_key)
            players = riot_client.fetch_tier_players(tier, count=15)
            db_players = [
                DbPlayer(
                    riot_id=p.riot_id,
                    region=p.region.lower(),
                    game_name=p.game_name,
                    tag_line=p.tag_line,
                    tier=tier,
                    depth=0,
                )
                for p in players
            ]
            added = self.db.add_players_batch(db_players)
            logger.info("Queried Riot API for %s: +%d players added", tier, added)
            return added
        except Exception as e:
            if "401" in str(e) or "403" in str(e) or getattr(e, "code", None) in (401, 403) or "Unauthorized" in str(e):
                logger.warning("Riot API key is invalid or expired (HTTP 401/403 Unauthorized). Disabling Riot API fallback.")
                self.riot_api_key = None
            else:
                logger.warning("Riot API query failed for %s: %s", tier, e)
            return 0

    def _broad_replenish(self) -> int:
        """Fallback to replenish all tiers when queues are completely dry."""
        total_added = self._inject_default_seeds()
        total_added += self._replenish_from_leaderboard(count=30)
        return total_added

    # -------------------------------------------------------------------------
    # Error Handling, Tor Rotation & Rate Limits
    # -------------------------------------------------------------------------

    def _handle_network_or_rate_limit_error(self, err: Exception, context_id: str) -> bool:
        """Handle HTTP 429 rate-limiting with Tor rotation or cooldown wait."""
        err_str = str(err)
        if "404" in err_str or getattr(err, "status_code", None) == 404 or getattr(err, "code", None) == 404:
            logger.debug("Resource %s not found (HTTP 404). Skipping immediately.", context_id)
            return False

        if "429" in err_str or getattr(err, "status_code", None) == 429 or getattr(err, "code", None) == 429:
            self.stats.total_errors_session += 1
            if self.tor_control_port:
                logger.info("HTTP 429 rate limit on %s. Rotating Tor circuit via port %d...", context_id, self.tor_control_port)
                if rotate_tor_identity(control_port=self.tor_control_port):
                    logger.info("Tor circuit rotated successfully. Retrying after brief pause.")
                    time.sleep(2.0)
                    return False

            if self.auto_wait_cooldown:
                retry_sec = getattr(err, "retry_after", None) or 30.0
                logger.warning("HTTP 429 on %s. Auto-cooling down for %ds...", context_id, int(retry_sec))
                self.stats.rate_limited_until = time.time() + retry_sec
            return False

        logger.warning("Network request error on %s: %s", context_id, err)
        self.stats.consecutive_errors += 1
        # Exponential backoff on consecutive network errors: min 5s, max 60s
        backoff = min(60.0, 5.0 * (2 ** min(self.stats.consecutive_errors, 4)))
        time.sleep(backoff)
        return False

    def _check_disk_space(self) -> bool:
        """Verify minimum disk free space to prevent filling Raspberry Pi storage."""
        try:
            usage = shutil.disk_usage(self.output_dir)
            free_gb = usage.free / (1024 ** 3)
            return free_gb >= self.min_disk_free_gb
        except Exception:
            return True

    # -------------------------------------------------------------------------
    # Health Monitoring & Status Reporting (Heartbeat)
    # -------------------------------------------------------------------------

    def write_status_heartbeat(self) -> None:
        """Write structured health status JSON file for external monitoring."""
        if not self.status_file:
            return

        now = time.time()
        counts = self.db.get_tier_counts()
        downloaded = counts["downloaded"]
        total_dl = sum(downloaded.values())
        uptime_sec = now - self.stats.started_at
        dl_per_hour = (self.stats.total_downloaded_session / (uptime_sec / 3600.0)) if uptime_sec > 60 else 0.0

        current_shares: dict[str, float] = {}
        for t in ALL_STANDARD_TIERS:
            current_shares[t] = (downloaded.get(t, 0) / total_dl) if total_dl > 0 else 0.0

        try:
            free_gb = shutil.disk_usage(self.output_dir).free / (1024 ** 3)
        except Exception:
            free_gb = 0.0

        status_data = {
            "service": "tft-ai-collector",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": round(uptime_sec, 1),
            "uptime_hours": round(uptime_sec / 3600.0, 2),
            "total_matches_on_disk": total_dl,
            "session_downloaded": self.stats.total_downloaded_session,
            "session_skipped": self.stats.total_skipped_session,
            "session_errors": self.stats.total_errors_session,
            "download_rate_per_hour": round(dl_per_hour, 1),
            "free_disk_gb": round(free_gb, 2),
            "last_action": self.stats.last_action,
            "distribution": {
                t: {
                    "downloaded": downloaded.get(t, 0),
                    "pending_games": counts["pending"].get(t, 0),
                    "unscanned_players": counts["unscanned_players"].get(t, 0),
                    "current_share_pct": round(current_shares[t] * 100, 1),
                    "target_share_pct": round(self.target_weights.get(t, 0.0) * 100, 1),
                }
                for t in ALL_STANDARD_TIERS
            },
            "analytics": self.db.get_analytics_summary(),
        }

        try:
            tmp_status = self.status_file.with_suffix(".tmp")
            with tmp_status.open("w", encoding="utf-8") as f:
                json.dump(status_data, f, indent=2)
            tmp_status.replace(self.status_file)
            self._last_status_write_at = now
        except Exception as e:
            logger.debug("Failed to write status heartbeat: %s", e)

    # -------------------------------------------------------------------------
    # Main Daemon Loop
    # -------------------------------------------------------------------------

    def run_forever(self) -> None:
        """Run the collection daemon indefinitely until terminated."""
        logger.info("=" * 70)
        logger.info(" Starting 24/7 TFT Continuous Balanced Collection Daemon")
        logger.info(" Destination: %s", self.output_dir.resolve())
        logger.info(" Target distribution: %s", {t: f"{w*100:.0f}%" for t, w in self.target_weights.items()})
        logger.info("=" * 70)

        self.initialize_data()
        self.write_status_heartbeat()

        try:
            while not self._stop_requested:
                try:
                    self.step()
                except Exception as cycle_err:
                    logger.error("Unhandled error in collection cycle: %s", cycle_err, exc_info=True)
                    time.sleep(5.0)

                # Periodic heartbeat (every 30 seconds)
                if time.time() - self._last_status_write_at > 30.0:
                    self.write_status_heartbeat()

                # Periodic legacy CSV export for backward compatibility (e.g. every hour)
                if self.export_csv_interval > 0 and (time.time() - self._last_csv_export_at > self.export_csv_interval):
                    try:
                        self.db.export_csv_manifests(self.output_dir / "graph")
                        self._last_csv_export_at = time.time()
                    except Exception as e:
                        logger.warning("Periodic CSV export failed: %s", e)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received.")
        finally:
            logger.info("Shutting down collector service...")
            self.write_status_heartbeat()
            try:
                self.db.export_csv_manifests(self.output_dir / "graph")
            except Exception:
                pass
            self.db.close()
            logger.info("Collector service shut down cleanly.")
