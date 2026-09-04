"""Player navigation graph and game manifest storage for two-step dataset collection."""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from .models import normalize_tier

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PlayerNode:
    """A player node in the exploration graph."""

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


@dataclass(slots=True)
class GameManifestEntry:
    """A match replay candidate in the games manifest."""

    match_uuid: str
    timeline_url: str
    focal_player_riot_id: str
    tier: str = "UNKNOWN"
    avg_rating: str | None = None
    tft_set: str = "TFTSet18"


@dataclass(slots=True)
class LobbyEdge:
    """A co-occurrence connection between two players in a match lobby."""

    player_a: str
    player_b: str
    match_uuid: str


class PlayerGraph:
    """In-memory player navigation graph with simple CSV persistence."""

    def __init__(self) -> None:
        self.nodes: dict[str, PlayerNode] = {}
        self.games: dict[str, GameManifestEntry] = {}
        self.edges: list[LobbyEdge] = []
        self._seen_edges: set[tuple[str, str, str]] = set()

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
    ) -> PlayerNode:
        """Add or update a player node in the graph."""
        tier_norm = normalize_tier(rank_text) if rank_text and rank_text != "UNRANKED" else normalize_tier(tier)
        if riot_id in self.nodes:
            node = self.nodes[riot_id]
            if rank_text != "UNRANKED":
                node.rank_text = rank_text
                node.tier = tier_norm
            elif node.tier in ("UNKNOWN", "UNRANKED") and tier_norm not in ("UNKNOWN", "UNRANKED"):
                node.tier = tier_norm
            if app_matches > 0:
                node.app_matches = app_matches
                node.is_app_user = True
            if scanned:
                node.scanned = True
            return node

        node = PlayerNode(
            riot_id=riot_id,
            region=region.lower(),
            game_name=game_name,
            tag_line=tag_line,
            tier=tier_norm,
            rank_text=rank_text,
            app_matches=app_matches,
            depth=depth,
            is_app_user=is_app_user or (app_matches > 0),
            scanned=scanned,
            discovered_from=discovered_from,
        )
        self.nodes[riot_id] = node
        return node

    def add_game(
        self,
        *,
        match_uuid: str,
        timeline_url: str,
        focal_player_riot_id: str,
        tier: str = "UNKNOWN",
        avg_rating: str | None = None,
        tft_set: str = "TFTSet18",
    ) -> GameManifestEntry:
        """Add a match entry to the games manifest."""
        tier_norm = normalize_tier(avg_rating or tier)
        entry = GameManifestEntry(
            match_uuid=match_uuid,
            timeline_url=timeline_url,
            focal_player_riot_id=focal_player_riot_id,
            tier=tier_norm,
            avg_rating=avg_rating,
            tft_set=tft_set,
        )
        self.games[match_uuid] = entry
        return entry

    def add_edge(self, player_a: str, player_b: str, match_uuid: str) -> None:
        """Record an edge between two players in a match lobby."""
        if player_a == player_b:
            return
        p1, p2 = sorted([player_a, player_b])
        key = (p1, p2, match_uuid)
        if key not in self._seen_edges:
            self._seen_edges.add(key)
            self.edges.append(LobbyEdge(player_a=p1, player_b=p2, match_uuid=match_uuid))

    def get_tier_distribution(self) -> dict[str, int]:
        """Return count of scanned/app-user players by tier."""
        distribution: dict[str, int] = defaultdict(int)
        for node in self.nodes.values():
            if node.scanned and node.is_app_user:
                distribution[node.tier] += 1
            elif node.scanned and node.tier != "UNKNOWN":
                distribution[node.tier] += 1
        return dict(distribution)

    def get_games_tier_distribution(self) -> dict[str, int]:
        """Return count of games in manifest by tier."""
        distribution: dict[str, int] = defaultdict(int)
        for game in self.games.values():
            distribution[game.tier] += 1
        return dict(distribution)

    def get_next_priority_player(
        self,
        *,
        target_tiers: set[str] | None = None,
        max_depth: int | None = None,
        max_players_per_tier: int | None = None,
        max_games_per_tier: int | None = None,
    ) -> PlayerNode | None:
        """Select the next unscanned player node with highest priority score.

        Priority score strictly favors:
        1. Leagues with the fewest candidate games and scanned players (e.g. Iron, Bronze, Silver, Challenger).
        2. Players matching target_tiers filters.
        3. Smaller lobby depth (closer to root seeds).
        """
        tier_counts = self.get_tier_distribution()
        games_counts = self.get_games_tier_distribution()

        unscanned = [
            node for node in self.nodes.values()
            if not node.scanned
            and (max_depth is None or node.depth <= max_depth)
            and (target_tiers is None or node.tier in target_tiers or node.tier == "UNKNOWN")
            and (
                max_players_per_tier is None
                or node.tier == "UNKNOWN"
                or tier_counts.get(node.tier, 0) < max_players_per_tier
            )
            and (
                max_games_per_tier is None
                or node.tier == "UNKNOWN"
                or games_counts.get(node.tier, 0) < max_games_per_tier
            )
        ]

        if not unscanned:
            return None

        def _priority_key(node: PlayerNode) -> tuple[float, int, int]:
            count_players = tier_counts.get(node.tier, 0)
            count_games = games_counts.get(node.tier, 0)
            # Rarity boost: leagues with fewer games & players get massive priority
            rarity_score = 1000000.0 / (1.0 + count_games * 2 + count_players)
            # Target tier boost
            if target_tiers and node.tier in target_tiers:
                rarity_score *= 10.0
            # App user boost
            app_score = 10 if node.is_app_user else 1
            if node.app_matches > 0:
                app_score += node.app_matches
            # Depth penalty: explore shallow first
            depth_penalty = node.depth * 5
            total_score = (rarity_score * app_score) - depth_penalty
            return (total_score, node.app_matches, -node.depth)

        unscanned.sort(key=_priority_key, reverse=True)
        return unscanned[0]

    def save_csv(self, root_dir: str | Path) -> None:
        """Persist graph nodes, games manifest, and edges to CSV files atomically."""
        root = Path(root_dir)
        root.mkdir(parents=True, exist_ok=True)

        # 1. Write players.csv atomically
        players_path = root / "players.csv"
        tmp_players = root / "players.csv.tmp"
        with tmp_players.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "riot_id",
                    "region",
                    "game_name",
                    "tag_line",
                    "tier",
                    "rank_text",
                    "app_matches",
                    "depth",
                    "is_app_user",
                    "scanned",
                    "discovered_from",
                ],
            )
            writer.writeheader()
            for node in self.nodes.values():
                writer.writerow({
                    "riot_id": node.riot_id,
                    "region": node.region,
                    "game_name": node.game_name,
                    "tag_line": node.tag_line,
                    "tier": node.tier,
                    "rank_text": node.rank_text,
                    "app_matches": node.app_matches,
                    "depth": node.depth,
                    "is_app_user": "1" if node.is_app_user else "0",
                    "scanned": "1" if node.scanned else "0",
                    "discovered_from": node.discovered_from or "",
                })
        tmp_players.replace(players_path)

        # 2. Write games.csv atomically
        games_path = root / "games.csv"
        tmp_games = root / "games.csv.tmp"
        with tmp_games.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "match_uuid",
                    "timeline_url",
                    "focal_player_riot_id",
                    "tier",
                    "avg_rating",
                    "tft_set",
                ],
            )
            writer.writeheader()
            for game in self.games.values():
                writer.writerow({
                    "match_uuid": game.match_uuid,
                    "timeline_url": game.timeline_url,
                    "focal_player_riot_id": game.focal_player_riot_id,
                    "tier": game.tier,
                    "avg_rating": game.avg_rating or "",
                    "tft_set": game.tft_set,
                })
        tmp_games.replace(games_path)

        # 3. Write edges.csv atomically
        edges_path = root / "edges.csv"
        tmp_edges = root / "edges.csv.tmp"
        with tmp_edges.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["player_a", "player_b", "match_uuid"],
            )
            writer.writeheader()
            for edge in self.edges:
                writer.writerow({
                    "player_a": edge.player_a,
                    "player_b": edge.player_b,
                    "match_uuid": edge.match_uuid,
                })
        tmp_edges.replace(edges_path)

        logger.info(
            "Saved PlayerGraph to %s: %d players, %d games, %d edges",
            root,
            len(self.nodes),
            len(self.games),
            len(self.edges),
        )

    def load_csv(self, root_dir: str | Path) -> None:
        """Load graph nodes, games manifest, and edges from CSV files if present."""
        root = Path(root_dir)
        if not root.exists():
            return

        players_path = root / "players.csv"
        if players_path.exists():
            with players_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rid = row.get("riot_id")
                    if not rid:
                        continue
                    raw_rank = row.get("rank_text", "UNRANKED")
                    raw_tier = row.get("tier", "UNKNOWN")
                    tier_norm = normalize_tier(raw_rank) if raw_rank and raw_rank != "UNRANKED" else normalize_tier(raw_tier)
                    self.nodes[rid] = PlayerNode(
                        riot_id=rid,
                        region=row.get("region", "na1"),
                        game_name=row.get("game_name", ""),
                        tag_line=row.get("tag_line", ""),
                        tier=tier_norm,
                        rank_text=raw_rank,
                        app_matches=int(row.get("app_matches", 0)),
                        depth=int(row.get("depth", 0)),
                        is_app_user=row.get("is_app_user") in ("1", "True", "true"),
                        scanned=row.get("scanned") in ("1", "True", "true"),
                        discovered_from=row.get("discovered_from") or None,
                    )

        games_path = root / "games.csv"
        if games_path.exists():
            with games_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    muuid = row.get("match_uuid")
                    if not muuid:
                        continue
                    self.games[muuid] = GameManifestEntry(
                        match_uuid=muuid,
                        timeline_url=row.get("timeline_url", ""),
                        focal_player_riot_id=row.get("focal_player_riot_id", ""),
                        tier=row.get("tier", "UNKNOWN"),
                        avg_rating=row.get("avg_rating") or None,
                        tft_set=row.get("tft_set", "TFTSet18"),
                    )

        edges_path = root / "edges.csv"
        if edges_path.exists():
            with edges_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    p1 = row.get("player_a")
                    p2 = row.get("player_b")
                    muuid = row.get("match_uuid")
                    if p1 and p2 and muuid:
                        self.add_edge(p1, p2, muuid)
