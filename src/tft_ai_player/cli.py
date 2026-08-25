"""Command-line entry points for MetaTFT discovery and timeline extraction."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from tqdm import tqdm

from .dataset import PlayerCsvWriter, TimelineValidationError, extract_pvp_rounds
from .metatft import LeaderboardPlayer, MetaTftClient, MetaTftRequestError, TrackedTimelineCandidate


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return a process exit code."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "profile":
            return _collect_profile(args)
        if args.command == "collect":
            return _collect_leaderboard(args)
        if args.command == "timeline":
            return _collect_timeline(args)
    except (MetaTftRequestError, TimelineValidationError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    parser.error(f"unsupported command: {args.command}")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tft-ai-player")
    subcommands = parser.add_subparsers(dest="command", required=True)

    profile_parser = subcommands.add_parser(
        "profile",
        help="collect a small number of CSV game files from one player",
    )
    profile_parser.add_argument("--region", required=True, help="Riot platform, for example LA2")
    profile_parser.add_argument("--game-name", required=True, help="Riot game name")
    profile_parser.add_argument("--tag-line", required=True, help="Riot tagline")
    profile_parser.add_argument("--tft-set", default="TFTSet17", help="TFT set to retain")
    profile_parser.add_argument("--games", type=int, default=1, help="maximum games to collect")
    profile_parser.add_argument(
        "-o",
        "--output",
        "--output-dir",
        dest="output",
        type=Path,
        default=Path("data"),
        help="destination directory where games/*.csv will be written (default: data)",
    )
    profile_parser.add_argument(
        "--request-interval",
        type=float,
        default=1.5,
        help="minimum seconds between API requests (default: 1.5)",
    )
    profile_parser.add_argument(
        "--allowed-queues",
        type=int,
        nargs="+",
        default=[1100],
        help="Riot queue IDs to retain (default: [1100] for Ranked TFT)",
    )

    collect_parser = subcommands.add_parser(
        "collect",
        help="iterate tracked leaderboard players and write one CSV per selected game",
    )
    collect_parser.add_argument("--players", type=int, default=10, help="number of leaderboard players to sample")
    collect_parser.add_argument(
        "--games-per-player",
        type=int,
        default=None,
        help="optional cap on games contributed by each player (default: unlimited)",
    )
    collect_parser.add_argument(
        "--max-games",
        type=int,
        help="optional cap across all players",
    )
    collect_parser.add_argument(
        "--leaderboard-offset",
        type=int,
        default=0,
        help="leaderboard offset used to choose a different player cohort",
    )
    collect_parser.add_argument("--tft-set", default="TFTSet17", help="TFT set to retain")
    collect_parser.add_argument(
        "-o",
        "--output",
        "--output-dir",
        dest="output",
        type=Path,
        default=Path("data"),
        help="destination directory where games/*.csv will be written (default: data)",
    )
    collect_parser.add_argument(
        "--request-interval",
        type=float,
        default=1.5,
        help="minimum seconds between API requests (default: 1.5)",
    )
    collect_parser.add_argument(
        "--allowed-queues",
        type=int,
        nargs="+",
        default=[1100],
        help="Riot queue IDs to retain (default: [1100] for Ranked TFT)",
    )

    timeline_parser = subcommands.add_parser(
        "timeline",
        help="fetch a known MetaTFT timeline URL and write one game CSV",
    )
    timeline_parser.add_argument("--timeline-url", required=True, help="known MetaTFT timeline JSON URL")
    timeline_parser.add_argument("--match-id", required=True, help="game ID used for the CSV file name")
    timeline_parser.add_argument("--tft-set", default="TFTSet17", help="TFT set for this match")
    timeline_parser.add_argument("--game-version", default="unknown", help="Riot patch, for example 16.16")
    timeline_parser.add_argument(
        "-o",
        "--output",
        "--output-dir",
        dest="output",
        type=Path,
        default=Path("data"),
        help="destination directory where games/*.csv will be written (default: data)",
    )
    timeline_parser.add_argument(
        "--request-interval",
        type=float,
        default=1.5,
        help="minimum seconds between API requests (default: 1.5)",
    )

    return parser


def _collect_profile(args: argparse.Namespace) -> int:
    client = MetaTftClient(
        minimum_request_interval_seconds=getattr(args, "request_interval", 1.5),
        retry_count=5,
    )
    writer = PlayerCsvWriter(args.output)
    player = LeaderboardPlayer(
        region=args.region,
        game_name=args.game_name,
        tag_line=args.tag_line,
        player_id=None,
    )
    profile = client.fetch_profile(
        region=player.region,
        game_name=player.game_name,
        tag_line=player.tag_line,
        tft_set=args.tft_set,
    )
    allowed_queues = getattr(args, "allowed_queues", (1100,))
    candidates = client.tracked_timeline_candidates(
        profile,
        tft_set=args.tft_set,
        allowed_queue_ids=allowed_queues,
    )
    seen_game_ids = _existing_game_ids(args.output)
    written, skipped = _download_player_games(
        client=client,
        writer=writer,
        player=player,
        candidates=candidates,
        tft_set=args.tft_set,
        games_per_player=args.games,
        seen_game_ids=seen_game_ids,
        remaining_games_budget=args.games,
    )
    print(f"collected profile {player.riot_id}: {written} new games written, {skipped} existing games skipped")
    return 0


def _collect_leaderboard(args: argparse.Namespace) -> int:
    if args.players <= 0:
        raise ValueError("players must be positive")
    if args.games_per_player is not None and args.games_per_player <= 0:
        raise ValueError("games_per_player must be positive when provided")
    if args.max_games is not None and args.max_games <= 0:
        raise ValueError("max_games must be positive when provided")

    client = MetaTftClient(
        minimum_request_interval_seconds=getattr(args, "request_interval", 1.5),
        retry_count=5,
    )
    writer = PlayerCsvWriter(args.output)
    players = client.fetch_leaderboard_players(count=args.players, offset=args.leaderboard_offset)
    seen_game_ids = _existing_game_ids(args.output)
    allowed_queues = getattr(args, "allowed_queues", (1100,))

    total_written = 0
    total_skipped = 0

    with tqdm(players, desc="collecting players", unit="player") as player_progress:
        for player in player_progress:
            player_progress.set_postfix({
                "player": player.riot_id,
                "new": total_written,
                "cached": total_skipped,
            })
            if args.max_games is not None and (total_written + total_skipped) >= args.max_games:
                break

            try:
                profile = client.fetch_profile(
                    region=player.region,
                    game_name=player.game_name,
                    tag_line=player.tag_line,
                    tft_set=args.tft_set,
                )
                candidates = client.tracked_timeline_candidates(
                    profile,
                    tft_set=args.tft_set,
                    allowed_queue_ids=allowed_queues,
                )
            except MetaTftRequestError as error:
                tqdm.write(f"skipped player {player.riot_id}: {error}", file=sys.stderr)
                continue

            remaining_budget = (
                (args.max_games - (total_written + total_skipped))
                if args.max_games is not None
                else None
            )
            written, skipped = _download_player_games(
                client=client,
                writer=writer,
                player=player,
                candidates=candidates,
                tft_set=args.tft_set,
                games_per_player=args.games_per_player,
                seen_game_ids=seen_game_ids,
                remaining_games_budget=remaining_budget,
            )
            total_written += written
            total_skipped += skipped
            player_progress.set_postfix({
                "player": player.riot_id,
                "new": total_written,
                "cached": total_skipped,
            })

    print(f"collection finished: {total_written} new games written, {total_skipped} existing games skipped across player CSV files")
    return 0


def _collect_timeline(args: argparse.Namespace) -> int:
    client = MetaTftClient(
        minimum_request_interval_seconds=getattr(args, "request_interval", 1.5),
        retry_count=5,
    )
    writer = PlayerCsvWriter(args.output)
    timeline = client.fetch_timeline(args.timeline_url)
    observations = extract_pvp_rounds(
        timeline,
        match_id=args.match_id,
        tft_set=args.tft_set,
        game_version=args.game_version,
    )
    path = writer.write_game(observations)
    if path is None:
        writer.add_to_blacklist(args.match_id, reason="no valid PVP rounds")
        print("no valid PVP observations found; match blacklisted and no CSV written", file=sys.stderr)
        return 0
    print(f"wrote {len(observations)} PVP observations to {path}")
    return 0


def _download_player_games(
    *,
    client: MetaTftClient,
    writer: PlayerCsvWriter,
    player: LeaderboardPlayer,
    candidates: Sequence[TrackedTimelineCandidate],
    tft_set: str,
    games_per_player: int | None,
    seen_game_ids: set[str],
    remaining_games_budget: int | None,
) -> tuple[int, int]:
    """Download up to games_per_player unique games for one player, returning (written, skipped)."""

    written = 0
    skipped = 0
    for candidate in candidates:
        if games_per_player is not None and (written + skipped) >= games_per_player:
            break
        if remaining_games_budget is not None and (written + skipped) >= remaining_games_budget:
            break

        game_id = candidate.app_match_uuid
        if game_id in seen_game_ids:
            skipped += 1
            continue

        try:
            timeline = client.fetch_timeline(candidate.timeline_url)
        except MetaTftRequestError as error:
            # Temporary connection or HTTP request error: do NOT blacklist to allow retry in future runs.
            tqdm.write(f"skipped game {game_id} from {player.riot_id} (temporary request error): {error}", file=sys.stderr)
            seen_game_ids.add(game_id)
            continue

        try:
            observations = extract_pvp_rounds(
                timeline,
                match_id=game_id,
                tft_set=tft_set,
                game_version=candidate.game_version or "unknown",
                focal_tier=candidate.focal_tier,
                focal_rating_numeric=candidate.focal_rating_numeric,
                avg_match_rating=candidate.avg_match_rating,
                avg_match_rating_numeric=candidate.avg_match_rating_numeric,
                focal_augments=candidate.focal_augments,
            )
        except (TimelineValidationError, ValueError) as error:
            writer.add_to_blacklist(game_id, reason=f"validation error: {error}")
            seen_game_ids.add(game_id)
            tqdm.write(f"blacklisted game {game_id} from {player.riot_id} (validation error): {error}", file=sys.stderr)
            continue

        path = writer.write_player_game(
            observations,
            collected_from_riot_id=player.riot_id,
            collected_from_region=player.region,
            match_id_ow=candidate.match_id_ow,
        )
        if path is None:
            writer.add_to_blacklist(game_id, reason="no valid PVP rounds")
            seen_game_ids.add(game_id)
            tqdm.write(f"blacklisted game {game_id} from {player.riot_id}: no valid PVP rounds")
            continue

        seen_game_ids.add(game_id)
        written += 1

    return written, skipped


def _existing_game_ids(root: Path) -> set[str]:
    import csv as _csv

    seen: set[str] = set()
    players_dir = root / "players"
    if players_dir.exists():
        for path in players_dir.glob("*.csv"):
            try:
                with path.open(newline="", encoding="utf-8") as file:
                    reader = _csv.DictReader(file)
                    for row in reader:
                        match_id = row.get("match_id")
                        if match_id:
                            seen.add(match_id)
            except Exception:
                continue

    games_dir = root / "games"
    if games_dir.exists():
        seen.update(path.stem for path in games_dir.glob("*.csv"))

    seen.update(PlayerCsvWriter(root).load_blacklist())
    return seen