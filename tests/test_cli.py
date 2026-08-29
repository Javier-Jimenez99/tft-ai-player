from __future__ import annotations

import csv
import json
from argparse import Namespace
from pathlib import Path

from tft_ai_player import cli
from tft_ai_player.metatft import LeaderboardPlayer, TrackedTimelineCandidate


def test_collect_writes_one_csv_for_each_player_game(tmp_path: Path, monkeypatch) -> None:
    players = [_player("Alpha"), _player("Bravo")]
    candidates = {
        "Alpha": [_candidate("alpha-game")],
        "Bravo": [_candidate("bravo-game")],
    }

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def fetch_leaderboard_players(self, **_: int) -> list[LeaderboardPlayer]:
            return players

        def fetch_profile(self, *, game_name: str, **_: str) -> dict[str, str]:
            return {"game_name": game_name}

        def tracked_timeline_candidates(
            self,
            profile: dict[str, str],
            **_: str,
        ) -> list[TrackedTimelineCandidate]:
            return candidates[profile["game_name"]]

        def fetch_timeline(self, _: str) -> dict[str, object]:
            return _timeline()

    monkeypatch.setattr(cli, "MetaTftClient", FakeClient)
    args = Namespace(
        players=2,
        games_per_player=1,
        max_games=2,
        leaderboard_offset=0,
        tft_set="TFTSet17",
        output=tmp_path,
    )

    assert cli._collect_leaderboard(args) == 0

    paths = sorted((tmp_path / "players").glob("*.csv"))
    assert [path.name for path in paths] == ["la2_Alpha_LAS.csv", "la2_Bravo_LAS.csv"]
    with paths[0].open(newline="", encoding="utf-8") as input_file:
        row = next(csv.DictReader(input_file))
    assert row["collected_from_riot_id"] == "Alpha#LAS"
    assert row["match_id"] == "alpha-game"


def test_collect_skips_already_downloaded_games_on_resume(tmp_path: Path, monkeypatch) -> None:
    players = [_player("Alpha"), _player("Bravo")]
    candidates = {
        "Alpha": [_candidate("alpha-game")],
        "Bravo": [_candidate("bravo-game")],
    }
    fetch_timeline_calls: list[str] = []

    # Pre-create la2_Alpha_LAS.csv on disk with alpha-game
    players_dir = tmp_path / "players"
    players_dir.mkdir(parents=True, exist_ok=True)
    with (players_dir / "la2_Alpha_LAS.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["match_id", "round_stage"])
        writer.writeheader()
        writer.writerow({"match_id": "alpha-game", "round_stage": "2-2"})

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def fetch_leaderboard_players(self, **_: int) -> list[LeaderboardPlayer]:
            return players

        def fetch_profile(self, *, game_name: str, **_: str) -> dict[str, str]:
            return {"game_name": game_name}

        def tracked_timeline_candidates(
            self,
            profile: dict[str, str],
            **_: str,
        ) -> list[TrackedTimelineCandidate]:
            return candidates[profile["game_name"]]

        def fetch_timeline(self, url: str) -> dict[str, object]:
            fetch_timeline_calls.append(url)
            return _timeline()

    monkeypatch.setattr(cli, "MetaTftClient", FakeClient)
    args = Namespace(
        players=2,
        games_per_player=1,
        max_games=2,
        leaderboard_offset=0,
        tft_set="TFTSet17",
        output=tmp_path,
    )

    assert cli._collect_leaderboard(args) == 0

    # Only bravo-game timeline should have been fetched
    assert fetch_timeline_calls == ["https://matches3.metatft.com/bravo-game.json"]
    paths = sorted(players_dir.glob("*.csv"))
    assert [path.name for path in paths] == ["la2_Alpha_LAS.csv", "la2_Bravo_LAS.csv"]


def test_collect_blacklists_empty_games_and_skips_on_resume(tmp_path: Path, monkeypatch) -> None:
    players = [_player("Alpha")]
    candidates = {"Alpha": [_candidate("empty-game"), _candidate("valid-game")]}
    fetch_timeline_calls: list[str] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def fetch_leaderboard_players(self, **_: int) -> list[LeaderboardPlayer]:
            return players

        def fetch_profile(self, *, game_name: str, **_: str) -> dict[str, str]:
            return {"game_name": game_name}

        def tracked_timeline_candidates(
            self,
            profile: dict[str, str],
            **_: str,
        ) -> list[TrackedTimelineCandidate]:
            return candidates[profile["game_name"]]

        def fetch_timeline(self, url: str) -> dict[str, object]:
            fetch_timeline_calls.append(url)
            if "empty-game" in url:
                # Stage data with no PVP rounds
                return {
                    "summoner_name": "Focal",
                    "stage_data": json.dumps([
                        {
                            "me": {"summoner_name": "Focal"},
                            "match_info": {"round_type": {"stage": "1-2", "name": "Minions", "type": "PVE"}},
                        }
                    ]),
                }
            return _timeline()

    monkeypatch.setattr(cli, "MetaTftClient", FakeClient)
    args = Namespace(
        players=1,
        games_per_player=2,
        max_games=2,
        leaderboard_offset=0,
        tft_set="TFTSet17",
        output=tmp_path,
    )

    # First run: downloads empty-game (blacklists it) and valid-game
    assert cli._collect_leaderboard(args) == 0
    blacklist_path = tmp_path / "blacklisted_games.txt"
    assert blacklist_path.exists()
    assert "empty-game" in blacklist_path.read_text(encoding="utf-8")

    # Second run: empty-game should be skipped without fetching
    fetch_timeline_calls.clear()
    assert cli._collect_leaderboard(args) == 0
    assert "https://matches3.metatft.com/empty-game.json" not in fetch_timeline_calls


def test_collect_does_not_blacklist_on_connection_error(tmp_path: Path, monkeypatch) -> None:
    from tft_ai_player.metatft import MetaTftRequestError

    players = [_player("Alpha")]
    candidates = {"Alpha": [_candidate("network-fail-game")]}

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def fetch_leaderboard_players(self, **_: int) -> list[LeaderboardPlayer]:
            return players

        def fetch_profile(self, *, game_name: str, **_: str) -> dict[str, str]:
            return {"game_name": game_name}

        def tracked_timeline_candidates(
            self,
            profile: dict[str, str],
            **_: str,
        ) -> list[TrackedTimelineCandidate]:
            return candidates[profile["game_name"]]

        def fetch_timeline(self, url: str) -> dict[str, object]:
            raise MetaTftRequestError("connection reset by peer")

    monkeypatch.setattr(cli, "MetaTftClient", FakeClient)
    args = Namespace(
        players=1,
        games_per_player=1,
        max_games=1,
        leaderboard_offset=0,
        tft_set="TFTSet17",
        output=tmp_path,
    )

    assert cli._collect_leaderboard(args) == 0
    blacklist_path = tmp_path / "blacklisted_games.txt"
    if blacklist_path.exists():
        assert "network-fail-game" not in blacklist_path.read_text(encoding="utf-8")


def test_collect_blacklists_player_after_5_consecutive_no_pvp_games(tmp_path: Path, monkeypatch) -> None:
    players = [_player("BadPlayer"), _player("GoodPlayer")]
    # BadPlayer has 6 invalid games in a row; game 6 should not even be fetched because player is skipped after 5
    candidates = {
        "BadPlayer": [_candidate(f"bad-game-{i}") for i in range(1, 7)],
        "GoodPlayer": [_candidate("good-game-1")],
    }
    fetch_timeline_calls: list[str] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def fetch_leaderboard_players(self, **_: int) -> list[LeaderboardPlayer]:
            return players

        def fetch_profile(self, *, game_name: str, **_: str) -> dict[str, str]:
            return {"game_name": game_name}

        def tracked_timeline_candidates(
            self,
            profile: dict[str, str],
            **_: str,
        ) -> list[TrackedTimelineCandidate]:
            return candidates[profile["game_name"]]

        def fetch_timeline(self, url: str) -> dict[str, object]:
            fetch_timeline_calls.append(url)
            if "bad-game" in url:
                # Stage data with no PVP rounds (PVE only)
                return {
                    "summoner_name": "Focal",
                    "stage_data": json.dumps([
                        {
                            "me": {"summoner_name": "Focal"},
                            "match_info": {"round_type": {"stage": "1-2", "name": "Minions", "type": "PVE"}},
                        }
                    ]),
                }
            return _timeline()

    monkeypatch.setattr(cli, "MetaTftClient", FakeClient)
    args = Namespace(
        players=2,
        games_per_player=10,
        max_games=10,
        leaderboard_offset=0,
        tft_set="TFTSet17",
        output=tmp_path,
    )

    # First run: downloads 5 bad games from BadPlayer, blacklists player, skips bad-game-6, and processes GoodPlayer
    assert cli._collect_leaderboard(args) == 0

    assert "https://matches3.metatft.com/bad-game-6.json" not in fetch_timeline_calls
    assert "https://matches3.metatft.com/good-game-1.json" in fetch_timeline_calls

    player_blacklist_path = tmp_path / "blacklisted_players.txt"
    assert player_blacklist_path.exists()
    assert "BadPlayer#LAS" in player_blacklist_path.read_text(encoding="utf-8")

    # Second run: BadPlayer should be skipped entirely at the profile level
    fetch_timeline_calls.clear()
    assert cli._collect_leaderboard(args) == 0
    # No calls for BadPlayer
    assert not any("bad-game" in call for call in fetch_timeline_calls)


def test_collect_profile_writes_csv(tmp_path: Path, monkeypatch) -> None:
    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def fetch_profile(self, *, game_name: str, **_: str) -> dict[str, str]:
            return {"game_name": game_name}

        def tracked_timeline_candidates(self, profile: dict[str, str], **_: str) -> list[TrackedTimelineCandidate]:
            return [_candidate("profile-game-1")]

        def fetch_timeline(self, _: str) -> dict[str, object]:
            return _timeline()

    monkeypatch.setattr(cli, "MetaTftClient", FakeClient)
    args = Namespace(
        region="la2",
        game_name="SoloPlayer",
        tag_line="LAS",
        tft_set="TFTSet17",
        games=1,
        output=tmp_path,
    )
    assert cli._collect_profile(args) == 0
    csv_path = tmp_path / "players" / "la2_SoloPlayer_LAS.csv"
    assert csv_path.exists()
    assert "profile-game-1" in csv_path.read_text(encoding="utf-8")


def test_collect_profile_skips_blacklisted_player(tmp_path: Path, monkeypatch) -> None:
    # Blacklist player first
    writer = cli.PlayerCsvWriter(tmp_path)
    writer.add_player_to_blacklist("SoloPlayer#LAS", reason="test")

    fetch_called = False

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def fetch_profile(self, *args, **kwargs) -> dict[str, str]:
            nonlocal fetch_called
            fetch_called = True
            return {}

    monkeypatch.setattr(cli, "MetaTftClient", FakeClient)
    args = Namespace(
        region="la2",
        game_name="SoloPlayer",
        tag_line="LAS",
        tft_set="TFTSet17",
        games=1,
        output=tmp_path,
    )
    assert cli._collect_profile(args) == 0
    assert not fetch_called


def test_collect_timeline_empty_blacklists_match(tmp_path: Path, monkeypatch) -> None:
    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def fetch_timeline(self, _: str) -> dict[str, object]:
            return {
                "summoner_name": "Focal",
                "stage_data": json.dumps([
                    {
                        "me": {"summoner_name": "Focal"},
                        "match_info": {"round_type": {"stage": "1-2", "name": "Minions", "type": "PVE"}},
                    }
                ]),
            }

    monkeypatch.setattr(cli, "MetaTftClient", FakeClient)
    args = Namespace(
        timeline_url="https://matches3.metatft.com/empty-timeline.json",
        match_id="empty-timeline-id",
        tft_set="TFTSet17",
        game_version="16.16",
        output=tmp_path,
    )
    assert cli._collect_timeline(args) == 0
    blacklist_path = tmp_path / "blacklisted_games.txt"
    assert blacklist_path.exists()
    assert "empty-timeline-id" in blacklist_path.read_text(encoding="utf-8")


def _player(game_name: str) -> LeaderboardPlayer:
    return LeaderboardPlayer(region="la2", game_name=game_name, tag_line="LAS", player_id=None)


def _candidate(game_id: str) -> TrackedTimelineCandidate:
    return TrackedTimelineCandidate(
        app_match_uuid=game_id,
        match_id_ow=f"{game_id}-internal",
        timeline_url=f"https://matches3.metatft.com/{game_id}.json",
        tft_set="TFTSet17",
        game_version="16.16",
        riot_match_id=None,
        created_timestamp=None,
        player_id=None,
    )


def _timeline() -> dict[str, object]:
    return {
        "summoner_name": "Focal",
        "stage_data": json.dumps(
            [
                {
                    "me": {"summoner_name": "Focal", "gold": 20, "xp": {"level": 4}},
                    "round_start_health": {
                        "player_status": {
                            "Focal": {"health": 90},
                            "Opponent": {"health": 88},
                        }
                    },
                    "match_info": {
                        "round_type": {"stage": "2-2", "name": "PVP", "type": "PVP"},
                        "opponent": {"name": "Opponent"},
                        "round_outcome": {"Focal": {"outcome": "victory"}},
                    },
                    "matchup_boards": {
                        "player_board": [{"unit": "TFT17_Aatrox"}],
                        "opponent_board": [{"unit": "TFT17_Veigar"}],
                    },
                }
            ]
        ),
    }