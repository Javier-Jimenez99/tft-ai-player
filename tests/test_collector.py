from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from tft_ai_player.dataset.collector_db import ALL_STANDARD_TIERS, CollectorDb, DbGame, DbPlayer
from tft_ai_player.dataset.collector_service import ContinuousCollectorService, DEFAULT_TARGET_WEIGHTS


def test_collector_db_basic_crud() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        db = CollectorDb(db_path)
        try:
            # 1. Add players
            assert db.add_player(
                riot_id="PlayerA#NA1",
                region="na1",
                game_name="PlayerA",
                tag_line="NA1",
                tier="BRONZE",
                rank_text="BRONZE I 50 LP",
            )
            assert db.add_player(
                riot_id="PlayerB#EUW",
                region="euw1",
                game_name="PlayerB",
                tag_line="EUW",
                tier="CHALLENGER",
                rank_text="CHALLENGER I 800 LP",
                app_matches=10,
            )

            # 2. Query unscanned players
            bronze_p = db.get_next_unscanned_player(tier="BRONZE")
            assert bronze_p is not None
            assert bronze_p.riot_id == "PlayerA#NA1"
            assert bronze_p.tier == "BRONZE"

            # 3. Mark scanned
            db.mark_player_scanned("PlayerA#NA1", app_matches=3, tier="BRONZE", rank_text="BRONZE I 75 LP")
            assert db.get_next_unscanned_player(tier="BRONZE") is None

            # 4. Add candidate games
            games = [
                DbGame(
                    match_uuid="match-1",
                    timeline_url="https://matches3.metatft.com/match-1.json",
                    focal_player_riot_id="PlayerA#NA1",
                    tier="BRONZE",
                ),
                DbGame(
                    match_uuid="match-2",
                    timeline_url="https://matches3.metatft.com/match-2.json",
                    focal_player_riot_id="PlayerB#EUW",
                    tier="CHALLENGER",
                ),
            ]
            assert db.add_games_batch(games) == 2

            # 5. Fetch next pending game
            next_bronze_game = db.get_next_pending_game(tier="BRONZE")
            assert next_bronze_game is not None
            assert next_bronze_game.match_uuid == "match-1"

            # 6. Record download
            db.record_match_downloaded(match_uuid="match-1", tier="BRONZE", focal_player="PlayerA#NA1")
            assert db.get_next_pending_game(tier="BRONZE") is None
            assert db.total_downloaded_count() == 1

            # 7. Check tier counts
            counts = db.get_tier_counts()
            assert counts["downloaded"]["BRONZE"] == 1
            assert counts["pending"]["CHALLENGER"] == 1
            assert counts["app_users"]["CHALLENGER"] == 1
        finally:
            db.close()


def test_collector_db_aging_app_users() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "aging.db"
        db = CollectorDb(db_path)
        try:
            # Add app user scanned 10 days ago
            ten_days_ago = time.time() - (10 * 86400.0)
            db.add_player(
                riot_id="OldAppUser#NA1",
                region="na1",
                game_name="OldAppUser",
                tag_line="NA1",
                tier="GOLD",
                is_app_user=True,
                scanned=True,
            )
            db.conn.execute(
                "UPDATE players SET last_scanned_at = ?, is_app_user = 1 WHERE riot_id = 'OldAppUser#NA1';",
                (ten_days_ago,),
            )

            # Add app user scanned 1 hour ago
            recent = time.time() - 3600.0
            db.add_player(
                riot_id="RecentAppUser#NA1",
                region="na1",
                game_name="RecentAppUser",
                tag_line="NA1",
                tier="GOLD",
                is_app_user=True,
                scanned=True,
            )
            db.conn.execute(
                "UPDATE players SET last_scanned_at = ?, is_app_user = 1 WHERE riot_id = 'RecentAppUser#NA1';",
                (recent,),
            )

            aging = db.get_aging_app_users(min_age_seconds=5 * 86400.0, tier="GOLD")
            assert len(aging) == 1
            assert aging[0].riot_id == "OldAppUser#NA1"
        finally:
            db.close()


def test_collector_service_deficits_and_scheduling() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "sched.db"
        out_dir = Path(tmp_dir) / "data"
        status_path = Path(tmp_dir) / "status.json"

        service = ContinuousCollectorService(
            db_path=db_path,
            output_dir=out_dir,
            status_file=status_path,
        )
        try:
            # Initially, all tiers have 0 downloaded matches, deficit matches target weights
            deficits = service.compute_deficits()
            assert deficits[0][0] == "MASTER"

            # Simulate that MASTER has 100 downloaded matches, but others have 0
            for i in range(100):
                service.db.record_match_downloaded(
                    match_uuid=f"master-{i}",
                    tier="MASTER",
                    focal_player=f"MasterPlayer{i}#NA1",
                )

            # Now MASTER is over-represented (100% of matches vs 20% target weight)
            new_deficits = service.compute_deficits()
            top_tier, top_deficit = new_deficits[0]
            assert top_tier in ("DIAMOND", "GRANDMASTER")
            master_deficit = next(d for t, d in new_deficits if t == "MASTER")
            assert master_deficit < 0

            # Heartbeat verification
            service.write_status_heartbeat()
            assert status_path.exists()
            with status_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                assert data["service"] == "tft-ai-collector"
                assert data["total_matches_on_disk"] == 100
                assert "MASTER" in data["distribution"]
                assert data["distribution"]["MASTER"]["downloaded"] == 100
        finally:
            service.db.close()


def test_collector_step_download_and_lobby_recycling() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "step.db"
        out_dir = Path(tmp_dir) / "data"

        service = ContinuousCollectorService(
            db_path=db_path,
            output_dir=out_dir,
            target_weights={"BRONZE": 1.0},
            request_interval=0.0,
        )
        try:
            # Add candidate game for BRONZE
            service.db.add_games_batch([
                DbGame(
                    match_uuid="test-bronze-uuid-1",
                    timeline_url="https://matches3.metatft.com/test-bronze-uuid-1.json",
                    focal_player_riot_id="FocalBronze#EUW",
                    tier="BRONZE",
                )
            ])

            mock_timeline = {
                "datetime": "2026-03-01T12:00:00Z",
                "summoner_name": "FocalBronze",
                "tagline": "EUW",
                "stage_data": json.dumps([
                    {
                        "me": {"summoner_name": "FocalBronze", "gold": 20, "xp": {"level": 4}},
                        "round_start_health": {
                            "player_status": {
                                "FocalBronze": {"health": 100, "xp": 4, "gold": 20},
                                "OpponentBronze": {"health": 100, "xp": 4, "gold": 20},
                            }
                        },
                        "match_info": {
                            "round_type": {"stage": "2-1", "name": "Augment", "type": "PVP"},
                            "opponent": {"name": "OpponentBronze"},
                            "round_outcome": {"FocalBronze": {"outcome": "victory"}},
                        },
                        "matchup_boards": {
                            "player_board": [{"unit": "TFT18_Renata", "tier": 1, "loc": "D1", "items": []}],
                            "opponent_board": [{"unit": "TFT18_Singed", "tier": 1, "loc": "A1", "items": []}],
                        },
                    }
                ]),
                "board_players": [
                    {"board": [{"summoner": "FocalBronze", "tag_line": "EUW"}]},
                    {"board": [{"summoner": "OpponentBronze", "tag_line": "EUW"}]},
                ],
            }

            with patch.object(service.client, "fetch_timeline", return_value=mock_timeline):
                work_done = service.step()
                assert work_done is True

                # Match should be marked downloaded
                assert service.db.total_downloaded_count() == 1
                assert service.stats.total_downloaded_session == 1

                # Lobby recycling: OpponentBronze#EUW should now be in players table!
                recycled = service.db.get_next_unscanned_player(tier="BRONZE")
                assert recycled is not None
                assert recycled.riot_id == "OpponentBronze#EUW"
                assert recycled.tier == "BRONZE"
                assert recycled.discovered_from == "test-bronze-uuid-1"
        finally:
            service.db.close()


def test_scheduler_falls_back_when_starved_tier_has_no_candidates() -> None:
    """Ensure that if the highest-deficit tier has no games or players, the service does not get stuck and downloads available games."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "sched_fallback.db"
        out_dir = Path(tmp_dir) / "data"

        service = ContinuousCollectorService(
            db_path=db_path,
            output_dir=out_dir,
            target_weights={"CHALLENGER": 0.8, "DIAMOND": 0.2},
        )
        try:
            # Add a pending game only for DIAMOND, none for CHALLENGER
            service.db.add_games_batch([
                DbGame(
                    match_uuid="diamond-game-1",
                    timeline_url="https://fake.url/diamond1",
                    focal_player_riot_id="DiaPlayer#NA1",
                    tier="DIAMOND",
                )
            ])

            mock_timeline = {
                "datetime": "2026-03-01T12:00:00Z",
                "summoner_name": "DiaPlayer",
                "tagline": "NA1",
                "stage_data": json.dumps([
                    {
                        "me": {"summoner_name": "DiaPlayer", "gold": 20, "xp": {"level": 4}},
                        "round_start_health": {
                            "player_status": {
                                "DiaPlayer": {"health": 100, "xp": 4, "gold": 20},
                                "Opponent": {"health": 100, "xp": 4, "gold": 20},
                            }
                        },
                        "match_info": {
                            "round_type": {"stage": "2-1", "name": "Augment", "type": "PVP"},
                            "opponent": {"name": "Opponent"},
                            "round_outcome": {"DiaPlayer": {"outcome": "victory"}},
                        },
                        "matchup_boards": {
                            "player_board": [{"unit": "TFT18_Renata", "tier": 1, "loc": "D1", "items": []}],
                            "opponent_board": [{"unit": "TFT18_Singed", "tier": 1, "loc": "A1", "items": []}],
                        },
                    }
                ]),
                "board_players": [
                    {"board": [{"summoner": "DiaPlayer", "tag_line": "NA1"}]},
                    {"board": [{"summoner": "Opponent", "tag_line": "NA1"}]},
                ],
            }

            with patch.object(service.client, "fetch_timeline", return_value=mock_timeline), patch.object(service, "_discover_for_tier", return_value=0):
                # Even though CHALLENGER has 80% deficit shortage, step() must fallback to download DIAMOND!
                success = service.step()
                assert success is True
                assert service.db.total_downloaded_count() == 1
                recent = service.db.get_recent_downloaded_matches(limit=1)
                assert recent[0]["match_uuid"] == "diamond-game-1"
                assert recent[0]["tier"] == "DIAMOND"
        finally:
            service.db.close()


def test_collector_embedded_api() -> None:
    import urllib.request
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "api.db"
        out_dir = Path(tmp_dir) / "data"
        status_path = Path(tmp_dir) / "status.json"

        # Use an ephemeral port by binding to 0
        service = ContinuousCollectorService(
            db_path=db_path,
            output_dir=out_dir,
            status_file=status_path,
            api_host="127.0.0.1",
            api_port=0,
        )
        service.start_api_server()
        assert service._http_server is not None
        port = service._http_server.server_port
        try:
            # 1. Health check
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health") as resp:
                assert resp.status == 200
                data = json.loads(resp.read().decode("utf-8"))
                assert data["status"] == "healthy"
                assert "frontend" in data

            # 2. Status check
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status") as resp:
                assert resp.status == 200
                data = json.loads(resp.read().decode("utf-8"))
                assert data["service"] == "tft-ai-collector"
                assert "distribution" in data
                assert "system" in data

            # 3. Root redirect/JSON check
            req = urllib.request.Request(f"http://127.0.0.1:{port}/", headers={"Accept": "application/json"})
            with urllib.request.urlopen(req) as resp:
                assert resp.status == 200
                data = json.loads(resp.read().decode("utf-8"))
                assert "frontend_portal" in data
        finally:
            service.stop_api_server()
            service.db.close()
