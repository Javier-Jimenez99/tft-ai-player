from __future__ import annotations

from unittest.mock import MagicMock
from tft_ai_player.metatft.riot_client import RiotTftClient


def test_riot_tft_client_fetch_tier_players() -> None:
    client = RiotTftClient("RGAPI-TEST-KEY")

    def mock_get(url: str):
        if "entries/GOLD/I" in url:
            return [
                {"puuid": "puuid-1", "tier": "GOLD", "rank": "I"},
                {"puuid": "puuid-2", "tier": "GOLD", "rank": "I"},
            ]
        if "accounts/by-puuid/puuid-1" in url:
            return {"gameName": "GoldPlayer1", "tagLine": "NA1"}
        if "accounts/by-puuid/puuid-2" in url:
            return {"gameName": "GoldPlayer2", "tagLine": "NA1"}
        return {}

    client._get_json = MagicMock(side_effect=mock_get)

    players = client.fetch_tier_players("GOLD", region="na1", division="I", count=10)
    assert len(players) == 2
    assert players[0].riot_id == "GoldPlayer1#NA1"
    assert players[1].riot_id == "GoldPlayer2#NA1"
