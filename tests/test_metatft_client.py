from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from tft_ai_player.metatft import MetaTftClient


def test_profile_candidates_keep_matches_without_a_data_url() -> None:
    profile = {
        "matches": [
            _match("LA2_1", "TFTSet17", "https://matches3.metatft.com/LA2_1.json"),
            _match("LA2_2", "TFTSet17", None),
            _match("LA2_3", "TFTSet16", "https://matches3.metatft.com/LA2_3.json"),
        ]
    }

    candidates = MetaTftClient.match_candidates(profile, tft_set="TFTSet17")

    assert [candidate.riot_match_id for candidate in candidates] == ["LA2_1", "LA2_2"]
    assert candidates[1].match_data_url is None


def test_fetch_profile_encodes_the_riot_id_and_set_filter() -> None:
    requested_urls: list[str] = []

    def fetcher(url: str) -> dict[str, object]:
        requested_urls.append(url)
        return {"matches": []}

    client = MetaTftClient(json_fetcher=fetcher)
    profile = client.fetch_profile(
        region="la2",
        game_name="Name With Space",
        tag_line="LAS",
        tft_set="TFTSet17",
    )

    parsed_url = urlparse(requested_urls[0])
    assert profile == {"matches": []}
    assert parsed_url.path.endswith("/LA2/Name%20With%20Space/LAS")
    assert parse_qs(parsed_url.query) == {
        "source": ["full_profile"],
        "tft_set": ["TFTSet17"],
        "include_revival_matches": ["true"],
    }


def test_fetch_leaderboard_players_keeps_only_distinct_trackable_riot_ids() -> None:
    requested_urls: list[str] = []

    def fetcher(url: str) -> dict[str, object]:
        requested_urls.append(url)
        return {
            "data": [
                _leaderboard_player("First Player#LAS", "la2", app_matches=True),
                _leaderboard_player("Untracked#LAS", "la2", app_matches=False),
                _leaderboard_player("First Player#LAS", "la2", app_matches=True),
                _leaderboard_player("Second Player#NA1", "na1", app_matches=True),
            ]
        }

    players = MetaTftClient(json_fetcher=fetcher).fetch_leaderboard_players(count=2)

    assert [(player.region, player.riot_id) for player in players] == [
        ("la2", "First Player#LAS"),
        ("na1", "Second Player#NA1"),
    ]
    parsed_url = urlparse(requested_urls[0])
    assert parse_qs(parsed_url.query) == {
        "offset": ["0"],
        "limit": ["2"],
        "filter_app_user": ["true"],
        "stat_type": ["recent"],
    }


def test_tracked_timeline_candidates_filters_non_ranked_matches_by_default() -> None:
    profile = {
        "matches": [
            _match("LA2_RANKED", "TFTSet17", "https://matches3.metatft.com/LA2_RANKED.json", queue_id=1100),
            _match("LA2_DOUBLE_UP", "TFTSet17", "https://matches3.metatft.com/LA2_DOUBLE_UP.json", queue_id=1160),
            _match("LA2_NORMAL", "TFTSet17", "https://matches3.metatft.com/LA2_NORMAL.json", queue_id=1090),
        ],
        "app_matches": [
            {
                "uuid": "ranked-uuid",
                "match_id_ow": "123",
                "match_data_url": "https://matches3.metatft.com/ranked-uuid.json",
                "created_timestamp": 1_700_000_000_100,
                "player_id": 42,
            },
            {
                "uuid": "double-up-uuid",
                "match_id_ow": "456",
                "match_data_url": "https://matches3.metatft.com/double-up-uuid.json",
                "created_timestamp": 1_700_000_000_100,
                "player_id": 42,
            },
            {
                "uuid": "unmapped-uuid",
                "match_id_ow": "789",
                "match_data_url": "https://matches3.metatft.com/unmapped-uuid.json",
            },
        ],
    }

    # Ranked match matches ranked-uuid, double-up matches double-up-uuid
    profile["matches"][0]["match_timestamp"] = 1_700_000_000_100
    profile["matches"][1]["match_timestamp"] = 1_700_000_000_200
    profile["app_matches"][1]["created_timestamp"] = 1_700_000_000_200

    candidates = MetaTftClient.tracked_timeline_candidates(profile, tft_set="TFTSet17")
    assert len(candidates) == 1
    assert candidates[0].app_match_uuid == "ranked-uuid"
    assert candidates[0].riot_match_id == "LA2_RANKED"
    assert candidates[0].queue_id == 1100

    # If allowed_queue_ids is None, all app matches are retained
    all_candidates = MetaTftClient.tracked_timeline_candidates(
        profile,
        tft_set="TFTSet17",
        allowed_queue_ids=None,
    )
    assert len(all_candidates) == 3


def test_tracked_timeline_candidates_use_app_match_urls_without_guessing_metadata() -> None:
    profile = {
        "matches": [_match("LA2_123", "TFTSet17", "https://matches3.metatft.com/LA2_123.json")],
        "app_matches": [
            {
                "uuid": "timeline-uuid",
                "match_id_ow": "123",
                "match_data_url": "https://matches3.metatft.com/timeline-uuid.json",
                "created_timestamp": 1_700_000_000_100,
                "player_id": 42,
            },
            {
                "uuid": "unmapped-timeline-uuid",
                "match_id_ow": "internal-match-id",
                "match_data_url": "https://matches3.metatft.com/unmapped-timeline-uuid.json",
            },
        ],
    }

    candidates = MetaTftClient.tracked_timeline_candidates(
        profile,
        tft_set="TFTSet17",
        allowed_queue_ids=None,
    )

    assert [candidate.timeline_url for candidate in candidates] == [
        "https://matches3.metatft.com/timeline-uuid.json",
        "https://matches3.metatft.com/unmapped-timeline-uuid.json",
    ]
    assert candidates[0].match_id_ow == "123"
    assert candidates[0].app_match_uuid == "timeline-uuid"
    assert candidates[0].riot_match_id == "LA2_123"
    assert candidates[0].game_version == "16.16"
    assert candidates[0].queue_id == 1100
    assert candidates[1].riot_match_id is None
    assert candidates[1].game_version is None


def test_tracked_timeline_candidates_treat_missing_app_matches_as_untracked() -> None:
    profile = {"matches": [_match("LA2_123", "TFTSet17", None)]}

    assert MetaTftClient.tracked_timeline_candidates(profile, tft_set="TFTSet17") == []


def test_get_json_retries_and_wraps_connection_reset_error(monkeypatch) -> None:
    from unittest.mock import MagicMock
    import pytest
    from tft_ai_player.metatft import MetaTftRequestError

    attempts = 0
    slept_durations: list[float] = []

    def mock_urlopen(request, timeout):
        nonlocal attempts
        attempts += 1
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None
        mock_resp.read.side_effect = ConnectionResetError(10054, "Connection reset by peer")
        return mock_resp

    def mock_sleep(seconds: float) -> None:
        slept_durations.append(seconds)

    monkeypatch.setattr("tft_ai_player.metatft.client.urlopen", mock_urlopen)
    monkeypatch.setattr("time.sleep", mock_sleep)

    client = MetaTftClient(minimum_request_interval_seconds=0, retry_count=2)
    with pytest.raises(MetaTftRequestError, match="Connection reset by peer"):
        client._get_json("https://example.com/test.json")

    assert attempts == 3
    assert len(slept_durations) == 2


def test_get_json_retries_with_retry_after_on_429(monkeypatch) -> None:
    from io import BytesIO
    from urllib.error import HTTPError
    from unittest.mock import MagicMock

    attempts = 0
    slept_durations: list[float] = []

    def mock_urlopen(request, timeout):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            headers = {"Retry-After": "12.5"}
            raise HTTPError("https://example.com", 429, "Too Many Requests", headers, BytesIO(b""))
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None
        mock_resp.read.return_value = b'{"success": true}'
        return mock_resp

    def mock_sleep(seconds: float) -> None:
        slept_durations.append(seconds)

    monkeypatch.setattr("tft_ai_player.metatft.client.urlopen", mock_urlopen)
    monkeypatch.setattr("time.sleep", mock_sleep)

    client = MetaTftClient(minimum_request_interval_seconds=0, retry_count=2)
    result = client._get_json("https://example.com/test.json")

    assert result == {"success": True}
    assert attempts == 2
    assert slept_durations == [12.5]


def _match(
    match_id: str,
    tft_set: str,
    match_data_url: str | None,
    queue_id: int = 1100,
) -> dict[str, object]:
    return {
        "riot_match_id": match_id,
        "tft_set": tft_set,
        "patch": "16.16",
        "match_data_url": match_data_url,
        "match_timestamp": 1_700_000_000_000,
        "queue_id": queue_id,
    }


def _leaderboard_player(riot_id: str, region: str, *, app_matches: bool) -> dict[str, object]:
    return {
        "player_id": "123",
        "riot_id": riot_id,
        "summoner_region": region,
        "stats": {"appMatches": app_matches},
    }