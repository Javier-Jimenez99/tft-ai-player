# TFT AI Player

Tools for collecting Teamfight Tactics PVP-round data as one CSV file per game.

## Scope

The first dataset targets `TFTSet17` and includes only PVP rounds. Each CSV contains all valid PVP observations from one game. Every row includes the TFT set, known Riot patch, source player, outcome label, and leakage-safe pre-combat state.

The collector writes only CSV files below `data/games/`; it does not retain raw API JSON, manifests, or Parquet partitions. `data/` is ignored by Git.

## Setup

```powershell
uv sync
```

## Collect Across Players And Games

Collect from multiple MetaTFT-tracked leaderboard players. The collector iterates player by player, immediately fetching and writing each player's tracked game CSV files to disk.

- **Streaming & Immediate Persistence**: Each game is saved into the player's CSV as soon as it is fetched, so progress is never lost if interrupted.
- **Automatic Resumption**: If restarted, the collector checks `data/players/` and skips already-downloaded games without making redundant API requests.

```powershell
uv run tft-ai-player collect --players 1000 --tft-set TFTSet17
```

- `-o`, `--output`, `--output-dir`: Destination folder where `players/*.csv` will be written (default: `data`).
- `--players`: Number of distinct leaderboard players to sample across all competitive regions (e.g. `1000`).
- `--tft-set`: Target TFT set (`TFTSet17`).
- `--request-interval`: Minimum seconds between network calls to prevent rate-limiting (default: `1.5`).
- `--games-per-player`: (Optional) Limit games per player. Defaults to unlimited (all Set 17 games available for each player).
- `--max-games`: (Optional) Total game cap across all players.
- `--allowed-queues`: (Optional) List of Riot queue IDs to retain (default: `1100` for Ranked TFT classification; excludes Double Up `1160`, Normals `1090`, etc.).
- `--leaderboard-offset`: (Optional) Start from a different offset in the leaderboard roster.

## Collect From One Player

For a targeted sample, collect a bounded number of games from one Riot ID.

```powershell
uv run tft-ai-player profile --region LA2 --game-name NickW29991 --tag-line LAS --games 2 --tft-set TFTSet17
```

## Extract a Known Timeline

When a MetaTFT timeline URL is available, download it and write one game CSV. Set `game-version` to the Riot patch when known; it defaults to `unknown`.

```powershell
uv run tft-ai-player timeline `
	--timeline-url https://matches3.metatft.com/<timeline-id>.json `
	--match-id <game-id> `
	--tft-set TFTSet17 `
	--game-version 16.16
```

The extractor accepts only PVP snapshots with a focal-player outcome and both focal and opponent boards. It excludes PVE rounds and does not write post-combat fields such as battle statistics, damage, or MetaTFT win-rate values into model features.

## Output Layout

```text
data/
	players/
		<region>_<player_riot_id>.csv
```

Each player CSV contains all valid PVP rounds for all collected matches of that player. The CSV columns are:
- `collected_from_riot_id`: Roster player through which the game was selected.
- `collected_from_region`: Region platform (e.g. `la2`, `kr`).
- `match_id_ow`: MetaTFT Overwolf internal match ID.
- `observation_id`: Stable identifier (`<match_id>:<round_stage>:<focal_player>`).
- `match_id`: Game UUID.
- `game_datetime`: ISO 8601 UTC timestamp when the match completed (e.g. `2026-08-23T12:07:19Z`).
- `round_stage`: Round stage (e.g. `2-2`, `3-5`).
- `round_type`: Round type (`PVP`).
- `tft_set`: TFT set identifier (`TFTSet17`).
- `game_version`: Patch/version string or `unknown`.
- `game_client_version`: Local client version if recorded.
- `timeline_schema_version`: MetaTFT data schema version.
- `portal`: Match opening portal / encounter rule (e.g. `TFT_Portals_Champions_ChampionStart`).
- `focal_player`: Summoner name of the tracked player.
- `focal_tier`: Rank tier of focal player (e.g. `CHALLENGER` or `CHALLENGER I 1756 LP`).
- `focal_rating_numeric`: Exact continuous numerical ELO rating (e.g. `4556`).
- `avg_match_rating`: Average rank tier of the 8-player match lobby (e.g. `GRANDMASTER I 937 LP`).
- `avg_match_rating_numeric`: Average numerical ELO rating of the match lobby (e.g. `3737`).
- `focal_health`: Focal player HP at round start (1–100).
- `focal_level`: Focal player level (1–11).
- `focal_gold`: Focal player available gold at round start.
- `focal_augments`: Focal player active augments (comma-separated).
- `focal_unit_count`: Number of champions fielded on focal player board.
- `focal_item_count`: Total items equipped across focal player board.
- `opponent`: Summoner name of the round opponent.
- `opponent_health`: Opponent HP at round start (1–100).
- `opponent_level`: Opponent level (1–11).
- `opponent_augments`: Opponent active augments (comma-separated).
- `opponent_unit_count`: Number of champions fielded on opponent board.
- `opponent_item_count`: Total items equipped across opponent board.
- `outcome`: `victory` or `defeat`.
- `label`: `1` for victory, `0` for defeat.
- `metatft_win_prob`: MetaTFT's benchmark prediction probability (0.0 to 1.0) or empty if unpredicted.
- `input_state_json`: Spatial board unit features.

### `input_state_json` Structure

```json
{
  "focal_board": [
    {
      "unit": "TFT17_Aatrox",
      "tier": 2,
      "loc": "D1",
      "items": ["TFT_Item_WarmogsArmor"]
    }
  ],
  "opponent_board": [
    {
      "unit": "TFT17_Veigar",
      "tier": 2,
      "loc": "A_1",
      "items": []
    }
  ]
}
```

Use the public endpoints responsibly: follow MetaTFT's terms, limit request volume, and retain source provenance.

