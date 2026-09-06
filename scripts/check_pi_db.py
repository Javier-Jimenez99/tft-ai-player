import sqlite3

c = sqlite3.connect('/home/javi/tft-ai-player/data/collector.db')
print("PLAYERS:")
for r in c.execute("SELECT tier, scanned, COUNT(*) FROM players GROUP BY tier, scanned"):
    print(" ", r)
print("\nGAMES BY TIER & STATUS:")
for r in c.execute("SELECT tier, status, COUNT(*) FROM games GROUP BY tier, status"):
    print(" ", r)

print("\nIRON GAMES DETAILS:")
for r in c.execute("SELECT match_uuid, focal_player_riot_id, avg_rating, tier, status FROM games WHERE tier = 'IRON'"):
    print(" ", r)

print("\nLOW ELO APP USERS:")
for r in c.execute("SELECT riot_id, tier, app_matches, last_scanned_at FROM players WHERE tier IN ('IRON', 'BRONZE', 'SILVER') AND is_app_user = 1"):
    print(" ", r)

print("\nGAMES WITH LOW ELO AVG_RATING:")
for r in c.execute("SELECT tier, avg_rating, count(*) FROM games WHERE avg_rating LIKE '%Iron%' OR avg_rating LIKE '%Bronze%' GROUP BY tier, avg_rating"):
    print(" ", r)

print("\nMISMATCHED TIER GAMES (avg_rating is Iron/Bronze but tier is not):")
for r in c.execute("SELECT match_uuid, tier, avg_rating, status FROM games WHERE avg_rating LIKE '%Iron%' AND tier != 'IRON'"):
    print(" ", r)

print("\nDEFICITS:")
import sys
sys.path.insert(0, '/home/javi/tft-ai-player/src')
from tft_ai_player.dataset.collector_service import ContinuousCollectorService, DEFAULT_TARGET_WEIGHTS
service = ContinuousCollectorService(output_dir="/home/javi/tft-ai-player/data", db_path="/home/javi/tft-ai-player/data/collector.db")
deficits = service.compute_deficits()
for t, d in deficits:
    print(f"  {t:14}: {d:+.4f} (downloaded: {service.db.get_tier_counts()['downloaded'].get(t, 0)})")
