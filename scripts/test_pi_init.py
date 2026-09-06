"""Verification script to test Pi environment and database state."""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tft_ai_player.cli import _load_env_file
from tft_ai_player.dataset.collector_db import CollectorDb
from tft_ai_player.dataset.collector_service import ContinuousCollectorService

_load_env_file()
key = os.environ.get("RIOT_API_KEY")
print(f"Riot API Key loaded from .env: {bool(key)} (length: {len(key) if key else 0})")

db_path = Path("data/collector.db")
db = CollectorDb(db_path)
disk_cnt = db.import_existing_disk_matches("data")
print(f"Matches indexed from data/ on Pi: {disk_cnt}")
p_cnt, g_cnt = db.import_existing_graph("data/graph")
print(f"Graph imported on Pi: {p_cnt} players, {g_cnt} games")

counts = db.get_tier_counts()
print(f"Downloaded matches per tier: {counts['downloaded']}")
print(f"Pending candidate matches per tier: {counts['pending']}")
db.close()

# Test service deficit computation
service = ContinuousCollectorService(
    db_path="data/collector.db",
    output_dir="data",
    status_file="data/status.json",
    proxy="socks5://127.0.0.1:9050",
    tor_control_port=9051,
)
deficits = service.compute_deficits()
print("\nTop 5 deficits (scheduling priority):")
for t, d in deficits[:5]:
    dl = counts['downloaded'].get(t, 0)
    print(f"  {t:14}: deficit={d:+.4f} (downloaded={dl})")

service.write_status_heartbeat()
print("Heartbeat written to data/status.json successfully!")
service.db.close()
