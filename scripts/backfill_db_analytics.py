"""Backfill elo_rating, placement, and is_top1 in collector.db from database and CSV files."""

import csv
import re
import sqlite3
import sys
import time
from pathlib import Path

# Add src to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from tft_ai_player.dataset.collector_db import CollectorDb, parse_elo_to_numeric


def backfill(data_dir: Path, db_path: Path):
    print(f"Opening database: {db_path}")
    db = CollectorDb(db_path)
    conn = db.conn

    # 1. Update elo_rating from games table where available
    print("Step 1: Recalculating elo_rating from 'games' table and tier fallback...")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT d.match_uuid, d.tier, g.avg_rating
        FROM downloaded_matches d
        LEFT JOIN games g ON d.match_uuid = g.match_uuid
    """)
    rows = cursor.fetchall()
    print(f"  Processing {len(rows)} matches to recompute accurate ELO...")
    
    updates = []
    for r in rows:
        elo = parse_elo_to_numeric(r["avg_rating"], tier_fallback=r["tier"])
        if elo:
            updates.append((elo, r["match_uuid"]))
            
    if updates:
        cursor.execute("BEGIN IMMEDIATE;")
        cursor.executemany("UPDATE downloaded_matches SET elo_rating = ? WHERE match_uuid = ?;", updates)
        cursor.execute("COMMIT;")
        print(f"  Updated {len(updates)} ELO ratings.")

    # 2. For remaining downloaded_matches without elo_rating, assign fallback by tier
    cursor.execute("""
        SELECT match_uuid, tier FROM downloaded_matches WHERE elo_rating IS NULL
    """)
    missing_elo = cursor.fetchall()
    if missing_elo:
        print(f"  Found {len(missing_elo)} matches without ELO. Assigning tier fallback...")
        tier_updates = []
        for r in missing_elo:
            elo = parse_elo_to_numeric(None, tier_fallback=r["tier"])
            if elo:
                tier_updates.append((elo, r["match_uuid"]))
        if tier_updates:
            cursor.execute("BEGIN IMMEDIATE;")
            cursor.executemany("UPDATE downloaded_matches SET elo_rating = ? WHERE match_uuid = ?;", tier_updates)
            cursor.execute("COMMIT;")
            print(f"  Updated {len(tier_updates)} ELO ratings with tier defaults.")

    # 3. Backfill placement and is_top1 from CSV files
    print("\nStep 2: Scanning CSV files for placement and Top 1 outcome...")
    csv_paths = list(data_dir.rglob("*.csv"))
    # filter out non-match csv files (like players.csv, games.csv)
    csv_paths = [p for p in csv_paths if "graph" not in str(p) and p.name not in ("players.csv", "games.csv")]
    print(f"  Found {len(csv_paths)} player CSV files.")

    # Map match_uuid -> final round row
    match_final_rounds: dict[str, dict] = {}

    for csv_file in csv_paths:
        try:
            with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    mid = row.get("match_id")
                    if not mid:
                        continue
                    # In CSV, rounds appear in chronological order, so the last round seen for a match_id is the terminal round
                    match_final_rounds[mid] = row
        except Exception:
            continue

    print(f"  Parsed {len(match_final_rounds)} matches from disk CSVs.")

    # Now calculate placement and is_top1
    match_analytics = []
    for mid, row in match_final_rounds.items():
        outcome = (row.get("outcome") or "").strip().lower()
        health_str = row.get("focal_health")
        try:
            health = int(float(health_str)) if health_str not in (None, "") else None
        except ValueError:
            health = None
            
        is_top1 = 1 if (outcome == "victory" and (health is None or health > 0)) else 0
        
        if is_top1:
            placement = 1
        else:
            stage_str = row.get("round_stage") or row.get("round_str") or ""
            st = (0, 0)
            m = re.match(r"^(\d+)-(\d+)$", stage_str.strip())
            if m:
                st = (int(m.group(1)), int(m.group(2)))
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
                
        rating_str = row.get("avg_match_rating") or row.get("focal_tier")
        elo = parse_elo_to_numeric(rating_str)

        match_analytics.append((is_top1, placement, elo, mid))

    print(f"  Applying updates to database for {len(match_analytics)} matches...")
    cursor.execute("BEGIN IMMEDIATE;")
    cursor.executemany("""
        UPDATE downloaded_matches
        SET is_top1 = ?,
            placement = ?,
            elo_rating = COALESCE(?, elo_rating)
        WHERE match_uuid = ?;
    """, match_analytics)
    cursor.execute("COMMIT;")

    # 4. Check for any remaining downloaded_matches without placement
    cursor.execute("SELECT count(*) FROM downloaded_matches WHERE placement IS NULL;")
    remaining_null = cursor.fetchone()[0]
    print(f"  Matches remaining with NULL placement: {remaining_null}")

    if remaining_null > 0:
        print("  Assigning default estimated placement (placement 4) to remaining matches...")
        cursor.execute("BEGIN IMMEDIATE;")
        cursor.execute("UPDATE downloaded_matches SET placement = 4, is_top1 = 0 WHERE placement IS NULL;")
        cursor.execute("COMMIT;")

    # 5. Output analytics summary
    print("\nStep 3: Verifying analytics summary...")
    summary = db.get_analytics_summary()
    print("Analytics Summary:")
    print(f"  Total matches: {summary['total_matches']}")
    print(f"  Top 1 count:   {summary['top1_count']} ({summary['top1_pct']}%)")
    print(f"  Top 4 count:   {summary['top4_count']} ({summary['top4_pct']}%)")
    print(f"  Avg ELO:       {summary['avg_elo']} (Min: {summary['min_elo']}, Max: {summary['max_elo']})")
    print("  Placement Histogram:")
    for lbl, cnt in zip(summary["placement_histogram"]["labels"], summary["placement_histogram"]["counts"]):
        print(f"    {lbl:12}: {cnt}")
    print("  ELO Histogram Sample (non-zero bins):")
    for lbl, cnt in zip(summary["elo_histogram"]["labels"], summary["elo_histogram"]["counts"]):
        if cnt > 0:
            print(f"    {lbl}: {cnt}")

    cursor.close()
    print("\nBackfill completed successfully!")


if __name__ == "__main__":
    data_directory = Path("/home/javi/tft-ai-player/data")
    if not data_directory.exists():
        data_directory = BASE_DIR / "data"
    
    db_file = data_directory / "collector.db"
    backfill(data_directory, db_file)
