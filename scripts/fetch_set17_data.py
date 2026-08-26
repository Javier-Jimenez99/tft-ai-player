"""Script to probe and download official TFT Set 17 champion/trait data from CommunityDragon / DataDragon."""

import json
import urllib.request
from pathlib import Path

def fetch_data():
    urls = [
        "https://raw.communitydragon.org/latest/cdragon/tft/en_us.json",
        "https://raw.communitydragon.org/pbe/cdragon/tft/en_us.json",
        "https://ddragon.leagueoflegends.com/cdn/14.23.1/data/en_US/tft-champion.json",
    ]
    
    for url in urls:
        print(f"Trying to fetch: {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as response:
                content = response.read().decode("utf-8")
                print(f"Successfully fetched {len(content)} bytes from {url}")
                data = json.loads(content)
                
                # Check for Set 17 data
                if "setData" in data:
                    for s in data["setData"]:
                        mutator = s.get("mutator", "")
                        print(f"  Found set: {mutator} (name: {s.get('name', '')})")
                        if "Set17" in mutator or "TFTSet17" in mutator or s.get("number") == 17:
                            print(f"  --> MATCHED SET 17! Champions: {len(s.get('champions', []))}, Traits: {len(s.get('traits', []))}")
                            Path("data/cdragon_set17.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
                            return s
                elif "sets" in data:
                    for set_num, set_info in data["sets"].items():
                        print(f"  Found set key: {set_num}")
                        if str(set_num) in ("17", "TFTSet17"):
                            Path("data/cdragon_set17.json").write_text(json.dumps(set_info, indent=2), encoding="utf-8")
                            return set_info
        except Exception as e:
            print(f"  Failed: {e}")
            
    print("Could not directly find Set 17 in CommunityDragon latest.")
    return None

if __name__ == "__main__":
    fetch_data()
