import json
import re
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def clean_champ(name: str) -> str:
    c = name.strip()
    c = re.sub(r"^(TFT18_|DA_18_|DA_)", "", c)
    c = re.sub(r"(18|_AP|_Small|_small)$", "", c)
    return c


def clean_item(name: str) -> str:
    it = name.strip()
    it = re.sub(r"^(TFT_Item_|DA_)", "", it)
    return it


def main():
    profile_path = Path("models/clustering/cluster_profiles.json")
    if not profile_path.exists():
        print(f"File {profile_path} does not exist.")
        return

    with open(profile_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 80)
    print(f" CLUSTER ANALYSIS & ARCHETYPE ASSIGNMENT (Total Archetypes: {data['num_archetypes']})")
    print("=" * 80)

    for arch in data["archetypes"][:3]:
        cid = arch["cluster_id"]
        size = arch["size"]
        pct = arch["percentage"]
        avg_place = arch["avg_placement"]

        top_units = [
            f"{clean_champ(u['champion'])} (★{u['avg_star_level']:.1f}, {u['frequency_pct']}%)"
            for u in arch["top_units"][:6]
        ]
        top_items = [
            f"{clean_item(it['item'])} ({it['frequency_pct']}%)"
            for it in arch["top_items"][:5]
        ]

        print(f"\n[{cid:02d}] Cluster {cid:02d} | Size: {size} ({pct}%) | Avg Place: #{avg_place:.2f}")
        print(f"     Core Champions: {', '.join(top_units)}")
        print(f"     Core Items:     {', '.join(top_items)}")
        print("     Sample Real Boards:")
        for idx, rep in enumerate(arch["representative_boards"][:3]):
            board_units = []
            for u_str in rep["units"]:
                # Parse 'DA_18_Diana (★2) [DA_Item1, DA_Item2]'
                match = re.search(r"^([^\s\(]+)\s*(\([^\)]+\))(?:\s*\[(.*)\])?", u_str)
                if match:
                    cname = clean_champ(match.group(1))
                    star = match.group(2)
                    items_raw = match.group(3)
                    if items_raw:
                        item_names = [clean_item(it) for it in items_raw.split(",") if it.strip()]
                        board_units.append(f"{cname} {star} [{', '.join(item_names)}]")
                    else:
                        board_units.append(f"{cname} {star}")
                else:
                    board_units.append(clean_champ(u_str))
            print(f"       Board #{idx+1} ({rep['round_stage']}, Place #{rep['placement']}, Dist={rep['distance_to_centroid']:.3f}):")
            print(f"         Units: {', '.join(board_units[:7])}")
            if len(board_units) > 7:
                print(f"                {', '.join(board_units[7:])}")


if __name__ == "__main__":
    main()
