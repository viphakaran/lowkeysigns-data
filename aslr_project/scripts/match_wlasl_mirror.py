import csv
import json
from pathlib import Path

METADATA_PATH = Path("data/metadata/selected_wlasl_metadata.json")
INVENTORY_PATH = Path("data/metadata/mirror_inventory.csv")
OUTPUT_PATH = Path("data/metadata/wlasl_mirror_match.csv")

with METADATA_PATH.open(encoding="utf-8") as f:
    metadata = json.load(f)

with INVENTORY_PATH.open(encoding="utf-8") as f:
    inventory = list(csv.DictReader(f))

inventory_by_id = {
    str(row["video_id"]): row
    for row in inventory
}

rows = []

for item in metadata:
    for instance in item["instances"]:
        video_id = str(instance["video_id"])
        local = inventory_by_id.get(video_id)

        row = {
            "gloss": item["gloss"],
            "video_id": video_id,
            "url": instance.get("url", ""),
            "split": instance.get("split", ""),
            "signer_id": instance.get("signer_id", ""),
            "local_path": local["path"] if local else "",
            "available": bool(local),
            "opened": local["opened"] if local else False
        }

        rows.append(row)

with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
    fieldnames = rows[0].keys() if rows else []
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

available_count = sum(1 for r in rows if r["available"])
print(f"Total instances checked: {len(rows)}, Available in mirror: {available_count}, Missing: {len(rows) - available_count}")
