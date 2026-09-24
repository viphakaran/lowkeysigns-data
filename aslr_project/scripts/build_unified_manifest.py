import csv
import json
from pathlib import Path
import cv2

METADATA_PATH = Path("data/metadata/selected_wlasl_metadata.json")
VIDEO_DIR = Path("data/raw/wlasl_mirror/videos")
OUTPUT_PATH = Path("data/metadata/unified_manifest.csv")

with METADATA_PATH.open(encoding="utf-8") as f:
    selected_meta = json.load(f)

# Build lookup by video_id
meta_by_id = {}
for entry in selected_meta:
    gloss = entry["gloss"]
    for inst in entry["instances"]:
        meta_by_id[str(inst["video_id"])] = {
            "gloss": gloss,
            "original_gloss": gloss,
            "instance_id": inst.get("instance_id", ""),
            "signer_id": f"wlasl_{inst.get('signer_id', 'unknown')}",
            "split": inst.get("split", "train"),
            "url": inst.get("url", "")
        }

rows = []
for video_file in sorted(VIDEO_DIR.glob("*.mp4")):
    vid = video_file.stem
    cap = cv2.VideoCapture(str(video_file))
    opened = cap.isOpened()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    cap.release()
    
    valid = opened and frame_count >= 5
    
    meta = meta_by_id.get(vid, {})
    label = meta.get("gloss", "unknown")
    original_label = meta.get("original_gloss", "unknown")
    signer_id = meta.get("signer_id", "wlasl_unknown")
    split_original = meta.get("split", "train")
    
    # Unified manifest row format
    row = {
        "path": str(video_file).replace("\\", "/"),
        "label": label,
        "original_label": original_label,
        "source_dataset": "WLASL",
        "source_id": vid,
        "signer_id": signer_id,
        "split_original": split_original,
        "license_note": "WLASL academic/computational use",
        "valid": valid
    }
    rows.append(row)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "path", "label", "original_label", "source_dataset", 
        "source_id", "signer_id", "split_original", "license_note", "valid"
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Unified manifest generated: {len(rows)} samples saved to {OUTPUT_PATH}")
