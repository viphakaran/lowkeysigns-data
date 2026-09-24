import csv
from pathlib import Path
import cv2

VIDEO_DIR = Path("data/raw/wlasl_mirror/videos")
OUTPUT = Path("data/metadata/mirror_inventory.csv")

VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".webm", ".mov", ".mkv"
}

rows = []

for path in VIDEO_DIR.rglob("*"):
    if not path.is_file():
        continue
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        continue

    cap = cv2.VideoCapture(str(path))
    opened = cap.isOpened()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
    cap.release()

    rows.append({
        "filename": path.name,
        "video_id": path.stem,
        "path": str(path),
        "opened": opened,
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "size_bytes": path.stat().st_size
    })

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
    writer.writeheader()
    writer.writerows(rows)

print("Indexed files:", len(rows))
