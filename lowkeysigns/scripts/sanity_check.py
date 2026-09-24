import os
import json
from pathlib import Path
import numpy as np

CORE_15 = [
    "help", "wait", "money", "form", "pain", "doctor", 
    "yes", "no", "thank you", "sign", "more", "problem", 
    "emergency", "where", "name"
]
STRETCH_5 = ["appointment", "sick", "please", "here", "now"]
ALL_20 = CORE_15 + STRETCH_5

LANDMARKS_DIR = Path("data/landmarks")
REPORT_PATH = "scripts/sanity_check_report.json"

class_stats = []
total_npy_count = 0
total_bytes = 0
at_risk_classes = []
emergency_stat = None

print(f"{'Class':<15} | {'NPY Files':<10} | {'Avg Frames':<12} | {'Min Frames':<10} | {'Max Frames':<10} | {'Status':<12}")
print("-" * 80)

for word in ALL_20:
    word_dir = LANDMARKS_DIR / word
    if not word_dir.exists():
        npy_files = []
    else:
        npy_files = list(word_dir.glob("*.npy"))
        
    num_files = len(npy_files)
    total_npy_count += num_files
    
    frame_counts = []
    for f in npy_files:
        total_bytes += f.stat().st_size
        try:
            arr = np.load(str(f))
            frame_counts.append(arr.shape[0])
        except Exception:
            pass
            
    if frame_counts:
        avg_frames = float(np.mean(frame_counts))
        min_frames = int(np.min(frame_counts))
        max_frames = int(np.max(frame_counts))
    else:
        avg_frames = 0.0
        min_frames = 0
        max_frames = 0
        
    status = "OK"
    if num_files < 5:
        status = "AT-RISK (<5)"
        at_risk_classes.append({"class": word, "count": num_files})
        
    row = {
        "class": word,
        "npy_count": num_files,
        "avg_frames": round(avg_frames, 2),
        "min_frames": min_frames,
        "max_frames": max_frames,
        "status": status
    }
    class_stats.append(row)
    
    if word == "emergency":
        emergency_stat = row
        
    print(f"{word:<15} | {num_files:<10} | {avg_frames:<12.1f} | {min_frames:<10} | {max_frames:<10} | {status:<12}")

print("-" * 80)

# Total disk size formatted
def format_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

disk_size_str = format_size(total_bytes)

print(f"\nTotal .npy files saved: {total_npy_count}")
print(f"Total disk size of data/landmarks/: {disk_size_str} ({total_bytes} bytes)")

if emergency_stat:
    print(f"\n[EXPLICIT NOTE: 'emergency']")
    print(f"  Count: {emergency_stat['npy_count']}")
    print(f"  Avg Frames: {emergency_stat['avg_frames']}")
    print(f"  Min Frames: {emergency_stat['min_frames']}, Max Frames: {emergency_stat['max_frames']}")
    print(f"  Status: {emergency_stat['status']}")

if at_risk_classes:
    print(f"\n[AT-RISK CLASSES (< 5 samples)]: {[c['class'] for c in at_risk_classes]}")
else:
    print("\n[ALL CLASSES HEALTHY]: All 20 classes have at least 5 valid .npy files!")

full_report = {
    "total_classes": len(ALL_20),
    "total_npy_files": total_npy_count,
    "total_disk_size_bytes": total_bytes,
    "total_disk_size_human": disk_size_str,
    "emergency_details": emergency_stat,
    "at_risk_classes": at_risk_classes,
    "class_statistics": class_stats
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(full_report, f, indent=2)

print(f"Report saved to {REPORT_PATH}")
