import json
import os
import sys
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm

SELECTED_METADATA_PATH = "data/selected_wlasl_metadata.json"
MANIFEST_PATH = "data/manifest.json"
RAW_VIDEOS_DIR = "data/raw_videos"

with open(SELECTED_METADATA_PATH, "r", encoding="utf-8") as f:
    selected_data = json.load(f)

# Load existing manifest if present to resume
if os.path.exists(MANIFEST_PATH):
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        manifest = {}
else:
    manifest = {}

# Ensure all glosses are initialized in manifest
for entry in selected_data:
    gloss = entry["gloss"]
    if gloss not in manifest:
        manifest[gloss] = {
            "attempted": [],
            "succeeded": [],
            "failed": []
        }

manifest_lock = threading.Lock()

def save_manifest():
    with manifest_lock:
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

def extract_clean_reason(stderr_text):
    if not stderr_text:
        return "Unknown error"
    lines = [line.strip() for line in stderr_text.splitlines() if line.strip()]
    # Look for ERROR: line first
    for line in reversed(lines):
        if "ERROR:" in line:
            return line.split("ERROR:", 1)[1].strip()
    return lines[-1] if lines else "Download failed"

tasks = []
for entry in selected_data:
    gloss = entry["gloss"]
    instances = sorted(entry["instances"], key=lambda x: int(x["instance_id"]))
    if gloss == "emergency":
        # All available instances
        selected_instances = instances
    else:
        # Cap of 15 per class
        selected_instances = instances[:15]
    
    gloss_dir = Path(RAW_VIDEOS_DIR) / gloss
    gloss_dir.mkdir(parents=True, exist_ok=True)
    
    for inst in selected_instances:
        tasks.append((gloss, inst))

print(f"Total video download tasks to attempt: {len(tasks)}")

def process_download(task):
    gloss, inst = task
    video_id = str(inst["video_id"])
    url = inst["url"]
    target_path = Path(RAW_VIDEOS_DIR) / gloss / f"{video_id}.mp4"
    
    with manifest_lock:
        if video_id not in manifest[gloss]["attempted"]:
            manifest[gloss]["attempted"].append(video_id)
        
        # Check if already succeeded on disk
        if target_path.exists() and target_path.stat().st_size > 0:
            if video_id not in manifest[gloss]["succeeded"]:
                manifest[gloss]["succeeded"].append(video_id)
            # Remove from failed if it was there previously
            manifest[gloss]["failed"] = [f for f in manifest[gloss]["failed"] if f.get("video_id") != video_id]
            return (gloss, video_id, True, "Already on disk")

    # Attempt download via yt-dlp
    command = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "--retries", "3",
        "--fragment-retries", "3",
        "--socket-timeout", "20",
        "-f", "mp4[height<=720]/mp4/best",
        "-o", str(target_path),
        url,
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        
        # Check if file was saved
        if target_path.exists() and target_path.stat().st_size > 0:
            with manifest_lock:
                if video_id not in manifest[gloss]["succeeded"]:
                    manifest[gloss]["succeeded"].append(video_id)
                manifest[gloss]["failed"] = [f for f in manifest[gloss]["failed"] if f.get("video_id") != video_id]
            save_manifest()
            return (gloss, video_id, True, "Downloaded")
        
        # Check if saved with a slightly different extension
        alt_files = list((Path(RAW_VIDEOS_DIR) / gloss).glob(f"{video_id}.*"))
        alt_files = [f for f in alt_files if f.stat().st_size > 0]
        if alt_files:
            alt_file = alt_files[0]
            if alt_file.suffix != ".mp4":
                alt_file.rename(target_path)
            with manifest_lock:
                if video_id not in manifest[gloss]["succeeded"]:
                    manifest[gloss]["succeeded"].append(video_id)
                manifest[gloss]["failed"] = [f for f in manifest[gloss]["failed"] if f.get("video_id") != video_id]
            save_manifest()
            return (gloss, video_id, True, "Downloaded (renamed)")
        
        # If we got here, download failed
        reason = extract_clean_reason(result.stderr)
        with manifest_lock:
            # Ensure not listed as succeeded
            if video_id in manifest[gloss]["succeeded"]:
                manifest[gloss]["succeeded"].remove(video_id)
            # Add to failed if not already recorded
            if not any(f.get("video_id") == video_id for f in manifest[gloss]["failed"]):
                manifest[gloss]["failed"].append({"video_id": video_id, "reason": reason})
        save_manifest()
        return (gloss, video_id, False, reason)
        
    except subprocess.TimeoutExpired:
        reason = "Timeout after 60s"
        with manifest_lock:
            if video_id in manifest[gloss]["succeeded"]:
                manifest[gloss]["succeeded"].remove(video_id)
            if not any(f.get("video_id") == video_id for f in manifest[gloss]["failed"]):
                manifest[gloss]["failed"].append({"video_id": video_id, "reason": reason})
        save_manifest()
        return (gloss, video_id, False, reason)
    except Exception as e:
        reason = str(e)
        with manifest_lock:
            if video_id in manifest[gloss]["succeeded"]:
                manifest[gloss]["succeeded"].remove(video_id)
            if not any(f.get("video_id") == video_id for f in manifest[gloss]["failed"]):
                manifest[gloss]["failed"].append({"video_id": video_id, "reason": reason})
        save_manifest()
        return (gloss, video_id, False, reason)

# Use ThreadPoolExecutor with 5 workers for fast, resilient parallel downloading
print("Starting downloads with 5 concurrent workers...")
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(process_download, t) for t in tasks]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Acquiring videos"):
        future.result()

save_manifest()
print("All download tasks completed and saved to data/manifest.json")
