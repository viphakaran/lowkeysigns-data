import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

INPUT_PATH = Path("data/metadata/wlasl_mirror_match.csv")
OUTPUT_PATH = Path("data/metadata/url_status.json")

with INPUT_PATH.open(encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

def check_single_url(row):
    if row["available"] == "True":
        status = "available_in_mirror"
    else:
        url = row["url"]
        if not url:
            status = "no_url"
        else:
            command = [
                sys.executable,
                "-m", "yt_dlp",
                "--simulate",
                "--no-playlist",
                "--ignore-config",
                "--socket-timeout", "5",
                url
            ]

            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                error_text = (result.stdout + result.stderr).lower()

                if result.returncode == 0:
                    status = "reachable"
                elif any(term in error_text for term in [
                    "video unavailable",
                    "private video",
                    "has been removed",
                    "video not found",
                    "no longer available",
                    "404"
                ]):
                    status = "likely_dead"
                elif any(term in error_text for term in [
                    "sign in",
                    "age-restricted",
                    "members-only",
                    "confirm your age",
                    "403"
                ]):
                    status = "restricted"
                elif any(term in error_text for term in [
                    "429",
                    "too many requests",
                    "bot",
                    "rate limit"
                ]):
                    status = "rate_limited"
                else:
                    status = "failed_or_unknown"
            except subprocess.TimeoutExpired:
                status = "timed_out"
            except Exception:
                status = "error"

    result_row = dict(row)
    result_row["status"] = status
    result_row["checked_at"] = datetime.now(timezone.utc).isoformat()
    return result_row

print(f"Checking URL status for {len(rows)} instances with 10 workers...")
results = []
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(check_single_url, r) for r in rows]
    for fut in tqdm(as_completed(futures), total=len(futures), desc="Checking URLs"):
        results.append(fut.result())

# Sort to maintain original order
results = sorted(results, key=lambda x: (x["gloss"], str(x["video_id"])))

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"URL status checked for {len(results)} instances. Saved to {OUTPUT_PATH}")
