import json
import re
from pathlib import Path

INPUT_PATH = Path("data/metadata/WLASL_v0.3.json")
OUTPUT_PATH = Path("data/metadata/selected_wlasl_metadata.json")

TARGET_GLOSSES = {
    "help", "wait", "money", "form", "pain",
    "doctor", "yes", "no", "thank you", "sign",
    "more", "problem", "emergency", "where",
    "name", "appointment", "sick", "please",
    "here", "now"
}

def normalize(text):
    text = str(text).lower().strip()
    text = text.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text)

with INPUT_PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)

selected = [
    item for item in data
    if normalize(item["gloss"]) in TARGET_GLOSSES
]

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(selected, f, indent=2, ensure_ascii=False)

print("Selected classes:")
for item in selected:
    print(item["gloss"], len(item["instances"]))

print(
    "Total instances:",
    sum(len(item["instances"]) for item in selected)
)
