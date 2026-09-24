import json
import re
from pathlib import Path

JSON_PATH = Path("data/metadata/WLASL_v0.3.json")

TARGET_GLOSSES = [
    "help", "wait", "money", "form", "pain",
    "doctor", "yes", "no", "thank you", "sign",
    "more", "problem", "emergency", "where",
    "name", "appointment", "sick", "please",
    "here", "now"
]

SUBSETS = {
    "WLASL100": 100,
    "WLASL300": 300,
    "WLASL1000": 1000,
    "WLASL2000": 2000
}

def normalize(text):
    text = str(text).lower().strip()
    text = text.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text)

with JSON_PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)

glosses = [item["gloss"] for item in data]
lookup = {normalize(gloss): (index, gloss) for index, gloss in enumerate(glosses)}

for target in TARGET_GLOSSES:
    result = lookup.get(normalize(target))

    if result is None:
        print(target, "NOT FOUND")
        continue

    index, original_gloss = result
    memberships = [
        subset for subset, size in SUBSETS.items()
        if index < size
    ]

    count = sum(
        len(item["instances"])
        for item in data
        if item["gloss"] == original_gloss
    )

    print({
        "requested": target,
        "official_gloss": original_gloss,
        "index": index,
        "subsets": memberships,
        "instances": count
    })
