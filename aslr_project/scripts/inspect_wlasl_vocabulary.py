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

def normalize(text):
    text = str(text).lower().strip()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text

with JSON_PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)

glosses = [item["gloss"] for item in data]
normalized_to_original = {
    normalize(gloss): gloss for gloss in glosses
}

for target in TARGET_GLOSSES:
    normalized_target = normalize(target)
    exact = normalized_to_original.get(normalized_target)

    related = [
        gloss for gloss in glosses
        if normalized_target in normalize(gloss)
        or normalize(gloss) in normalized_target
    ]

    print("=" * 70)
    print("Requested:", target)
    print("Exact:", exact)
    print("Related:", related[:20])
