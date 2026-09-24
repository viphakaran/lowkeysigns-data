import json
import re
import sys

CORE_15 = [
    "help", "wait", "money", "form", "pain", "doctor",
    "yes", "no", "thank you", "sign", "more", "problem",
    "emergency", "where", "name"
]

STRETCH_5 = ["appointment", "sick", "please", "here", "now"]

ALL_20 = CORE_15 + STRETCH_5

def normalize(text):
    text = text.lower().strip()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text

with open("WLASL_repo/start_kit/WLASL_v0.3.json", "r", encoding="utf-8") as f:
    data = json.load(f)

lookup = {normalize(item["gloss"]): (idx, item) for idx, item in enumerate(data)}

report_items = []
all_found = True
missing_words = []

print(f"{'Word':<15} | {'Found':<6} | {'Gloss Index':<12} | {'Instance Count':<14}")
print("-" * 55)

for word in ALL_20:
    norm_word = normalize(word)
    if norm_word in lookup:
        idx, item = lookup[norm_word]
        inst_count = len(item["instances"])
        row = {
            "word": word,
            "found": "yes",
            "gloss_index": idx,
            "original_gloss": item["gloss"],
            "instance_count": inst_count
        }
        report_items.append(row)
        print(f"{word:<15} | {'yes':<6} | {idx:<12} | {inst_count:<14}")
    else:
        all_found = False
        missing_words.append(word)
        row = {
            "word": word,
            "found": "no",
            "gloss_index": None,
            "original_gloss": None,
            "instance_count": 0
        }
        report_items.append(row)
        print(f"{word:<15} | {'no':<6} | {'N/A':<12} | {0:<14}")

report_data = {
    "all_found": all_found,
    "missing_words": missing_words,
    "vocabulary_report": report_items,
    "total_words": len(ALL_20),
    "emergency_instance_count": next((item["instance_count"] for item in report_items if item["word"] == "emergency"), None)
}

with open("scripts/vocabulary_verification_report.json", "w", encoding="utf-8") as f:
    json.dump(report_data, f, indent=2)

print("\nSaved report to scripts/vocabulary_verification_report.json")

if not all_found:
    print(f"\nCRITICAL: Missing words found: {missing_words}")
    sys.exit(1)
else:
    print("\nAll 20 words verified successfully in WLASL_v0.3.json!")
