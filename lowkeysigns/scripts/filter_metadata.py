import json
import re

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
    wlasl_data = json.load(f)

lookup = {normalize(item["gloss"]): item for item in wlasl_data}

selected_entries = []
total_instances = 0

print(f"{'Class':<15} | {'Instance Count':<15}")
print("-" * 35)

for word in ALL_20:
    norm = normalize(word)
    entry = lookup[norm]
    # Keep the word as gloss name or keep original, let's keep original or standard
    selected_entries.append(entry)
    count = len(entry["instances"])
    total_instances += count
    print(f"{word:<15} | {count:<15}")

print("-" * 35)
print(f"Total instances across all 20 classes: {total_instances}")

output_path = "data/selected_wlasl_metadata.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(selected_entries, f, indent=2)

print(f"Saved filtered metadata to {output_path}")
