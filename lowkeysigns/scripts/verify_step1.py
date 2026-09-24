import json
import sys

with open("WLASL_repo/start_kit/WLASL_v0.3.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total glosses: {len(data)}")
print(f"First gloss: {data[0]['gloss']}")
print(f"First gloss instance count: {len(data[0]['instances'])}")

actual_keys = sorted(list(data[0]['instances'][0].keys()))
print(f"Instance keys: {actual_keys}")

expected_keys = sorted([
    "bbox", "fps", "frame_end", "frame_start",
    "instance_id", "signer_id", "source", "split",
    "url", "variation_id", "video_id"
])

print(f"Expected keys: {expected_keys}")
if actual_keys == expected_keys:
    print("KEY_CHECK_PASSED")
else:
    print("KEY_CHECK_FAILED")
    sys.exit(1)
