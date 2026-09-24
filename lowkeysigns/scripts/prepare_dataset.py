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

# 7.4. Build and save a label mapping: {0: "help", 1: "wait", ...}
# index assignment order = the order words appear in ALL_20
label_mapping = {idx: word for idx, word in enumerate(ALL_20)}
word_to_idx = {word: idx for idx, word in enumerate(ALL_20)}

LABEL_MAPPING_PATH = "data/label_mapping.json"
with open(LABEL_MAPPING_PATH, "w", encoding="utf-8") as f:
    json.dump(label_mapping, f, indent=2)

print(f"Saved label mapping to {LABEL_MAPPING_PATH}")

LANDMARKS_DIR = Path("data/landmarks")
FIXED_WINDOW = 60
FEATURE_DIM = 254

X_list = []
y_list = []
samples_per_class = {word: 0 for word in ALL_20}

# Iterate through classes in ALL_20 order
for word in ALL_20:
    class_idx = word_to_idx[word]
    word_dir = LANDMARKS_DIR / word
    if not word_dir.exists():
        continue
        
    for npy_file in sorted(word_dir.glob("*.npy")):
        try:
            seq = np.load(str(npy_file))
            if seq.ndim != 2 or seq.shape[1] != FEATURE_DIM:
                print(f"Warning: Unexpected shape {seq.shape} in {npy_file}")
                continue
                
            T = seq.shape[0]
            sample = np.zeros((FIXED_WINDOW, FEATURE_DIM), dtype=np.float32)
            
            # 7.1: pad with zero-vectors if shorter, truncate from end if longer
            if T < FIXED_WINDOW:
                sample[:T, :] = seq
            else:
                sample[:, :] = seq[:FIXED_WINDOW, :]
                
            X_list.append(sample)
            y_list.append(class_idx)
            samples_per_class[word] += 1
            
        except Exception as e:
            print(f"Error loading {npy_file}: {e}")

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int64)

DATASET_PATH = "data/train_ready_dataset.npz"
np.savez(DATASET_PATH, X=X, y=y)

npz_size_bytes = os.path.getsize(DATASET_PATH)

def format_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

print("\n" + "=" * 50)
print("TRAIN-READY DATASET PREPARED SUCCESSFULLY")
print("=" * 50)
print(f"Final shape of X: {X.shape} (dtype: {X.dtype})")
print(f"Final shape of y: {y.shape} (dtype: {y.dtype})")
print(f"Total samples: {len(y)}")
print(f"File size of {DATASET_PATH}: {format_size(npz_size_bytes)} ({npz_size_bytes} bytes)")
print(f"Label mapping confirmed: {len(label_mapping)} classes saved at {LABEL_MAPPING_PATH}")
print("\nSamples per class:")
for word in ALL_20:
    print(f"  {word_to_idx[word]:2d}: {word:<15} -> {samples_per_class[word]} samples")
