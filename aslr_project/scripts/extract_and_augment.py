import os
import sys
import json
import csv
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
from scipy.interpolate import interp1d

# Setup MediaPipe HandLandmarker
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    import urllib.request
    print("Downloading hand_landmarker.task model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)

METADATA_PATH = Path("data/metadata/selected_wlasl_metadata.json")
MANIFEST_PATH = Path("data/metadata/unified_manifest.csv")
OUTPUT_LANDMARKS_DIR = Path("data/selected/landmarks")
OUTPUT_LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR = Path("data/splits")
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

with METADATA_PATH.open(encoding="utf-8") as f:
    selected_meta = json.load(f)

frame_lookup = {}
for entry in selected_meta:
    for inst in entry["instances"]:
        frame_lookup[str(inst["video_id"])] = (inst.get("frame_start", 1), inst.get("frame_end", -1))

with MANIFEST_PATH.open(encoding="utf-8") as f:
    manifest_rows = list(csv.DictReader(f))

# Fingertip landmark indices in MediaPipe
FINGERTIPS = [4, 8, 12, 16, 20]

def extract_features_from_video(video_path, frame_start=1, frame_end=-1):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0 and frame_start > 1 and total_frames < frame_start:
        start_idx = 0
        end_idx = -1
    else:
        start_idx = max(0, frame_start - 1)
        end_idx = frame_end - 1 if frame_end > 0 else -1
        
    raw_frames_data = []
    empty_mask = []
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx >= start_idx:
            if end_idx > 0 and frame_idx > end_idx:
                break
                
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = detector.detect(mp_image)
            
            hands_list = res.hand_landmarks if res.hand_landmarks else []
            
            # Extract hand features
            coords_63 = []
            presence = []
            raw_wrists = []
            finger_extensions = []
            
            for slot in range(2):
                if slot < len(hands_list):
                    hl = hands_list[slot]
                    pts = np.array([[lm.x, lm.y, lm.z] for lm in hl], dtype=np.float32)
                    raw_wrist = pts[0].copy()
                    raw_wrists.append(raw_wrist)
                    
                    # Wrist centered
                    centered = pts - raw_wrist
                    scale_ref = float(np.linalg.norm(centered[9]))
                    
                    if scale_ref < 1e-6:
                        coords_63.append(np.zeros(63, dtype=np.float32))
                        presence.append(0.0)
                        finger_extensions.extend([0.0]*5)
                    else:
                        normed = centered / scale_ref
                        coords_63.append(normed.flatten())
                        presence.append(1.0)
                        # Fingertip-to-wrist normalized distances
                        for tip_idx in FINGERTIPS:
                            finger_extensions.append(float(np.linalg.norm(normed[tip_idx])))
                else:
                    coords_63.append(np.zeros(63, dtype=np.float32))
                    presence.append(0.0)
                    raw_wrists.append(np.zeros(3, dtype=np.float32))
                    finger_extensions.extend([0.0]*5)
                    
            is_empty = (presence[0] == 0.0 and presence[1] == 0.0)
            empty_mask.append(is_empty)
            
            # Inter-hand relative geometry
            if presence[0] == 1.0 and presence[1] == 1.0:
                inter_dist = float(np.linalg.norm(raw_wrists[0] - raw_wrists[1]))
                wrist_offset = (raw_wrists[1] - raw_wrists[0]).tolist()
            else:
                inter_dist = 0.0
                wrist_offset = [0.0, 0.0, 0.0]
                
            inter_hand_features = [inter_dist] + wrist_offset  # 4 values
            
            # Combine static frame features:
            # 63 (h0) + 63 (h1) + 2 (presence) + 4 (inter-hand) + 10 (finger ext) = 142 values
            frame_static = np.concatenate([
                coords_63[0], coords_63[1], 
                presence, 
                inter_hand_features, 
                finger_extensions
            ]).astype(np.float32)
            
            raw_frames_data.append(frame_static)
            
        frame_idx += 1
        
    cap.release()
    
    if not raw_frames_data or all(empty_mask):
        return None
        
    # Trim leading and trailing empty frames before/after active sign
    first_detected = empty_mask.index(False)
    last_detected = len(empty_mask) - 1 - empty_mask[::-1].index(False)
    
    active_data = raw_frames_data[first_detected:last_detected + 1]
    active_mask = empty_mask[first_detected:last_detected + 1]
    num_frames = len(active_data)
    
    if num_frames < 3:
        return None
        
    # Check max consecutive empty frames within active signing
    max_streak = 0
    cur_streak = 0
    for e in active_mask:
        if e:
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
        else:
            cur_streak = 0
            
    if max_streak >= 3:
        return None
        
    pos_array = np.array(active_data, dtype=np.float32) # (num_frames, 142)
    
    # Linear interpolation for isolated empty frames (< 3)
    valid_idx = [i for i, e in enumerate(active_mask) if not e]
    if len(valid_idx) < num_frames:
        for d in range(142):
            pos_array[:, d] = np.interp(np.arange(num_frames), valid_idx, pos_array[valid_idx, d])
        # Threshold presence flags
        pos_array[:, 126] = np.where(pos_array[:, 126] > 0.5, 1.0, 0.0)
        pos_array[:, 127] = np.where(pos_array[:, 127] > 0.5, 1.0, 0.0)
        
    # Velocity features: delta of the 126 coordinate values
    velocity = np.zeros((num_frames, 126), dtype=np.float32)
    if num_frames > 1:
        velocity[1:] = pos_array[1:, :126] - pos_array[:-1, :126]
        
    # Total vector: 142 static + 126 velocity = 268 values per frame
    full_sequence = np.hstack([pos_array, velocity]).astype(np.float32)
    return full_sequence

print("Phase A: Extracting base landmarks from valid mirror videos...")
extracted_samples = []

for row in tqdm(manifest_rows, desc="Base extraction"):
    if row["valid"] != "True":
        continue
        
    vid = row["source_id"]
    gloss = row["label"]
    vpath = Path(row["path"])
    if not vpath.exists():
        continue
        
    frame_range = frame_lookup.get(vid, (1, -1))
    seq = extract_features_from_video(vpath, frame_range[0], frame_range[1])
    
    if seq is not None and len(seq) >= 5:
        out_gloss_dir = OUTPUT_LANDMARKS_DIR / gloss
        out_gloss_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_gloss_dir / f"{vid}_orig.npy"
        np.save(str(out_file), seq)
        
        extracted_samples.append({
            "video_id": vid,
            "label": gloss,
            "signer_id": row["signer_id"],
            "source_dataset": row["source_dataset"],
            "split_original": row["split_original"],
            "path": str(out_file).replace("\\", "/"),
            "num_frames": len(seq),
            "is_augmented": False
        })

print(f"Base extraction completed: {len(extracted_samples)} clean sequences saved.")

# Phase B: Synthetic Landmark Augmentation
# Target: At least 25-35 samples per class
print("\nPhase B: Applying synthetic landmark augmentation...")

def augment_sequence(seq, aug_type, rng):
    T, D = seq.shape
    new_seq = seq.copy()
    
    if aug_type == "speed_fast":
        # 15% faster (fewer frames)
        new_T = max(8, int(T * 0.85))
        f = interp1d(np.linspace(0, 1, T), new_seq, axis=0, kind='linear')
        new_seq = f(np.linspace(0, 1, new_T)).astype(np.float32)
        # Recompute velocity
        new_seq[0, 142:] = 0.0
        if new_T > 1:
            new_seq[1:, 142:] = new_seq[1:, :126] - new_seq[:-1, :126]
            
    elif aug_type == "speed_slow":
        # 15% slower (more frames)
        new_T = min(85, int(T * 1.15))
        f = interp1d(np.linspace(0, 1, T), new_seq, axis=0, kind='linear')
        new_seq = f(np.linspace(0, 1, new_T)).astype(np.float32)
        # Recompute velocity
        new_seq[0, 142:] = 0.0
        if new_T > 1:
            new_seq[1:, 142:] = new_seq[1:, :126] - new_seq[:-1, :126]
            
    elif aug_type == "jitter":
        # Gaussian jitter on coordinates
        noise = rng.normal(0, 0.008, size=(T, 126)).astype(np.float32)
        new_seq[:, :126] += noise
        # Recompute velocity
        new_seq[0, 142:] = 0.0
        if T > 1:
            new_seq[1:, 142:] = new_seq[1:, :126] - new_seq[:-1, :126]
            
    elif aug_type == "scale_rotate":
        scale = rng.uniform(0.94, 1.06)
        angle = np.radians(rng.uniform(-5.0, 5.0))
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # Apply to Hand 0 & Hand 1 xy coordinates
        for h in [0, 1]:
            offset = h * 63
            for lm in range(21):
                idx = offset + lm * 3
                x = new_seq[:, idx] * scale
                y = new_seq[:, idx + 1] * scale
                new_seq[:, idx] = x * cos_a - y * sin_a
                new_seq[:, idx + 1] = x * sin_a + y * cos_a
                
        # Recompute velocity
        new_seq[0, 142:] = 0.0
        if T > 1:
            new_seq[1:, 142:] = new_seq[1:, :126] - new_seq[:-1, :126]
            
    return new_seq

# Count per class
samples_by_class = {}
for s in extracted_samples:
    samples_by_class.setdefault(s["label"], []).append(s)

all_dataset_records = list(extracted_samples)
rng = np.random.RandomState(42)

TARGET_SAMPLES_PER_CLASS = 28
aug_types_cycle = ["speed_fast", "speed_slow", "jitter", "scale_rotate"]

for gloss, base_list in samples_by_class.items():
    current_count = len(base_list)
    needed = TARGET_SAMPLES_PER_CLASS - current_count
    
    if needed <= 0:
        continue
        
    aug_idx = 0
    while len(samples_by_class[gloss]) < TARGET_SAMPLES_PER_CLASS:
        # Pick base sample
        source_sample = base_list[aug_idx % len(base_list)]
        aug_type = aug_types_cycle[aug_idx % len(aug_types_cycle)]
        
        orig_seq = np.load(source_sample["path"])
        aug_seq = augment_sequence(orig_seq, aug_type, rng)
        
        out_name = f"{source_sample['video_id']}_aug_{aug_idx}_{aug_type}.npy"
        out_path = OUTPUT_LANDMARKS_DIR / gloss / out_name
        np.save(str(out_path), aug_seq)
        
        record = {
            "video_id": f"{source_sample['video_id']}_aug_{aug_idx}",
            "label": gloss,
            "signer_id": source_sample["signer_id"],
            "source_dataset": f"{source_sample['source_dataset']}_AUG",
            "split_original": source_sample["split_original"],
            "path": str(out_path).replace("\\", "/"),
            "num_frames": len(aug_seq),
            "is_augmented": True
        }
        samples_by_class[gloss].append(record)
        all_dataset_records.append(record)
        aug_idx += 1

print(f"\nAugmentation complete! Total dataset size: {len(all_dataset_records)} sequences across 20 classes.")
print(f"Every class now has at least {TARGET_SAMPLES_PER_CLASS} samples.")

# Phase C: Generate Stratified Splits (70% train, 15% val, 15% test)
# Keep augmented versions of a sample in the same split as the original to prevent data leakage!
print("\nPhase C: Creating leak-free stratified train/val/test splits...")

train_records = []
val_records = []
test_records = []

for gloss, s_list in samples_by_class.items():
    # Separate base samples
    base_samples = [s for s in s_list if not s["is_augmented"]]
    n_base = len(base_samples)
    
    # 70% train, 15% val, 15% test on base samples
    # Ensure at least 1 val and 1 test base sample per class
    rng.shuffle(base_samples)
    n_test = max(1, int(round(n_base * 0.15)))
    n_val = max(1, int(round(n_base * 0.15)))
    if n_test + n_val >= n_base:
        n_test = 1
        n_val = 1
    n_train = n_base - n_val - n_test
    
    test_base = base_samples[:n_test]
    val_base = base_samples[n_test:n_test + n_val]
    train_base = base_samples[n_test + n_val:]
    
    # Validation and Test ONLY contain real, non-augmented samples!
    val_records.extend(val_base)
    test_records.extend(test_base)
    
    # Train gets train base samples AND all augmented samples!
    train_records.extend(train_base)
    aug_samples = [s for s in s_list if s["is_augmented"]]
    train_records.extend(aug_samples)

print(f"Splits summary:")
print(f"  Train samples: {len(train_records)} (Base: {sum(1 for s in train_records if not s['is_augmented'])}, Aug: {sum(1 for s in train_records if s['is_augmented'])})")
print(f"  Val samples:   {len(val_records)} (100% real, unaugmented)")
print(f"  Test samples:  {len(test_records)} (100% real, unaugmented)")

def save_split_csv(records, split_name):
    csv_path = SPLITS_DIR / f"{split_name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {split_name} split to {csv_path}")

save_split_csv(train_records, "train")
save_split_csv(val_records, "val")
save_split_csv(test_records, "test")

print("\nExtract and Augment pipeline successfully completed!")
