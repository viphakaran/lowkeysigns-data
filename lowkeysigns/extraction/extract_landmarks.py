import os
import sys
import json
import types
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm

MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    import urllib.request
    print("Downloading hand_landmarker.task model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

class NormalizedLandmark:
    __slots__ = ("x", "y", "z")
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class NormalizedLandmarkList:
    def __init__(self, landmark_list):
        self.landmark = [NormalizedLandmark(lm.x, lm.y, lm.z) for lm in landmark_list]

class HandsResult:
    def __init__(self, hand_landmarks):
        if hand_landmarks and len(hand_landmarks) > 0:
            self.multi_hand_landmarks = [NormalizedLandmarkList(hl) for hl in hand_landmarks]
        else:
            self.multi_hand_landmarks = None

class HandsCompat:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process(self, rgb_frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        res = self.detector.detect(mp_image)
        return HandsResult(res.hand_landmarks)

# Setup MediaPipe interface as specified in Step 5.1
mp.solutions = types.SimpleNamespace(hands=types.SimpleNamespace(Hands=HandsCompat))
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

METADATA_PATH = "data/selected_wlasl_metadata.json"
MANIFEST_PATH = "data/manifest.json"
RAW_VIDEOS_DIR = Path("data/raw_videos")
LANDMARKS_DIR = Path("data/landmarks")
LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    selected_metadata = json.load(f)

frame_lookup = {}
for entry in selected_metadata:
    for inst in entry["instances"]:
        frame_lookup[str(inst["video_id"])] = (inst.get("frame_start", 1), inst.get("frame_end", -1))

if os.path.exists(MANIFEST_PATH):
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
else:
    manifest = {}

# Gather all downloaded video files
video_tasks = []
for gloss_dir in sorted(RAW_VIDEOS_DIR.iterdir()):
    if gloss_dir.is_dir():
        gloss = gloss_dir.name
        for video_file in sorted(gloss_dir.glob("*.mp4")):
            video_tasks.append((gloss, video_file))

print(f"Total video clips to process for landmark extraction: {len(video_tasks)}")

def process_clip(gloss, video_path):
    video_id = video_path.stem
    frame_range = frame_lookup.get(video_id, (1, -1))
    frame_start, frame_end = frame_range
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, "Failed to open video"
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine frame slicing
    if total_frames > 0 and frame_start > 1 and total_frames < frame_start:
        # Video was already trimmed
        start_idx = 0
        end_idx = -1
    else:
        start_idx = max(0, frame_start - 1)
        end_idx = frame_end - 1 if frame_end > 0 else -1
        
    position_features_list = []
    empty_frames_mask = []
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx >= start_idx:
            if end_idx > 0 and frame_idx > end_idx:
                break
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # Step 5.3: Feature vector construction
            hand_features = []
            presence_flags = []
            
            multi_landmarks = results.multi_hand_landmarks if results.multi_hand_landmarks else []
            
            for slot in range(2):
                if slot < len(multi_landmarks):
                    hl = multi_landmarks[slot]
                    raw_coords = np.array([[lm.x, lm.y, lm.z] for lm in hl.landmark], dtype=np.float32)
                    
                    # Normalization:
                    wrist = raw_coords[0].copy()
                    centered = raw_coords - wrist
                    scale_ref = float(np.linalg.norm(centered[9]))
                    
                    if scale_ref < 1e-6:
                        hand_features.append(np.zeros(63, dtype=np.float32))
                        presence_flags.append(0.0)
                    else:
                        normed = centered / scale_ref
                        hand_features.append(normed.flatten())
                        presence_flags.append(1.0)
                else:
                    hand_features.append(np.zeros(63, dtype=np.float32))
                    presence_flags.append(0.0)
                    
            hand0_present, hand1_present = presence_flags
            is_empty = (hand0_present == 0.0 and hand1_present == 0.0)
            empty_frames_mask.append(is_empty)
            
            # Vector: 126 coordinate values + 2 presence flags = 128
            pos_vec = np.concatenate([hand_features[0], hand_features[1], presence_flags]).astype(np.float32)
            position_features_list.append(pos_vec)
            
        frame_idx += 1
        
    cap.release()
    
    raw_num_frames = len(position_features_list)
    if raw_num_frames == 0 or all(empty_frames_mask):
        return False, "no valid hand detections in clip"
        
    # Trim leading and trailing empty resting frames before and after the active sign
    first_detected = empty_frames_mask.index(False)
    last_detected = len(empty_frames_mask) - 1 - empty_frames_mask[::-1].index(False)
    
    active_pos_list = position_features_list[first_detected:last_detected + 1]
    active_empty_mask = empty_frames_mask[first_detected:last_detected + 1]
    num_frames = len(active_pos_list)
    
    if num_frames < 3:
        return False, "active sign length too short (< 3 frames)"
        
    # Step 5.5: Check for 3 or more CONSECUTIVE empty frames within the active signing segment
    max_consecutive_empty = 0
    current_consecutive = 0
    for is_empty in active_empty_mask:
        if is_empty:
            current_consecutive += 1
            if current_consecutive > max_consecutive_empty:
                max_consecutive_empty = current_consecutive
        else:
            current_consecutive = 0
            
    if max_consecutive_empty >= 3:
        return False, "too many consecutive empty frames"
        
    pos_array = np.array(active_pos_list, dtype=np.float32) # shape (num_frames, 128)
    
    # Linear interpolation if any empty frames (< 3 consecutive)
    valid_indices = [i for i, is_empty in enumerate(active_empty_mask) if not is_empty]
    if len(valid_indices) < num_frames:
        for d in range(126): # Interpolate coordinates
            pos_array[:, d] = np.interp(np.arange(num_frames), valid_indices, pos_array[valid_indices, d])
        # Presence flags
        pos_array[:, 126] = np.where(pos_array[:, 126] > 0.5, 1.0, 0.0)
        pos_array[:, 127] = np.where(pos_array[:, 127] > 0.5, 1.0, 0.0)
        
    # Step 5.4: Velocity features
    velocity = np.zeros((num_frames, 126), dtype=np.float32)
    if num_frames > 1:
        velocity[1:] = pos_array[1:, :126] - pos_array[:-1, :126]
        
    # Final per-frame feature vector = 128 + 126 = 254
    final_features = np.hstack([pos_array, velocity]).astype(np.float32)
    
    # Save landmark file
    out_dir = LANDMARKS_DIR / gloss
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_id}.npy"
    np.save(str(out_path), final_features)
    
    return True, f"Saved {num_frames} frames"

if __name__ == "__main__":
    processed_count = 0
    succeeded_clips = 0
    failed_clips = 0

    for idx, (gloss, video_path) in enumerate(tqdm(video_tasks, desc="Extracting landmarks")):
        video_id = video_path.stem
        ok, msg = process_clip(gloss, video_path)
        processed_count += 1
        
        if ok:
            succeeded_clips += 1
            if gloss in manifest:
                if video_id not in manifest[gloss]["succeeded"]:
                    manifest[gloss]["succeeded"].append(video_id)
                manifest[gloss]["failed"] = [f for f in manifest[gloss]["failed"] if f.get("video_id") != video_id]
        else:
            failed_clips += 1
            if gloss in manifest:
                if video_id in manifest[gloss]["succeeded"]:
                    manifest[gloss]["succeeded"].remove(video_id)
                if not any(f.get("video_id") == video_id for f in manifest[gloss]["failed"]):
                    manifest[gloss]["failed"].append({"video_id": video_id, "reason": msg})
                    
        # 5.7: Report progress every 10 clips processed
        if processed_count % 10 == 0 or processed_count == len(video_tasks):
            print(f"Status update: [{gloss}] - {processed_count}/{len(video_tasks)} clips processed ({succeeded_clips} succeeded, {failed_clips} skipped)")

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nExtraction complete! Total processed: {processed_count}, Successfully saved: {succeeded_clips}, Skipped: {failed_clips}")
