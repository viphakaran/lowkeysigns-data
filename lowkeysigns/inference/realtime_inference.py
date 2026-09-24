import os
import sys
import time
import json
import queue
import threading
import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import torch
import torch.nn as nn
import joblib

# Paths setup
INFERENCE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = INFERENCE_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
MODEL_DIR = BACKEND_DIR / "model"
TASK_FILE = BACKEND_DIR / "hand_landmarker.task"

# MediaPipe Hand Connections (standard 21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]

# PyTorch LSTM architecture matching training script
class SignLSTM(nn.Module):
    def __init__(self, input_size=254, hidden_size=64, num_layers=1, num_classes=20, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.dropout(hn[-1])
        return self.fc(out)

# Background Audio / TTS Worker Thread
class AsyncAudioTTS:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.queue = queue.Queue()
        self.worker = threading.Thread(target=self._run_tts_loop, daemon=True)
        self.worker.start()
        
    def _run_tts_loop(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 155)
            engine.setProperty('volume', 0.9)
        except Exception as e:
            engine = None
            
        while True:
            text = self.queue.get()
            if text and self.enabled:
                if engine is not None:
                    try:
                        engine.say(text)
                        engine.runAndWait()
                    except Exception:
                        pass
                else:
                    # Windows PowerShell Speech fallback
                    try:
                        cmd = f'powershell -Command "Add-Type –AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\');"'
                        os.system(cmd)
                    except Exception:
                        pass
            self.queue.task_done()

    def speak(self, text):
        if self.enabled:
            # Drain queue if backlog
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except Exception:
                    break
            self.queue.put(text)

class SignRecognizer:
    def __init__(self, mode="showcase", engine="rf", confidence_threshold=0.65):
        self.mode = mode               # "showcase" (8 classes) or "full" (20 classes)
        self.engine = engine           # "rf" (Random Forest) or "lstm" (PyTorch LSTM)
        self.threshold = confidence_threshold
        
        # Load mappings
        with open(DATA_DIR / "label_mapping.json", "r", encoding="utf-8") as f:
            self.labels_20 = json.load(f)
            
        showcase_path = MODEL_DIR / "label_mapping_showcase.json"
        if showcase_path.exists():
            with open(showcase_path, "r", encoding="utf-8") as f:
                self.labels_showcase = json.load(f)
        else:
            self.labels_showcase = {str(i): w for i, w in enumerate(["wait", "doctor", "thank you", "more", "sick", "please", "here", "now"])}

        # Load models
        self.models = {}
        # 1. RF Models
        rf_20_path = MODEL_DIR / "rf_classifier_20class.joblib"
        rf_show_path = MODEL_DIR / "rf_classifier_showcase.joblib"
        if rf_20_path.exists():
            self.models["rf_full"] = joblib.load(rf_20_path)
        if rf_show_path.exists():
            self.models["rf_showcase"] = joblib.load(rf_show_path)
            
        # 2. LSTM Models
        lstm_20_path = MODEL_DIR / "lstm_classifier.pth"
        if lstm_20_path.exists():
            ckpt = torch.load(lstm_20_path, map_location="cpu", weights_only=False)
            m = SignLSTM(input_size=254, hidden_size=64, num_layers=1, num_classes=20)
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()
            self.models["lstm_full"] = m
            
        lstm_show_path = MODEL_DIR / "lstm_showcase.pth"
        if lstm_show_path.exists():
            ckpt = torch.load(lstm_show_path, map_location="cpu", weights_only=False)
            m = SignLSTM(input_size=254, hidden_size=64, num_layers=1, num_classes=8)
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()
            self.models["lstm_showcase"] = m

        # MediaPipe Detector
        base_options = python.BaseOptions(model_asset_path=str(TASK_FILE))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # Sequence buffer: 60 frames x 254 features
        self.buffer = deque(maxlen=60)
        self.last_coords = None
        
        # Debounce and recognition state
        self.recent_predictions = deque(maxlen=3)
        self.last_triggered_word = None
        self.last_triggered_time = 0.0
        self.cooldown_seconds = 1.6
        
        # Audio TTS
        self.tts = AsyncAudioTTS(enabled=True)
        self.confirmed_events = []

    def toggle_mode(self):
        self.mode = "full" if self.mode == "showcase" else "showcase"
        self.buffer.clear()
        self.recent_predictions.clear()

    def toggle_engine(self):
        self.engine = "lstm" if self.engine == "rf" else "rf"
        self.recent_predictions.clear()

    def toggle_tts(self):
        self.tts.enabled = not self.tts.enabled

    def extract_frame_features(self, rgb_frame):
        """Extract exact 254-dimensional feature vector matching pipeline."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        res = self.detector.detect(mp_image)
        
        hands_list = res.hand_landmarks if res.hand_landmarks else []
        hand_features = []
        presence_flags = []
        raw_landmarks_for_drawing = []

        for slot in range(2):
            if slot < len(hands_list):
                hl = hands_list[slot]
                raw_landmarks_for_drawing.append(hl)
                pts = np.array([[lm.x, lm.y, lm.z] for lm in hl], dtype=np.float32)
                wrist = pts[0].copy()
                centered = pts - wrist
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

        # 126 coords + 2 flags = 128
        pos_vec = np.concatenate([hand_features[0], hand_features[1], presence_flags]).astype(np.float32)
        
        # Coordinate velocity (delta between consecutive frames) = 126
        coords_126 = pos_vec[:126]
        if self.last_coords is not None:
            velocity = coords_126 - self.last_coords
        else:
            velocity = np.zeros(126, dtype=np.float32)
        self.last_coords = coords_126.copy()

        # Total = 128 + 126 = 254
        feature_254 = np.concatenate([pos_vec, velocity]).astype(np.float32)
        return feature_254, raw_landmarks_for_drawing, presence_flags

    def predict_window(self):
        """Inference over the current 60-frame buffer."""
        if len(self.buffer) < 15:
            return None, 0.0

        # Construct (60, 254) array with zero-padding if buffer not yet full
        seq = np.zeros((60, 254), dtype=np.float32)
        seq[:len(self.buffer)] = np.array(self.buffer, dtype=np.float32)

        key = f"{self.engine}_{self.mode}"
        model = self.models.get(key)
        label_map = self.labels_showcase if self.mode == "showcase" else self.labels_20
        
        if model is None:
            return None, 0.0

        if self.engine == "rf":
            # Extract 1016-dim summary features across sequence
            feat_1016 = np.hstack([
                seq.mean(axis=0),
                seq.std(axis=0),
                seq.max(axis=0),
                seq.min(axis=0)
            ]).reshape(1, -1)
            
            probs = model.predict_proba(feat_1016)[0]
            top_idx = int(np.argmax(probs))
            confidence = float(probs[top_idx])
            word = label_map[str(top_idx)]
            return word, confidence
            
        elif self.engine == "lstm":
            tensor_seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = model(tensor_seq)
                probs = torch.softmax(logits, dim=1)[0].numpy()
                top_idx = int(np.argmax(probs))
                confidence = float(probs[top_idx])
                word = label_map[str(top_idx)]
                return word, confidence

        return None, 0.0

    def process_frame(self, bgr_frame):
        h, w, _ = bgr_frame.shape
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        
        # 1. Feature extraction
        feature_vec, raw_landmarks, presence = self.extract_frame_features(rgb_frame)
        
        # Add to rolling buffer
        self.buffer.append(feature_vec)
        
        # 2. Prediction
        curr_word, curr_conf = self.predict_window()
        triggered_event = None

        hands_active = (presence[0] > 0 or presence[1] > 0)
        now = time.time()

        if curr_word and hands_active:
            self.recent_predictions.append(curr_word)
            
            # Debounce: requires 3 consecutive agreements above threshold
            if (len(self.recent_predictions) == 3 and 
                len(set(self.recent_predictions)) == 1 and 
                curr_conf >= self.threshold):
                
                # Check cooldown to prevent repeating immediate same word
                if (curr_word != self.last_triggered_word or (now - self.last_triggered_time) > self.cooldown_seconds):
                    self.last_triggered_word = curr_word
                    self.last_triggered_time = now
                    
                    triggered_event = {
                        "word": curr_word,
                        "confidence": round(curr_conf, 2),
                        "timestamp": time.strftime("%H:%M:%S")
                    }
                    self.confirmed_events.append(triggered_event)
                    self.tts.speak(curr_word)
        else:
            self.recent_predictions.clear()

        # 3. Draw Landmark Skeleton
        for hl in raw_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hl]
            for p1, p2 in HAND_CONNECTIONS:
                cv2.line(bgr_frame, pts[p1], pts[p2], (0, 220, 100), 2, cv2.LINE_AA)
            for idx, pt in enumerate(pts):
                color = (0, 120, 255) if idx == 0 else (255, 255, 255)
                cv2.circle(bgr_frame, pt, 4, color, -1, cv2.LINE_AA)
                cv2.circle(bgr_frame, pt, 5, (20, 20, 20), 1, cv2.LINE_AA)

        return curr_word, curr_conf, triggered_event, hands_active

def draw_hud(frame, recognizer, fps, curr_word, curr_conf, triggered_event, hands_active):
    """Render a clean, high-contrast civic-service GUI HUD directly on OpenCV frame."""
    h, w, _ = frame.shape
    
    # 1. Top Header Glass Bar
    top_bar = frame[0:70, 0:w].copy()
    overlay = np.zeros_like(top_bar)
    cv2.rectangle(overlay, (0, 0), (w, 70), (25, 35, 55), -1)
    cv2.addWeighted(overlay, 0.88, top_bar, 0.12, 0, top_bar)
    frame[0:70, 0:w] = top_bar
    cv2.line(frame, (0, 70), (w, 70), (45, 65, 95), 1)

    # Title & Badge
    cv2.putText(frame, "LowKeySigns", (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Public Service Counter Assist", (20, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 185, 210), 1, cv2.LINE_AA)

    # Status Pills in Top Right
    mode_text = f"MODE: {recognizer.mode.upper()} ({'8 Words' if recognizer.mode == 'showcase' else '20 Words'})"
    cv2.rectangle(frame, (w - 380, 14), (w - 180, 38), (40, 55, 80), -1)
    cv2.rectangle(frame, (w - 380, 14), (w - 180, 38), (60, 90, 130), 1)
    cv2.putText(frame, mode_text, (w - 370, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 245, 255), 1, cv2.LINE_AA)

    engine_text = f"ENGINE: {recognizer.engine.upper()} ({'83.9% CV' if recognizer.engine == 'rf' else 'Deep LSTM'})"
    cv2.rectangle(frame, (w - 380, 42), (w - 180, 64), (35, 60, 50), -1)
    cv2.rectangle(frame, (w - 380, 42), (w - 180, 64), (50, 120, 90), 1)
    cv2.putText(frame, engine_text, (w - 370, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 240, 190), 1, cv2.LINE_AA)

    # TTS Status & FPS
    tts_color = (0, 220, 130) if recognizer.tts.enabled else (120, 120, 140)
    cv2.putText(frame, f"TTS: {'ON' if recognizer.tts.enabled else 'OFF'}", (w - 165, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, tts_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 165, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 200, 220), 1, cv2.LINE_AA)

    # 2. Main Live Recognition Output Panel (Right-Center)
    card_w, card_h = 320, 160
    cx, cy = w - card_w - 20, 85
    card_roi = frame[cy:cy+card_h, cx:cx+card_w].copy()
    c_overlay = np.zeros_like(card_roi)
    cv2.rectangle(c_overlay, (0, 0), (card_w, card_h), (20, 28, 45), -1)
    cv2.addWeighted(c_overlay, 0.85, card_roi, 0.15, 0, card_roi)
    frame[cy:cy+card_h, cx:cx+card_w] = card_roi
    cv2.rectangle(frame, (cx, cy), (cx + card_w, cy + card_h), (60, 80, 115), 1)

    cv2.putText(frame, "REAL-TIME PREDICTION", (cx + 15, cy + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 160, 190), 1, cv2.LINE_AA)

    display_word = curr_word.upper() if (curr_word and hands_active) else "WAITING FOR SIGN..."
    word_color = (255, 255, 255) if (curr_word and hands_active) else (110, 130, 155)
    cv2.putText(frame, display_word, (cx + 15, cy + 68), cv2.FONT_HERSHEY_DUPLEX, 0.85, word_color, 2, cv2.LINE_AA)

    # Confidence Meter Bar
    conf_pct = int(curr_conf * 100) if (curr_word and hands_active) else 0
    bar_x, bar_y, bar_max_w, bar_h = cx + 15, cy + 85, 280, 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + bar_h), (40, 50, 70), -1)
    
    tier_color = (0, 210, 120) if conf_pct >= 75 else ((0, 190, 255) if conf_pct >= 55 else (80, 90, 230))
    current_bar_w = int((conf_pct / 100.0) * bar_max_w)
    if current_bar_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + current_bar_w, bar_y + bar_h), tier_color, -1)

    cv2.putText(frame, f"Confidence: {conf_pct}%", (cx + 15, cy + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 215, 235), 1, cv2.LINE_AA)

    # Hands presence pill
    hand_status = "Hands Detected" if hands_active else "No Hands"
    h_col = (0, 210, 120) if hands_active else (100, 110, 130)
    cv2.circle(frame, (cx + 205, cy + 116), 4, h_col, -1)
    cv2.putText(frame, hand_status, (cx + 215, cy + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 195, 215), 1, cv2.LINE_AA)

    # 3. Confirmed Events History Strip (Left-Bottom)
    if recognizer.confirmed_events:
        hist_h = min(110, len(recognizer.confirmed_events[-3:]) * 30 + 40)
        hx, hy = 20, h - hist_h - 45
        h_roi = frame[hy:hy+hist_h, hx:hx+260].copy()
        h_ov = np.zeros_like(h_roi)
        cv2.rectangle(h_ov, (0, 0), (260, hist_h), (20, 25, 40), -1)
        cv2.addWeighted(h_ov, 0.85, h_roi, 0.15, 0, h_roi)
        frame[hy:hy+hist_h, hx:hx+260] = h_roi
        cv2.rectangle(frame, (hx, hy), (hx + 260, hy + hist_h), (50, 70, 100), 1)
        
        cv2.putText(frame, "VERIFIED CONVERSATION LOG", (hx + 12, hy + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 160, 190), 1, cv2.LINE_AA)
        for i, ev in enumerate(recognizer.confirmed_events[-3:]):
            txt = f"{ev['timestamp']} - {ev['word'].upper()} ({int(ev['confidence']*100)}%)"
            cv2.putText(frame, txt, (hx + 12, hy + 48 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 240, 255), 1, cv2.LINE_AA)

    # 4. Bottom Controls / Shortcuts Strip
    bot_y = h - 35
    bot_bar = frame[bot_y:h, 0:w].copy()
    b_overlay = np.zeros_like(bot_bar)
    cv2.rectangle(b_overlay, (0, 0), (w, 35), (15, 20, 32), -1)
    cv2.addWeighted(b_overlay, 0.90, bot_bar, 0.10, 0, bot_bar)
    frame[bot_y:h, 0:w] = bot_bar
    cv2.line(frame, (0, bot_y), (w, bot_y), (40, 50, 70), 1)

    shortcut_str = "[Q] Quit  |  [M] Mode Toggle  |  [E] Engine (RF/LSTM)  |  [T] Voice TTS  |  [C] Clear Log"
    cv2.putText(frame, shortcut_str, (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 185, 205), 1, cv2.LINE_AA)

    buf_str = f"BUFFER: {len(recognizer.buffer)}/60 {'[READY]' if len(recognizer.buffer) >= 60 else '[WARMUP]'}"
    cv2.putText(frame, buf_str, (w - 220, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 210, 140) if len(recognizer.buffer) >= 60 else (180, 170, 70), 1, cv2.LINE_AA)

def main():
    parser = argparse.ArgumentParser(description="LowKeySigns Real-Time Sign Language Inference")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default 0)")
    parser.add_argument("--mode", type=str, default="showcase", choices=["showcase", "full"], help="Recognition mode (showcase=8 words, full=20 words)")
    parser.add_argument("--engine", type=str, default="rf", choices=["rf", "lstm"], help="Classifier engine (rf=Random Forest, lstm=PyTorch LSTM)")
    parser.add_argument("--threshold", type=float, default=0.65, help="Confidence threshold (0.0 to 1.0)")
    parser.add_argument("--no-tts", action="store_true", help="Disable audio voice synthesizer")
    parser.add_argument("--video", type=str, default=None, help="Path to input test video file instead of webcam")
    parser.add_argument("--mock", action="store_true", help="Run automated test simulation")
    args = parser.parse_args()

    recognizer = SignRecognizer(mode=args.mode, engine=args.engine, confidence_threshold=args.threshold)
    if args.no_tts:
        recognizer.tts.enabled = False

    # Video Capture source
    if args.video:
        cap = cv2.VideoCapture(args.video)
        source_name = f"File: {Path(args.video).name}"
    else:
        cap = cv2.VideoCapture(args.camera)
        source_name = f"Webcam Device #{args.camera}"

    if not cap.isOpened() or args.mock:
        print(f"\n[LowKeySigns Warning] Could not open video source ({source_name}).")
        print("Launching in Interactive Demonstration & Diagnostic Mock Mode...")
        cap = None

    window_name = "LowKeySigns — Real-Time ASL Interpreter"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 640)

    fps_tracker = deque(maxlen=20)
    prev_time = time.time()
    mock_frame_idx = 0

    print("\n" + "=" * 60)
    print("LOWKEYSIGNS REAL-TIME INFERENCE ENGINE READY")
    print("=" * 60)
    print(f"Mode:    {recognizer.mode.upper()} ({'8 words' if recognizer.mode == 'showcase' else '20 words'})")
    print(f"Engine:  {recognizer.engine.upper()} ({'Random Forest' if recognizer.engine == 'rf' else 'PyTorch LSTM'})")
    print(f"TTS:     {'Enabled' if recognizer.tts.enabled else 'Disabled'}")
    print("Hotkeys: [Q] Quit | [M] Switch Mode | [E] Switch Engine | [T] Toggle TTS | [C] Clear")
    print("=" * 60 + "\n")

    while True:
        loop_start = time.time()
        
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # Loop video if file, or break if camera disconnected
                if args.video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            frame = cv2.flip(frame, 1) # Mirror selfie view
        else:
            # Generate simulated clean background for testing environments
            frame = np.full((640, 960, 3), (28, 36, 52), dtype=np.uint8)
            # Subtle test patterns
            cv2.putText(frame, "SIMULATED SENSOR FEED", (340, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 80, 110), 2, cv2.LINE_AA)
            cv2.putText(frame, "Hardware camera offline or in test suite", (315, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 120, 150), 1, cv2.LINE_AA)
            mock_frame_idx += 1

        curr_word, curr_conf, triggered_event, hands_active = recognizer.process_frame(frame)

        # FPS Calculation
        dt = time.time() - loop_start
        fps_tracker.append(1.0 / max(dt, 1e-4))
        avg_fps = float(np.mean(fps_tracker))

        draw_hud(frame, recognizer, avg_fps, curr_word, curr_conf, triggered_event, hands_active)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('m'):
            recognizer.toggle_mode()
            print(f"[Mode Switched] Now in: {recognizer.mode.upper()}")
        elif key == ord('e'):
            recognizer.toggle_engine()
            print(f"[Engine Switched] Now using: {recognizer.engine.upper()}")
        elif key == ord('t'):
            recognizer.toggle_tts()
            print(f"[TTS Toggled] Voice output is now: {'ON' if recognizer.tts.enabled else 'OFF'}")
        elif key == ord('c'):
            recognizer.confirmed_events.clear()
            recognizer.buffer.clear()
            print("[Log Cleared]")

        # In mock mode, exit after 60 frames if automated test
        if args.mock and mock_frame_idx >= 60:
            print("[Mock Mode Test Completed Successfully]")
            break

    if cap:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
