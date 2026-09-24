# LowKeySigns — WLASL Data Pipeline

LowKeySigns is a real-time American Sign Language (ASL) recognition pipeline designed for accessibility tech and public service environments.

This repository implements the end-to-end data pipeline from raw WLASL video dataset acquisition and metadata filtering to normalized landmark sequence extraction and train-ready dataset packaging.

---

## 1. Vocabulary (20 Words)

The vocabulary consists of 20 high-priority words for accessibility and public service interactions:

* **CORE_15**: `help`, `wait`, `money`, `form`, `pain`, `doctor`, `yes`, `no`, `thank you`, `sign`, `more`, `problem`, `emergency`, `where`, `name`
* **STRETCH_5**: `appointment`, `sick`, `please`, `here`, `now`

---

## 2. Directory Structure

```
lowkeysigns/
├── data/
│   ├── raw_videos/                     # Downloaded MP4 clips organized by gloss
│   ├── landmarks/                      # Extracted (num_frames, 254) float32 landmark .npy files
│   ├── selected_wlasl_metadata.json    # Filtered WLASL entries (318 instances)
│   ├── manifest.json                   # Real-time download & extraction tracking
│   ├── label_mapping.json              # Integer index to gloss mapping (0 to 19)
│   └── train_ready_dataset.npz         # Packaged X: (135, 60, 254), y: (135,)
├── extraction/
│   └── extract_landmarks.py            # Landmark extraction & normalization pipeline
├── model/                              # Reserved for classification models
├── inference/                          # Reserved for real-time inference
├── scripts/
│   ├── verify_step1.py                 # WLASL metadata schema validator
│   ├── verify_vocabulary.py            # Vocabulary verification against WLASL
│   ├── filter_metadata.py              # Metadata filtering script
│   ├── download_videos.py              # Concurrent video acquisition script
│   ├── report_downloads.py             # Video download reporting utility
│   ├── sanity_check.py                 # Quality inspection & sequence length metrics
│   ├── prepare_dataset.py              # Train-ready dataset generator
│   ├── vocabulary_verification_report.json
│   └── sanity_check_report.json
├── requirements.txt
└── README.md
```

---

## 3. Feature Representation (254 Dimensions per Frame)

Each video frame is transformed into a **254-dimensional feature vector**:
1. **Up to 2 Hands Tracked (Slots 0 and 1)**:
   - 21 landmarks per hand with $(x, y, z)$ coordinates = 63 values per hand.
   - Hand 0 coordinates + Hand 1 coordinates = 126 coordinate values.
2. **Per-Hand Normalization**:
   - Wrist landmark (index 0) subtracted from all 21 landmarks (wrist-centered).
   - Normalized by the scale reference distance between Wrist (index 0) and Middle Finger MCP (index 9).
3. **Presence Flags**:
   - 2 binary flags `[hand0_present, hand1_present]` indicating hand detection status.
   - Position feature vector: 126 coords + 2 flags = **128 values**.
4. **Velocity Features**:
   - Delta of the 126 coordinate values between consecutive frames ($t$ and $t-1$).
   - 126 zeros for the initial frame.
5. **Final Per-Frame Vector**: $128 \text{ (position)} + 126 \text{ (velocity)} = \mathbf{254} \text{ values}$ (`float32`).

---

## 4. Train-Ready Dataset Specification

* **Storage**: `data/train_ready_dataset.npz`
* **Array `X`**: Shape `(135, 60, 254)` (`float32`), fixed window of 60 frames (zero-padded if shorter, truncated from end if longer).
* **Array `y`**: Shape `(135,)` (`int64`), integer class labels `[0, 19]`.
* **Label Mapping**: `data/label_mapping.json`
* **Data Quality**: 0 NaN values, 0 Inf values across the entire dataset.
