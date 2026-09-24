# LowKeySigns — Isolated ASL Sign Recognition & Multilingual Service-Counter Pipeline

> **Scope Definition**: Isolated American Sign Language (ASL) sign recognition prototype for selected service-counter vocabulary.  
> This system recognizes **isolated signs**, not unrestricted continuous ASL conversation. It is designed for **American Sign Language (ASL)** and does not support Indian Sign Language (ISL).

---

## 1. Project Objective

LowKeySigns bridges communication at public service counters (hospitals, government desks, clinic reception, transit hubs) between Deaf individuals using American Sign Language and service-desk personnel. 

The pipeline translates isolated ASL gestures into structured, canonical multi-lingual service phrases across **English**, **Tamil**, and **Hindi**, providing immediate text feedback and low-latency speech synthesis.

---

## 2. Target Vocabulary (20 Signs)

The system is trained on 20 locked, service-counter vocabulary classes:

| Index | Gloss | Category | Functional Context |
| :---: | :--- | :---: | :--- |
| `0` | **help** | CORE_15 | Assistance requests |
| `1` | **wait** | CORE_15 | Queuing and timing |
| `2` | **money** | CORE_15 | Billing and fees |
| `3` | **form** | CORE_15 | Paperwork / intake |
| `4` | **pain** | CORE_15 | Medical triage |
| `5` | **doctor** | CORE_15 | Medical personnel |
| `6` | **yes** | CORE_15 | Confirmation |
| `7` | **no** | CORE_15 | Negation |
| `8` | **thank you** | CORE_15 | Courtesy |
| `9` | **sign** | CORE_15 | Verification / signature |
| `10` | **more** | CORE_15 | Repetition / quantity |
| `11` | **problem** | CORE_15 | Issue escalation |
| `12` | **emergency** | CORE_15 | Urgent response |
| `13` | **where** | CORE_15 | Wayfinding |
| `14` | **name** | CORE_15 | Identification |
| `15` | **appointment** | STRETCH_5 | Scheduling |
| `16` | **sick** | STRETCH_5 | Health state |
| `17` | **please** | STRETCH_5 | Politeness |
| `18` | **here** | STRETCH_5 | Location indicator |
| `19` | **now** | STRETCH_5 | Immediate urgency |

---

## 3. Dataset Sources, Attribution & Licenses

Treat all raw datasets as research/education resources. Do not redistribute raw videos.

1. **WLASL (Word-Level American Sign Language Video Dataset)**:
   - **Official Project**: [dxli94/WLASL](https://github.com/dxli94/WLASL)
   - **Metadata**: `WLASL_v0.3.json` (2,000 glosses, 318 target vocabulary entries).
   - **Terms**: Academic and computational research use.
2. **MS-ASL (Microsoft American Sign Language Dataset)**:
   - **Official Project**: [Microsoft Research MS-ASL](https://www.microsoft.com/en-us/research/project/ms-asl/)
   - **Terms**: Microsoft Research License Terms.
3. **ASL Citizen**:
   - **Official Project**: [Microsoft Research ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/)
   - **Terms**: Microsoft Research License Terms.
4. **ASLLVD (American Sign Language Lexicon Video Dataset)**:
   - **Official Project**: [Boston University ASLLRP](https://www.bu.edu/asllrp/av/dai-asllvd.html)
   - **Terms**: BU ASLLRP research terms.

---

## 4. Directory Structure

```
aslr_project/
├── data/
│   ├── raw/
│   │   ├── wlasl_official/             # Recovered official WLASL clips
│   │   ├── wlasl_mirror/videos/        # Indexed mirror videos
│   │   ├── msasl/                      # MS-ASL dataset metadata & clips
│   │   ├── asl_citizen/                # ASL Citizen references
│   │   └── asllvd/                     # ASLLVD verified entries
│   ├── selected/
│   │   └── landmarks/                  # Extracted 268-dim landmark sequences (.npy)
│   ├── metadata/
│   │   ├── WLASL_v0.3.json             # Official WLASL dataset metadata
│   │   ├── selected_wlasl_metadata.json# Target 20 classes filtered metadata
│   │   ├── mirror_inventory.csv        # Local video inventory & validation
│   │   ├── wlasl_mirror_match.csv      # Matched mirror inventory against target instances
│   │   ├── unified_manifest.csv        # Complete dataset manifest with origins & licenses
│   │   ├── url_status.json             # Diagnostic availability log of original URLs
│   │   ├── label_mapping.json          # Index to gloss dictionary
│   │   └── phrase_templates.json       # Service-counter multilingual phrase mappings
│   └── splits/
│       ├── train.csv                   # Stratified training split (518 sequences)
│       ├── val.csv                     # 100% real, unaugmented validation split (21 sequences)
│       └── test.csv                    # 100% real, unaugmented test split (21 sequences)
├── models/
│   ├── bigru_classifier.pth            # Trained PyTorch Bi-GRU checkpoint
│   ├── confusion_matrix.png            # Validation/Test confusion matrix plot
│   └── evaluation_report.json          # Classification report & per-class metrics
├── scripts/
│   ├── inspect_wlasl_vocabulary.py     # Gloss exact & related match analyzer
│   ├── check_wlasl_subsets.py          # WLASL100/300/1000/2000 subset membership inspector
│   ├── filter_wlasl_metadata.py        # 20-word metadata extractor
│   ├── build_mirror_inventory.py       # OpenCV file validation & dimension indexer
│   ├── match_wlasl_mirror.py           # Instance-to-file resolver
│   ├── check_wlasl_urls.py             # URL status & reachability auditor
│   ├── extract_and_augment.py          # 268-dim feature extractor & synthetic augmentor
│   └── train_bigru.py                  # Bi-GRU model trainer & evaluator
├── app/
│   ├── main.py                         # FastAPI service application
│   ├── inference_engine.py             # Real-time sequence prediction engine
│   ├── temporal_decoder.py             # 8-frame sliding window & majority voting stabilizer
│   └── phrase_builder.py               # Multilingual canonical phrase translator
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 5. Dataset Samples & Data Balancing

* **Original Available Videos**: 149 valid clips across all 20 classes.
* **Balanced Dataset**: Augmented via physics-constrained landmark perturbations (temporal warping $\pm 15\%$, coordinate jitter $\mathcal{N}(0, 0.008)$, spatial scaling $\pm 6\%$, 2D rotation $\pm 5^\circ$).
* **Total Sequences**: **560 sequences** across 20 classes (**28 verified/augmented samples per class**).
* **Split Policy**:
  * **Train Split**: 518 samples (real + augmented).
  * **Val Split**: 21 samples (100% unaugmented, real video landmarks).
  * **Test Split**: 21 samples (100% unaugmented, real video landmarks).
  * Strict leak-free isolation: Augmented variants are never placed in validation or test.

---

## 6. Feature Representation (268 Dimensions per Frame)

Every video frame is transformed into a **268-dimensional geometric representation**:
1. **Normalized Landmarks (126 values)**:
   - Up to 2 hands (slots 0 and 1), 21 landmarks each $(x, y, z)$.
   - Centered on wrist ($p_i - p_0$).
   - Scale-normalized by wrist-to-middle MCP distance $\|p_9 - p_0\|$.
2. **Presence Flags (2 values)**: Binary detection indicator `[hand0_present, hand1_present]`.
3. **Inter-Hand 3D Spatial Geometry (4 values)**:
   - 3D Euclidean distance between Hand 0 wrist and Hand 1 wrist.
   - Relative 3D displacement vector $(\Delta x, \Delta y, \Delta z)$ between wrists.
   *Distinguishes 2-handed contact signs (`more`, `help`, `problem`) from 1-handed signs (`yes`, `where`).*
4. **Fingertip-to-Wrist Extensions (10 values)**:
   - 5 Euclidean distances per hand from wrist to fingertips (Thumb, Index, Middle, Ring, Pinky).
   *Encodes hand posture (open palm, fist, index point, pinched tips).*
5. **Velocity Deltas (126 values)**:
   - Frame-to-frame coordinate displacement: $v_t = c_t - c_{t-1}$.

---

## 7. Model Architecture (Bi-GRU + Temporal Attention)

* **Recurrent Core**: 2-layer Bidirectional GRU (`hidden_size=64`, bidirectional output dimension = 128).
* **Attention Mechanism**: Temporal attention network weighting salient inflection frames.
* **Dropout**: 0.35 dropout on recurrent context and linear projection.
* **Parameters**: 212,437 trainable parameters.
* **Optimization**: AdamW (`lr=0.0015`, `weight_decay=1e-3`), Cosine Annealing learning rate schedule.

---

## 8. Evaluation & Test Results

Evaluated on unseen, real unaugmented test videos:

* **Top-1 Test Accuracy**: **100.0%**
* **Macro F1-Score**: **1.000**
* **Thin Class Performance**:
  * **`emergency`**: Precision: **1.00**, Recall: **1.00**, F1-Score: **1.00**
  * **`pain`**: Precision: **1.00**, Recall: **1.00**, F1-Score: **1.00**
* **Confusion Matrix**: Saved to `models/confusion_matrix.png`.
* **Full Report**: Saved to `models/evaluation_report.json`.

---

## 9. Temporal Decoding & Phrase Builder

To prevent single-frame flickering or stuttering:
* **Sliding Window**: Buffers the last 8 frame predictions.
* **Majority Voting**: Commits a sign token only if $\ge 5$ of the last 8 predictions agree with confidence $\ge 0.75$.
* **Cooldown**: Imposes a 10-frame debounce after committing a sign.
* **Canonical Phrase Mapping**:
  * `["help", "form"]` $\rightarrow$
    * **English**: *"I need help with the form."*
    * **Tamil**: *"படிவத்திற்கு உதவி வேண்டும்."*
    * **Hindi**: *"मुझे फ़ॉर्म में मदद चाहिए।"*
  * `["doctor", "pain"]` $\rightarrow$
    * **English**: *"I need to see a doctor, I am in pain."*
    * **Tamil**: *"எனக்கு மருத்துவர் வேண்டும், வலி அதிகமாக இருக்கிறது."*
    * **Hindi**: *"मुझे डॉक्टर से मिलना है, मुझे दर्द हो रहा है।"*
  * `["where", "emergency"]` $\rightarrow$
    * **English**: *"Where is the emergency department?"*
    * **Tamil**: *"அவசர சிகிச்சை பிரிவு எங்கே உள்ளது?"*
    * **Hindi**: *"आपातकालीन विभाग कहाँ है?"*

---

## 10. API Quickstart

### Start the Service:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints:
* `GET /` — Service status & metadata.
* `GET /vocabulary` — Supported 20 ASL classes.
* `POST /predict_sequence` — Accepts `(60, 268)` feature array, returns prediction + confidence.
* `POST /build_phrase` — Accepts list of tokens `["help", "form"]`, returns multilingual translations.
* `POST /reset_decoder` — Resets temporal stabilization buffer.

---

## 11. Known Limitations & Scope Boundaries

1. **Isolated Signs Only**: The model classifies individual sign sequences. It does not perform continuous finger-spelling or grammar parsing of rapid natural ASL discourse.
2. **ASL Specificity**: Trained exclusively on American Sign Language. Not compatible with Indian Sign Language (ISL) or British Sign Language (BSL).
3. **Lighting & Occlusion**: MediaPipe HandLandmarker performance degrades in extreme low-light environments or under heavy hand-on-hand occlusion.

---

## 12. Privacy & Ethics Statement

* No raw video footage is uploaded, stored, or redistributed by this project.
* Inference is performed entirely on mathematical landmark coordinate vectors.
* Identity features (facial characteristics, voice prints) are not captured or preserved in dataset manifests.

---

## 13. Citation

If using this pipeline or dataset metadata in academic or research settings:

```bibtex
@inproceedings{li2020wordlevel,
  title={Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison},
  author={Li, Dongxu and Rodriguez, Cristian and Yu, Xin and Li, Hongdong},
  booktitle={The IEEE Winter Conference on Applications of Computer Vision (WACV)},
  pages={1459--1469},
  year={2020}
}
```
