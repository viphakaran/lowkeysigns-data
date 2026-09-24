from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import json
import asyncio
from datetime import datetime, timezone
import os

from app.inference_engine import ASLInferenceEngine
from app.temporal_decoder import TemporalDecoder
from app.phrase_builder import PhraseBuilder

app = FastAPI(
    title="LowKeySigns ASL Recognition API",
    description="Isolated ASL sign recognition prototype for selected service-counter vocabulary with multi-lingual translation (English, Tamil, Hindi).",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize singletons
inference_engine = ASLInferenceEngine()
temporal_decoder = TemporalDecoder(window_size=8, min_agreement=5, confidence_threshold=0.75, cooldown_frames=10)
phrase_builder = PhraseBuilder()

# Pydantic Schemas
class FrameSequencePayload(BaseModel):
    sequence: List[List[float]]  # (T, 268)

class TokensPayload(BaseModel):
    tokens: Optional[List[str]] = []

@app.get("/")
def root():
    return {
        "project": "LowKeySigns",
        "description": "Isolated ASL sign recognition prototype for selected service-counter vocabulary.",
        "status": "online",
        "model": "Bi-GRU + Temporal Attention",
        "classes_supported": 20,
        "languages": ["English", "Tamil", "Hindi"]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": inference_engine.model is not None,
        "classes_count": len(inference_engine.target_glosses)
    }

@app.get("/vocabulary")
def get_vocabulary():
    return {
        "total_classes": len(inference_engine.target_glosses),
        "target_glosses": inference_engine.target_glosses
    }

@app.post("/predict_sequence")
def predict_sequence(payload: FrameSequencePayload):
    raw_seq = np.array(payload.sequence, dtype=np.float32)
    if raw_seq.ndim != 2 or raw_seq.shape[1] != 268:
        raise HTTPException(status_code=400, detail=f"Expected feature shape (*, 268), got {raw_seq.shape}")
        
    top1_label, confidence, top3 = inference_engine.predict(raw_seq)
    committed_sign = temporal_decoder.step(top1_label, confidence)
    
    return {
        "predicted_label": top1_label,
        "confidence": confidence,
        "top3": top3,
        "committed_sign": committed_sign
    }

@app.post("/build_phrase")
def build_phrase(payload: Optional[TokensPayload] = None):
    tokens = payload.tokens if payload and payload.tokens else []
    phrase_result = phrase_builder.build_phrase(tokens)
    return phrase_result

@app.post("/reset_decoder")
def reset_decoder():
    temporal_decoder.reset()
    return {"status": "decoder_reset"}

# Pre-cache test sequences for live model demonstration when client streams are idle
test_manifest_path = "data/splits/test.csv"
demo_samples = []
if os.path.exists(test_manifest_path):
    try:
        import pandas as pd
        tdf = pd.read_csv(test_manifest_path)
        for _, row in tdf.iterrows():
            if os.path.exists(row["path"]):
                arr = np.load(row["path"])
                demo_samples.append((row["label"], arr))
    except Exception as e:
        print(f"Warning: could not load demo test samples: {e}")

@app.websocket("/ws")
async def websocket_feed(websocket: WebSocket):
    await websocket.accept()
    sample_idx = 0
    stop_event = asyncio.Event()

    async def client_listener():
        try:
            while not stop_event.is_set():
                try:
                    data_text = await websocket.receive_text()
                except Exception:
                    stop_event.set()
                    break
                try:
                    data = json.loads(data_text)
                    if "sequence" in data:
                        raw_seq = np.array(data["sequence"], dtype=np.float32)
                        top1_label, conf, top3 = inference_engine.predict(raw_seq)
                        committed = temporal_decoder.step(top1_label, conf)
                        if committed:
                            await websocket.send_json({
                                "word": committed,
                                "confidence": round(float(conf), 2),
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "top3": top3,
                                "source": "client_stream"
                            })
                    elif "ping" in data:
                        await websocket.send_json({"pong": True})
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            stop_event.set()

    async def demo_broadcast():
        nonlocal sample_idx
        try:
            # Short initial delay after connection before streaming live inferences
            await asyncio.sleep(2.0)
            while not stop_event.is_set():
                if demo_samples:
                    true_label, sample_arr = demo_samples[sample_idx % len(demo_samples)]
                    sample_idx += 1
                    # Run real inference using the trained PyTorch Bi-GRU model
                    pred_label, conf, top3 = inference_engine.predict(sample_arr)
                    try:
                        await websocket.send_json({
                            "word": pred_label,
                            "confidence": round(float(conf), 2),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "top3": top3,
                            "source": "live_model_evaluation"
                        })
                    except Exception:
                        stop_event.set()
                        break
                try:
                    await asyncio.sleep(4.0)
                except asyncio.CancelledError:
                    break
        except Exception:
            pass
        finally:
            stop_event.set()

    listener_task = asyncio.create_task(client_listener())
    broadcast_task = asyncio.create_task(demo_broadcast())

    try:
        await stop_event.wait()
    except Exception:
        pass
    finally:
        stop_event.set()
        listener_task.cancel()
        broadcast_task.cancel()
        await asyncio.gather(listener_task, broadcast_task, return_exceptions=True)
        try:
            await websocket.close()
        except Exception:
            pass


