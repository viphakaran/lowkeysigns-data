from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np

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
    tokens: List[str]

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
    
    # Run temporal decoder
    committed_sign = temporal_decoder.step(top1_label, confidence)
    
    return {
        "predicted_label": top1_label,
        "confidence": confidence,
        "top3": top3,
        "committed_sign": committed_sign
    }

@app.post("/build_phrase")
def build_phrase(payload: TokensPayload):
    phrase_result = phrase_builder.build_phrase(payload.tokens)
    return phrase_result

@app.post("/reset_decoder")
def reset_decoder():
    temporal_decoder.reset()
    return {"status": "decoder_reset"}
