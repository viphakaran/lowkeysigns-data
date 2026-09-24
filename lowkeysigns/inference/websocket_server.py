import asyncio
import json
import time
from pathlib import Path
from typing import Set

import cv2
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Import SignRecognizer from realtime_inference
from realtime_inference import SignRecognizer

app = FastAPI(title="LowKeySigns WebSocket Bridge")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

connected_clients: Set[WebSocket] = set()

# Shared recognizer instance (showcase mode by default for high precision)
recognizer = SignRecognizer(mode="showcase", engine="rf", confidence_threshold=0.65)
recognizer.tts.enabled = False  # Browser handles TTS on the frontend

async def recognition_producer():
    """Background task reading video frames and pushing recognition events to clients."""
    cap = cv2.VideoCapture(0)
    is_mock = not cap.isOpened()
    if is_mock:
        print("[WebSocket Bridge] No hardware webcam found, running mock event generator.")

    mock_counter = 0
    showcase_vocab = ["wait", "doctor", "thank you", "more", "sick", "please", "here", "now"]

    while True:
        if not is_mock:
            ret, frame = cap.read()
            if ret:
                curr_word, curr_conf, triggered_event, hands_active = recognizer.process_frame(frame)
                if triggered_event and connected_clients:
                    payload = json.dumps({
                        "word": triggered_event["word"],
                        "confidence": float(triggered_event["confidence"]),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "source": "live_webcam"
                    })
                    for client in list(connected_clients):
                        try:
                            await client.send_text(payload)
                        except Exception:
                            connected_clients.discard(client)
            await asyncio.sleep(0.03)  # ~30 FPS
        else:
            # Emit structured mock sign every 2.5s for testing
            mock_counter += 1
            word = showcase_vocab[mock_counter % len(showcase_vocab)]
            conf = round(0.78 + (mock_counter % 20) * 0.01, 2)
            payload = json.dumps({
                "word": word,
                "confidence": conf,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source": "simulated_bridge"
            })
            for client in list(connected_clients):
                try:
                    await client.send_text(payload)
                except Exception:
                    connected_clients.discard(client)
            await asyncio.sleep(2.5)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(recognition_producer())

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "LowKeySigns API",
        "mode": recognizer.mode,
        "engine": recognizer.engine
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    print(f"[WebSocket] Client connected (Total active: {len(connected_clients)})")
    try:
        while True:
            # Keepalive / incoming ping
            msg = await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.discard(websocket)
        print(f"[WebSocket] Client disconnected (Remaining: {len(connected_clients)})")

if __name__ == "__main__":
    uvicorn.run("websocket_server:app", host="127.0.0.1", port=8000, reload=False)
