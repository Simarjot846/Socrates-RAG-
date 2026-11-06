import os
import sys
import shutil
import tempfile
import time
import asyncio
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Ensure backend directory is in sys.path when run from outside this folder
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Load environment variables
load_dotenv(dotenv_path=os.path.join(backend_dir, ".env"))

from pipeline import RAGPipeline

app = FastAPI(title="Voice-Enabled RAG Hackathon API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = None

# ── Keep-alive self-ping ──────────────────────────────────────────────────────
# Render free tier spins down after 15 min of inactivity.
# This background task pings /api/ping every 10 minutes so the service
# stays warm and never cold-starts during a demo.
SELF_URL = os.environ.get("RENDER_EXTERNAL_URL", "")

async def _keep_alive_loop():
    """Ping own /api/ping every 10 minutes to prevent Render spin-down."""
    await asyncio.sleep(30)   # wait for startup to finish first
    while True:
        if SELF_URL:
            try:
                requests.get(f"{SELF_URL}/api/ping", timeout=10)
                print("[keep-alive] ping sent")
            except Exception as e:
                print(f"[keep-alive] ping failed: {e}")
        await asyncio.sleep(600)   # 10 minutes

@app.on_event("startup")
async def startup_event():
    global pipeline
    print("Initializing RAG Pipeline components on startup...")
    pipeline = RAGPipeline()
    print("RAG Pipeline initialized and ready.")
    # Start keep-alive background loop only when deployed on Render
    if SELF_URL:
        asyncio.create_task(_keep_alive_loop())
        print(f"[keep-alive] background pinger started → {SELF_URL}")

class TextQueryPayload(BaseModel):
    query: str

# ── Health / keep-alive endpoints ─────────────────────────────────────────────
@app.get("/api/ping")
def ping():
    """Lightweight keep-alive endpoint. Returns instantly."""
    return {"ok": True}

@app.post("/api/query-text")
async def query_text(payload: TextQueryPayload):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized yet")
    try:
        result = await pipeline.run(text_query=payload.query)
        return result
    except Exception as e:
        return {
            "error": f"Internal pipeline error: {str(e)}",
            "flagged": True,
            "grounded": False,
            "latency_ms": {"stt": 0, "guard_in": 0, "retrieval": 0, "generation": 0, "guard_out": 0, "total": 0}
        }

@app.post("/api/predict")
async def predict_audio(file: UploadFile = File(...)):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized yet")
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    ext = os.path.splitext(file.filename)[1] or ".wav"
    temp_file_path = os.path.join(temp_dir, f"upload_{os.getpid()}_{int(time.time())}{ext}")
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        result = await pipeline.run(audio_file_path=temp_file_path)
        return result
    except Exception as e:
        return {
            "error": f"Internal audio pipeline error: {str(e)}",
            "flagged": True,
            "grounded": False,
            "latency_ms": {"stt": 0, "guard_in": 0, "retrieval": 0, "generation": 0, "guard_out": 0, "total": 0}
        }
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass

@app.get("/api/status")
def get_status():
    if not pipeline:
        return {"status": "initializing", "db_count": 0}
    try:
        count = pipeline.qdrant_client.get_collection(pipeline.collection_name).points_count
        return {
            "status": "ready",
            "db_count": count,
            "embedding_model": pipeline.model_name,
            "groq_model": pipeline.groq_model
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "db_count": 0}

@app.get("/api/debug-net")
def debug_net():
    """Test which external APIs are reachable from Render."""
    endpoints = {
        "cohere": "https://api.cohere.com",
        "groq": "https://api.groq.com",
        "hf_inference": "https://api-inference.huggingface.co",
        "voyage": "https://api.voyageai.com",
        "together": "https://api.together.xyz",
        "jina": "https://api.jina.ai",
    }
    results = {}
    for name, url in endpoints.items():
        try:
            r = requests.get(url, timeout=5)
            results[name] = f"reachable (HTTP {r.status_code})"
        except Exception as e:
            results[name] = f"BLOCKED: {str(e)[:80]}"
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
