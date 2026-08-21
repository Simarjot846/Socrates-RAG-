import os
import shutil
import tempfile
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Import RAGPipeline (it will import and use sentence-transformers and chromadb)
from pipeline import RAGPipeline

app = FastAPI(title="Voice-Enabled RAG Hackathon API")

# Enable CORS for frontend compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline container
pipeline = None

@app.on_event("startup")
def startup_event():
    global pipeline
    print("Initializing RAG Pipeline components on startup...")
    pipeline = RAGPipeline()
    print("RAG Pipeline initialized and ready.")

class TextQueryPayload(BaseModel):
    query: str

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
    
    # Save UploadFile to a local temporary file on D drive (since C drive is out of space)
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
        # Clean up temp file
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
        count = pipeline.collection.count()
        return {
            "status": "ready",
            "db_count": count,
            "embedding_model": pipeline.model_name,
            "groq_model": pipeline.groq_model
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "db_count": 0}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
