"""
Standalone FastAPI embedding microservice.
Deploy this on HF Spaces (free) as a separate app.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

class EmbedRequest(BaseModel):
    text: str

@app.post("/embed")
def embed(req: EmbedRequest):
    vec = model.encode([req.text])[0].tolist()
    return {"embedding": vec}

@app.get("/health")
def health():
    return {"status": "ok"}
