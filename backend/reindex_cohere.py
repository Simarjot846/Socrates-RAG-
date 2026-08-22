"""
Run this ONCE locally to re-index Qdrant with Cohere embeddings.
Usage: python reindex_cohere.py
"""
import os
import re
import time
import uuid
import requests
import pandas as pd
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")
COHERE_MODEL = "embed-multilingual-light-v3.0"
COHERE_DIM = 384
QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "msmarco_rag")
LIMIT_PER_LANG = int(os.environ.get("DATASET_SUBSET_LIMIT", "150"))
LANGUAGES = [l.strip() for l in os.environ.get("LANGUAGES", "hi,ta,bn,te,mr").split(",") if l.strip()]
HF_REPO_ID = os.environ.get("HF_REPO_ID", "ai4bharat/MSMARCO-XI")

lang_map = {
    "as": "asm", "bn": "ben", "gu": "guj", "hi": "hin", "kn": "kan",
    "ml": "mal", "mr": "mar", "ne": "nep", "or": "ori", "pa": "pan",
    "sa": "san", "ta": "tam", "te": "tel", "ur": "urd"
}

def cohere_embed(texts: list[str], input_type="search_document") -> list[list[float]]:
    url = "https://api.cohere.com/v1/embed"
    headers = {"Authorization": f"Bearer {COHERE_API_KEY}", "Content-Type": "application/json"}
    payload = {"texts": texts, "model": COHERE_MODEL, "input_type": input_type, "truncate": "END"}
    for attempt in range(3):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            if r.status_code == 200:
                return r.json()["embeddings"]
            print(f"Cohere status {r.status_code}: {r.text[:100]}")
        except Exception as e:
            print(f"Cohere attempt {attempt+1} failed: {e}")
        time.sleep(2)
    raise RuntimeError("Cohere embedding failed")

def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?|।])\s+', text) if s.strip()]

def chunk_fixed(text, chunk_size=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - 20):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def chunk_boundary(text, max_words=200):
    sentences = split_sentences(text)
    chunks, current, count = [], [], 0
    for s in sentences:
        wc = len(s.split())
        if count + wc > max_words and current:
            chunks.append(" ".join(current))
            current, count = [s], wc
        else:
            current.append(s)
            count += wc
    if current:
        chunks.append(" ".join(current))
    return chunks

def chunk_semantic(text):
    sentences = split_sentences(text)
    if len(sentences) <= 2:
        return [text]
    # Simple: group every 3 sentences
    chunks = []
    for i in range(0, len(sentences), 3):
        chunks.append(" ".join(sentences[i:i+3]))
    return chunks

# Connect Qdrant
print("Connecting to Qdrant...")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Recreate collection with Cohere dim (384)
try:
    client.delete_collection(COLLECTION_NAME)
    print(f"Deleted old collection: {COLLECTION_NAME}")
except Exception:
    pass

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(size=COHERE_DIM, distance=models.Distance.COSINE)
)
print(f"Created collection: {COLLECTION_NAME} (dim={COHERE_DIM})")

all_docs, all_metas, all_ids = [], [], []

for lang in LANGUAGES:
    lang_prefix = lang_map.get(lang, lang)
    file_name = f"validation/{lang_prefix}val.parquet"
    url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{file_name}"
    temp_dest = f"temp_{lang}.parquet"

    print(f"\nDownloading {lang}...")
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(temp_dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
    except Exception as e:
        print(f"Failed to download {lang}: {e}")
        continue

    try:
        df = pd.read_parquet(temp_dest, columns=["query_id", "passages"])
        rows = df.head(LIMIT_PER_LANG).to_dict(orient="records")
        print(f"Loaded {len(rows)} rows for {lang}")
    except Exception as e:
        print(f"Failed to read {lang}: {e}")
        os.remove(temp_dest)
        continue

    for row in rows:
        query_id = str(row.get("query_id", ""))
        passages = row.get("passages", {})
        target = passages.get("Translated_passages") or passages.get("English_passages") or []
        if isinstance(target, np.ndarray):
            target = target.tolist()
        for p_idx, passage in enumerate(target):
            if not passage or not str(passage).strip():
                continue
            passage = str(passage)
            src_id = f"{query_id}_{p_idx}"
            for strategy, chunks in [("fixed", chunk_fixed(passage)), ("boundary", chunk_boundary(passage)), ("semantic", chunk_semantic(passage))]:
                for c_idx, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue
                    cid = f"{strategy}_{lang}_{src_id}_{c_idx}"
                    all_docs.append(chunk)
                    all_ids.append(cid)
                    all_metas.append({"chunk_id": cid, "strategy": strategy, "source_passage_id": src_id, "language": lang, "text": chunk})

    os.remove(temp_dest)

print(f"\nTotal chunks: {len(all_docs)}")
print("Embedding and uploading in batches of 96...")

BATCH = 96
total_uploaded = 0
for i in range(0, len(all_docs), BATCH):
    batch_docs = all_docs[i:i+BATCH]
    batch_metas = all_metas[i:i+BATCH]
    batch_ids = all_ids[i:i+BATCH]

    vectors = cohere_embed(batch_docs, input_type="search_document")
    points = []
    for idx in range(len(batch_docs)):
        points.append(models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, batch_ids[idx])),
            vector=vectors[idx],
            payload=batch_metas[idx]
        ))
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    total_uploaded += len(points)
    print(f"Uploaded {total_uploaded}/{len(all_docs)} chunks...")
    time.sleep(0.3)  # respect Cohere rate limits

print(f"\nDone! Total chunks indexed: {total_uploaded}")
print(f"Collection count: {client.get_collection(COLLECTION_NAME).points_count}")
