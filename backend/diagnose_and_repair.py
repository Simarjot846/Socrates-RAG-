import os
# Force offline mode for model loading to avoid slow network/update checks
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import re
import sys
import time
import requests
from collections import Counter
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import fsspec
import pyarrow.parquet as pq

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

LANGUAGES = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "sa", "ta", "te", "ur"]
MIN_CHUNKS_THRESHOLD = 50  # below this, treat the language as "under-indexed" and needs repair

# Global variables
model = None
tokenizer = None
chroma_client = None
chroma_collection = None

def get_token_count(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?|।])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_fixed_size(text: str, chunk_size=256, overlap=50) -> list[dict]:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    if not tokens:
        return chunks
    overlap = min(overlap, chunk_size - 1)
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append({
            "text": chunk_text,
            "token_count": len(chunk_tokens)
        })
        if i + chunk_size >= len(tokens):
            break
        i += (chunk_size - overlap)
    return chunks

def chunk_semantic(text: str, threshold=0.6) -> list[dict]:
    sentences = split_into_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [{"text": sentences[0], "token_count": get_token_count(sentences[0])}]
    
    embeddings = model.encode(sentences)
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(len(sentences) - 1):
        emb_curr = embeddings[i]
        emb_next = embeddings[i+1]
        dot_val = np.dot(emb_curr, emb_next)
        norm_curr = np.linalg.norm(emb_curr)
        norm_next = np.linalg.norm(emb_next)
        similarity = dot_val / (norm_curr * norm_next) if (norm_curr > 0 and norm_next > 0) else 0.0
        
        if similarity < threshold:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "token_count": get_token_count(chunk_text)
            })
            current_chunk = [sentences[i+1]]
        else:
            current_chunk.append(sentences[i+1])
            
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append({
            "text": chunk_text,
            "token_count": get_token_count(chunk_text)
        })
    return chunks

def chunk_boundary_aware(text: str, max_tokens=256) -> list[dict]:
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sent in sentences:
        sent_tokens = get_token_count(sent)
        if sent_tokens > max_tokens:
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "token_count": current_tokens
                })
                current_chunk = []
                current_tokens = 0
            chunks.append({
                "text": sent,
                "token_count": sent_tokens
            })
        elif current_tokens + sent_tokens > max_tokens:
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "token_count": current_tokens
                })
            current_chunk = [sent]
            current_tokens = sent_tokens
        else:
            current_chunk.append(sent)
            current_tokens += sent_tokens
            
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append({
            "text": chunk_text,
            "token_count": current_tokens
        })
    return chunks

def run_diagnostic():
    global chroma_client
    collection_name = os.environ.get("CHROMA_COLLECTION_NAME", "msmarco_rag")
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception as e:
        print(f"Error loading collection '{collection_name}': {e}")
        return [], {}

    try:
        results = collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])
    except Exception as e:
        print(f"Error fetching metadata from Chroma: {e}")
        metadatas = []

    lang_counts = Counter(m.get("language", "unknown") for m in metadatas if m)
    
    print("\n" + "="*50)
    print(f"{'Language':<15}{'Chunks':<15}{'Status'}")
    print("="*50)
    under_indexed = []
    for lang in LANGUAGES:
        count = lang_counts.get(lang, 0)
        status = "OK" if count >= MIN_CHUNKS_THRESHOLD else "NEEDS REPAIR"
        if count < MIN_CHUNKS_THRESHOLD:
            under_indexed.append(lang)
        print(f"{lang:<15}{count:<15}{status}")
    print("="*50)
    
    return under_indexed, lang_counts

def fetch_rows_for_language(lang, total):
    repo_id = os.environ.get("HF_REPO_ID", "ai4bharat/MSMARCO-XI")
    split_name = os.environ.get("HF_SPLIT", "validation")
    
    lang_map = {
        "as": "asm", "bn": "ben", "gu": "guj", "hi": "hin", "kn": "kan",
        "ml": "mal", "mr": "mar", "ne": "nep", "or": "ori", "pa": "pan",
        "sa": "san", "ta": "tam", "te": "tel", "ur": "urd"
    }
    lang_prefix = lang_map.get(lang, lang)
    split_suffix = "val" if split_name == "validation" else "train"
    file_name = f"{split_name}/{lang_prefix}{split_suffix}.parquet"
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{file_name}"
    
    # Open using fsspec with readahead caching and 1MB block size to prevent range request hangs
    f = fsspec.open(url, "rb", cache_type="readahead", block_size=1024*1024).open()
    pf = pq.ParquetFile(f)
        
    rows = []
    # Fetch only query_id and passages columns to speed up download
    batch_iter = pf.iter_batches(batch_size=50, columns=["query_id", "passages"])
    while len(rows) < total:
        try:
            batch = next(batch_iter)
            df_batch = batch.to_pandas()
            if df_batch.empty:
                break
            for r in df_batch.to_dict(orient="records"):
                r["lang_code"] = lang
                rows.append(r)
        except StopIteration:
            break
            
    return rows[:total]

def fetch_rows_for_language_safe(lang, total, max_retries=3):
    for attempt in range(max_retries):
        try:
            rows = fetch_rows_for_language(lang, total=total)
            if len(rows) == 0:
                print(f"[{lang}] WARNING: fetch succeeded but returned 0 rows — check if this config/split combination is valid")
            return rows
        except Exception as e:
            wait = 2 ** attempt
            print(f"[{lang}] attempt {attempt+1} failed: {e} — retrying in {wait}s")
            time.sleep(wait)
    print(f"[{lang}] FAILED after {max_retries} attempts — skipping this language")
    return []

def flatten_passages(rows):
    passages_key = os.environ.get("HF_PASSAGES_KEY", "Translated_passages")
    documents = []
    for index, row in enumerate(rows):
        query_id = str(row.get("query_id", index))
        lang_code = row.get("lang_code")
        passages = row.get("passages", {})
        
        target_passages = passages.get(passages_key)
        if target_passages is None:
            target_passages = passages.get("Translated_passages")
        if target_passages is None:
            target_passages = passages.get("English_passages")
            
        if target_passages is None:
            target_passages = []
        elif isinstance(target_passages, (list, np.ndarray)):
            if isinstance(target_passages, np.ndarray):
                target_passages = target_passages.tolist()
            else:
                target_passages = list(target_passages)
        else:
            target_passages = [target_passages]
            
        for p_idx, passage in enumerate(target_passages):
            if not passage or not passage.strip():
                continue
            documents.append({
                "query_id": query_id,
                "passage_id": f"{query_id}_{p_idx}",
                "text": passage,
                "language": lang_code
            })
    return documents

def index_documents(documents, lang):
    global chroma_collection, model
    
    chunks_to_add = []
    for doc in documents:
        passage = doc["text"]
        source_passage_id = doc["passage_id"]
        
        # Fixed
        fixed_chunks = chunk_fixed_size(passage)
        for c_idx, chunk in enumerate(fixed_chunks):
            chunk_id = f"fixed_{source_passage_id}_{c_idx}"
            chunks_to_add.append({
                "id": chunk_id,
                "text": chunk["text"],
                "metadata": {
                    "chunk_id": chunk_id,
                    "strategy": "fixed",
                    "source_passage_id": source_passage_id,
                    "token_count": int(chunk["token_count"]),
                    "language": lang
                }
            })
            
        # Semantic
        semantic_chunks = chunk_semantic(passage)
        for c_idx, chunk in enumerate(semantic_chunks):
            chunk_id = f"semantic_{source_passage_id}_{c_idx}"
            chunks_to_add.append({
                "id": chunk_id,
                "text": chunk["text"],
                "metadata": {
                    "chunk_id": chunk_id,
                    "strategy": "semantic",
                    "source_passage_id": source_passage_id,
                    "token_count": int(chunk["token_count"]),
                    "language": lang
                }
            })
            
        # Boundary
        boundary_chunks = chunk_boundary_aware(passage)
        for c_idx, chunk in enumerate(boundary_chunks):
            chunk_id = f"boundary_{source_passage_id}_{c_idx}"
            chunks_to_add.append({
                "id": chunk_id,
                "text": chunk["text"],
                "metadata": {
                    "chunk_id": chunk_id,
                    "strategy": "boundary",
                    "source_passage_id": source_passage_id,
                    "token_count": int(chunk["token_count"]),
                    "language": lang
                }
            })
            
    if not chunks_to_add:
        return 0
        
    ids = [c["id"] for c in chunks_to_add]
    docs = [c["text"] for c in chunks_to_add]
    metadatas = [c["metadata"] for c in chunks_to_add]
    
    # Compute embeddings manually using the loaded model
    print(f"[{lang}] Computing embeddings for {len(docs)} chunks...")
    embeddings = model.encode(docs).tolist()
    
    batch_size = 500
    for i in range(0, len(chunks_to_add), batch_size):
        end_idx = min(i + batch_size, len(chunks_to_add))
        chroma_collection.add(
            embeddings=embeddings[i:end_idx],
            documents=docs[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )
        
    return len(chunks_to_add)

def main():
    global model, tokenizer, chroma_client, chroma_collection
    
    # Initialize Chroma client ONCE and reuse it
    chroma_dir = os.environ.get("CHROMA_DB_DIR", "d:/RAG/backend/chroma_db")
    collection_name = os.environ.get("CHROMA_COLLECTION_NAME", "msmarco_rag")
    chroma_client = chromadb.PersistentClient(path=chroma_dir)
    
    print("Step 1: Running initial diagnostic sweep...")
    under_indexed, initial_counts = run_diagnostic()
    
    if not under_indexed:
        print("\nAll languages are fully indexed! No repair needed.")
        sys.exit(0)
        
    print(f"\nFound {len(under_indexed)} under-indexed languages: {under_indexed}")
    
    # Initialize SentenceTransformer model once
    print("\nInitializing SentenceTransformer model inline...")
    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    model = SentenceTransformer(model_name)
    tokenizer = model.tokenizer
    
    # Get existing collection directly
    chroma_collection = chroma_client.get_collection(name=collection_name)
    
    # Step 2: Repair each under-indexed language
    repaired_counts = {}
    for lang in under_indexed:
        print(f"\nRepairing {lang}...")
        rows = fetch_rows_for_language_safe(lang, total=200)  # slightly higher than before to clear the threshold
        if not rows:
            repaired_counts[lang] = 0
            continue
        documents = flatten_passages(rows)
        for doc in documents:
            doc["language"] = lang
            
        new_chunk_count = index_documents(documents, lang)
        repaired_counts[lang] = new_chunk_count
        print(f"[{lang}] indexed {new_chunk_count} new chunks")
        time.sleep(0.3)  # stay polite to the public API between languages
        
    print("\nStep 3: Running final diagnostic sweep after repairs...")
    remaining_under, final_counts = run_diagnostic()
    
    failed_repairs = [lang for lang in under_indexed if final_counts.get(lang, 0) < MIN_CHUNKS_THRESHOLD]
    
    if failed_repairs:
        print(f"\nWARNING: The following languages could not be repaired: {failed_repairs}")
        print("Please check Hugging Face dataset connection or config for these languages.")
    else:
        print("\nAll languages have been successfully indexed and repaired!")

if __name__ == "__main__":
    main()
