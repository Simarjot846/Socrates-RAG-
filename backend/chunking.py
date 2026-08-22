import os
import re
import time
import requests
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

def split_into_sentences(text: str) -> list[str]:
    # Robust sentence splitter handling standard punctuation and Hindi danda (।)
    sentences = re.split(r'(?<=[.!?|।])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def fetch_rows_with_retry(repo_id, config, split, offset, length, max_retries=3):
    url = "https://datasets-server.huggingface.co/rows"
    params = {
        "dataset": repo_id,
        "config": config,
        "split": split,
        "offset": offset,
        "length": length,
    }
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json().get("rows", [])
        except Exception as e:
            if attempt == max_retries:
                print(f"Failed to fetch rows at offset {offset} after {max_retries} retries: {e}")
                return []
            wait_time = 2 ** attempt
            print(f"Error fetching rows at offset {offset}: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
    return []

def main():
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
    print("Initializing embedding model...")
    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    model = SentenceTransformer(model_name)
    tokenizer = model.tokenizer

    def get_token_count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    # 1. Chunking Strategy 1: Fixed-size with overlap
    def chunk_fixed_size(text: str, chunk_size=256, overlap=50) -> list[dict]:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        if not tokens:
            return chunks
        
        # Enforce sanity check on overlap
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

    # 2. Chunking Strategy 2: Semantic chunking
    def chunk_semantic(text: str, threshold=0.6) -> list[dict]:
        sentences = split_into_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [{"text": sentences[0], "token_count": get_token_count(sentences[0])}]
        
        # Embed sentences
        embeddings = model.encode(sentences)
        
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(len(sentences) - 1):
            emb_curr = embeddings[i]
            emb_next = embeddings[i+1]
            
            # Cosine similarity
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

    # 3. Chunking Strategy 3: Passage/sentence-boundary aware chunking
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

    # Fetch parameters from environment
    repo_id = os.environ.get("HF_REPO_ID", "ai4bharat/MSMARCO-XI")
    split_name = os.environ.get("HF_SPLIT", "validation")
    passages_key = os.environ.get("HF_PASSAGES_KEY", "Translated_passages")
    collection_name = os.environ.get("CHROMA_COLLECTION_NAME", "msmarco_rag")
    limit_per_lang = int(os.environ.get("DATASET_SUBSET_LIMIT", "150"))

    languages_env = os.environ.get("LANGUAGES", "hi,ta,bn")
    LANGUAGES = [lang.strip() for lang in languages_env.split(",") if lang.strip()]
    lang_map = {
        "as": "asm", "bn": "ben", "gu": "guj", "hi": "hin", "kn": "kan",
        "ml": "mal", "mr": "mar", "ne": "nep", "or": "ori", "pa": "pan",
        "sa": "san", "ta": "tam", "te": "tel", "ur": "urd"
    }

    import pyarrow.parquet as pq
    all_rows = []

    for lang in LANGUAGES:
        lang_prefix = lang_map.get(lang, lang)
        split_suffix = "val" if split_name == "validation" else "train"
        file_name = f"{split_name}/{lang_prefix}{split_suffix}.parquet"
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{file_name}"
        
        temp_dest = os.path.join(os.path.dirname(__file__), f"temp_{lang}.parquet")
        print(f"Streaming dataset for {lang} via local download from Hugging Face URL: {url}")
        
        max_retries = 3
        success = False
        for attempt in range(max_retries + 1):
            try:
                t0 = time.time()
                r = requests.get(url, stream=True, timeout=60)
                r.raise_for_status()
                with open(temp_dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                print(f"Downloaded {lang} Parquet in {time.time() - t0:.2f}s.")
                success = True
                break
            except Exception as e:
                if attempt == max_retries:
                    print(f"Failed to download Parquet for {lang}: {e}")
                    break
                wait_time = 2 ** attempt
                print(f"Error downloading Parquet for {lang}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

        if not success or not os.path.exists(temp_dest):
            continue

        try:
            print(f"Reading up to {limit_per_lang} rows from local parquet...")
            df = pd.read_parquet(temp_dest, columns=["query_id", "passages"])
            rows_list = df.head(limit_per_lang).to_dict(orient="records")
            for r in rows_list:
                r["lang_code"] = lang
            all_rows.extend(rows_list)
            print(f"Successfully loaded {len(rows_list)} rows for {lang}.")
        except Exception as e:
            print(f"Error reading local parquet for {lang}: {e}")
        finally:
            if os.path.exists(temp_dest):
                try:
                    os.remove(temp_dest)
                except Exception:
                    pass

    print(f"Successfully streamed {len(all_rows)} total rows across all languages.")

    # Initialize Qdrant Client
    print("Initializing Qdrant Client...")
    qdrant_url = os.environ.get("QDRANT_URL", "")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY", "")
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # Re-create/get the collection
    if os.environ.get("CHROMA_CLEAR", "true").lower() == "true":
        try:
            qdrant_client.delete_collection(collection_name)
            print(f"Cleared existing collection: {collection_name}")
        except Exception:
            pass

    try:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            )
        )
        print(f"Created collection: {collection_name}")
    except Exception as e:
        print(f"Collection creation skipped: {e}")

    print("Indexing passages...")

    total_passages_processed = 0
    total_chunks_added = 0
    
    # Accumulate inputs for batch insertion to optimize write speed
    documents = []
    metadatas = []
    ids = []

    # Store stats of chunk counts per language
    lang_chunk_counts = {lang: 0 for lang in LANGUAGES}

    for index, row in enumerate(all_rows):
        query_id = str(row.get("query_id", index))
        lang = row.get("lang_code", "hi")
        passages = row.get("passages", {})
        
        # Default to passages_key, fall back to "Translated_passages" or "English_passages"
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

        # If no target passages, skip
        if len(target_passages) == 0:
            continue
            
        for p_idx, passage in enumerate(target_passages):
            if not passage or not passage.strip():
                continue
                
            source_passage_id = f"{query_id}_{p_idx}"
            
            # Apply fixed-size strategy
            fixed_chunks = chunk_fixed_size(passage)
            for c_idx, chunk in enumerate(fixed_chunks):
                chunk_id = f"fixed_{lang}_{source_passage_id}_{c_idx}"
                documents.append(chunk["text"])
                ids.append(chunk_id)
                metadatas.append({
                    "chunk_id": chunk_id,
                    "strategy": "fixed",
                    "source_passage_id": source_passage_id,
                    "token_count": int(chunk["token_count"]),
                    "language": lang
                })
                lang_chunk_counts[lang] += 1
                
            # Apply semantic strategy
            semantic_chunks = chunk_semantic(passage)
            for c_idx, chunk in enumerate(semantic_chunks):
                chunk_id = f"semantic_{lang}_{source_passage_id}_{c_idx}"
                documents.append(chunk["text"])
                ids.append(chunk_id)
                metadatas.append({
                    "chunk_id": chunk_id,
                    "strategy": "semantic",
                    "source_passage_id": source_passage_id,
                    "token_count": int(chunk["token_count"]),
                    "language": lang
                })
                lang_chunk_counts[lang] += 1
                
            # Apply boundary aware strategy
            boundary_chunks = chunk_boundary_aware(passage)
            for c_idx, chunk in enumerate(boundary_chunks):
                chunk_id = f"boundary_{lang}_{source_passage_id}_{c_idx}"
                documents.append(chunk["text"])
                ids.append(chunk_id)
                metadatas.append({
                    "chunk_id": chunk_id,
                    "strategy": "boundary",
                    "source_passage_id": source_passage_id,
                    "token_count": int(chunk["token_count"]),
                    "language": lang
                })
                lang_chunk_counts[lang] += 1
                
            total_passages_processed += 1

    # Add to Qdrant in batches
    batch_size = 256
    print(f"Adding {len(documents)} chunks to Qdrant in batches of {batch_size}...")
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        batch_docs = documents[i:end_idx]
        batch_metas = metadatas[i:end_idx]
        batch_ids = ids[i:end_idx]
        
        # 1. Generate embeddings using local SentenceTransformer
        batch_vectors = model.encode(batch_docs, show_progress_bar=False).tolist()
        
        # 2. Prepare points for Qdrant
        points = []
        import uuid
        for idx in range(len(batch_ids)):
            payload = batch_metas[idx].copy()
            payload["text"] = batch_docs[idx]
            
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, batch_ids[idx]))
            
            points.append(models.PointStruct(
                id=point_uuid,
                vector=batch_vectors[idx],
                payload=payload
            ))
            
        # 3. Upload to Qdrant
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
        total_chunks_added += len(points)
        print(f"Uploaded {total_chunks_added}/{len(documents)} chunks...")

    print("\n" + "="*50)
    print("INDEXING STATS PER LANGUAGE:")
    print("="*50)
    for lang, count in lang_chunk_counts.items():
        print(f"  {lang:<10} : {count} chunks")
    print("="*50)

    print("Indexing completed successfully!")
    print(f"Total source passages processed: {total_passages_processed}")
    print(f"Total chunks indexed: {total_chunks_added}")
    try:
        print(f"Collection count: {qdrant_client.get_collection(collection_name).points_count}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
