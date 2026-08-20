import os
import sys
import time
import chromadb
from sentence_transformers import SentenceTransformer

print("1. Loading environment variables...")
chroma_dir = os.environ.get("CHROMA_DB_DIR", "d:/RAG/backend/chroma_db")
collection_name = os.environ.get("CHROMA_COLLECTION_NAME", "msmarco_rag")
print(f"   Chroma dir: {chroma_dir}, collection: {collection_name}")

print("2. Initializing chromadb.PersistentClient...")
chroma_client = chromadb.PersistentClient(path=chroma_dir)
print("   Client initialized.")

print("3. Getting collection first time...")
collection = chroma_client.get_collection(name=collection_name)
print(f"   Collection retrieved. Count: {collection.count()}")

print("4. Initializing SentenceTransformer model...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("   Model initialized.")

print("5. Getting collection second time...")
collection2 = chroma_client.get_collection(name=collection_name)
print("   Collection retrieved second time.")

print("6. Fetching test Parquet URL from Hugging Face...")
from fsspec.implementations.http import HTTPFileSystem
import pyarrow.parquet as pq
url = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/asmval.parquet"
fs = HTTPFileSystem()
f = fs.open(url)
pf = pq.ParquetFile(f)
print("   Parquet file opened successfully.")

print("All tests completed successfully!")
