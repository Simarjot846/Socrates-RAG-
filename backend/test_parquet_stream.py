import time
from fsspec.implementations.http import HTTPFileSystem
import pyarrow.parquet as pq

print("1. Initializing HTTPFileSystem...")
fs = HTTPFileSystem()

url = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/asmval.parquet"
print(f"2. Opening connection to: {url}")
t0 = time.time()
f = fs.open(url)
print(f"   Opened in {time.time() - t0:.2f}s.")

print("3. Parsing ParquetFile metadata...")
t0 = time.time()
pf = pq.ParquetFile(f)
print(f"   Parsed in {time.time() - t0:.2f}s. Row count: {pf.metadata.num_rows}")

print("4. Reading first batch of size 5...")
t0 = time.time()
batch_iter = pf.iter_batches(batch_size=5)
try:
    batch = next(batch_iter)
    df_batch = batch.to_pandas()
    print(f"   Success! Read {len(df_batch)} rows in {time.time() - t0:.2f}s.")
    print("   First row query:", df_batch.iloc[0].get("query", "N/A"))
except Exception as e:
    print(f"   Failed to read batch: {e}")

print("Test completed.")
