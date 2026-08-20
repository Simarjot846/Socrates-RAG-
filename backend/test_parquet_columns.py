import time
from fsspec.implementations.http import HTTPFileSystem
import pyarrow.parquet as pq

print("1. Initializing HTTPFileSystem...")
fs = HTTPFileSystem()

url = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/asmval.parquet"
print(f"2. Opening stream to: {url}")
t0 = time.time()
f = fs.open(url)
print(f"   Opened in {time.time() - t0:.2f}s.")

print("3. Parsing ParquetFile metadata...")
t0 = time.time()
pf = pq.ParquetFile(f)
print(f"   Parsed in {time.time() - t0:.2f}s. Row count: {pf.metadata.num_rows}")

print("4. Reading only passages column for the first 200 rows...")
t0 = time.time()
try:
    # We can use pf.iter_batches with columns and batch_size
    batch_iter = pf.iter_batches(batch_size=200, columns=["passages", "query_id"])
    batch = next(batch_iter)
    df = batch.to_pandas()
    print(f"   Success! Read {len(df)} rows in {time.time() - t0:.2f}s.")
    print("   First row query_id:", df.iloc[0].get("query_id", "N/A"))
    print("   First row passages structure:", list(df.iloc[0].get("passages", {}).keys()))
except Exception as e:
    print(f"   Failed to read columns: {e}")

print("Test completed.")
