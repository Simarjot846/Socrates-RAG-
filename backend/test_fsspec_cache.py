import time
import fsspec
import pyarrow.parquet as pq

url = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/asmval.parquet"

print("1. Opening url with fsspec blockcache...")
t0 = time.time()
try:
    # Use fsspec.open with caching options
    f = fsspec.open(url, "rb", cache_type="readahead", block_size=1024*1024).open()
    print(f"   Opened in {time.time() - t0:.2f}s.")
    
    print("2. Parsing ParquetFile metadata...")
    t0 = time.time()
    pf = pq.ParquetFile(f)
    print(f"   Parsed in {time.time() - t0:.2f}s. Row count: {pf.metadata.num_rows}")
    
    print("3. Reading first batch using iter_batches...")
    t0 = time.time()
    batch_iter = pf.iter_batches(batch_size=5, columns=["query_id"])
    batch = next(batch_iter)
    df = batch.to_pandas()
    print(f"   Success! Read first batch in {time.time() - t0:.2f}s.")
    print("   Query ID:", df.iloc[0].get("query_id"))
except Exception as e:
    print(f"   Failed: {e}")

print("Test completed.")
