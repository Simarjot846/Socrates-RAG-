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
print(f"   Parsed in {time.time() - t0:.2f}s. Row count: {pf.metadata.num_rows}, Row groups: {pf.num_row_groups}")

print("4. Reading row group 0...")
t0 = time.time()
try:
    table = pf.read_row_group(0)
    df = table.to_pandas()
    print(f"   Success! Read row group 0 with {len(df)} rows in {time.time() - t0:.2f}s.")
except Exception as e:
    print(f"   Failed to read row group: {e}")

print("Test completed.")
