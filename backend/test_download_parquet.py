import os
import time
import requests
import pandas as pd

url = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/asmval.parquet"
dest = "test_asmval.parquet"

print(f"1. Downloading {url} to {dest}...")
t0 = time.time()
try:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        f.write(resp.content)
    print(f"   Downloaded {os.path.getsize(dest)} bytes in {time.time() - t0:.2f}s.")
except Exception as e:
    print(f"   Download failed: {e}")

print("2. Reading local parquet file using pandas...")
t0 = time.time()
try:
    df = pd.read_parquet(dest)
    print(f"   Read {len(df)} rows in {time.time() - t0:.2f}s.")
    print("   Columns:", list(df.columns))
    print("   First row query:", df.iloc[0].get("query", "N/A"))
except Exception as e:
    print(f"   Failed to read local parquet: {e}")

# Clean up
if os.path.exists(dest):
    os.remove(dest)
    print("3. Cleaned up temporary file.")

print("Test completed.")
