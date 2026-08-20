import time
import requests

repo_id = "ai4bharat/MSMARCO-XI"
config = "asm"  # Assamese
split = "validation"
url = "https://datasets-server.huggingface.co/rows"

params = {
    "dataset": repo_id,
    "config": config,
    "split": split,
    "offset": 0,
    "length": 10
}

print(f"Querying HF Datasets Server: {url} with params {params}...")
t0 = time.time()
try:
    resp = requests.get(url, params=params, timeout=10)
    print(f"Status Code: {resp.status_code} in {time.time() - t0:.2f}s")
    if resp.status_code == 200:
        data = resp.json()
        rows = data.get("rows", [])
        print(f"Success! Fetched {len(rows)} rows.")
        if rows:
            print("First row query text snippet:", rows[0]["row"].get("query", "N/A")[:100])
    else:
        print("Response text:", resp.text)
except Exception as e:
    print("Request failed:", e)
