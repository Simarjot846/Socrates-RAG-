import os
import time
import asyncio
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Import pipeline
from pipeline import RAGPipeline

async def main():
    print("Initializing RAG Pipeline for benchmarking...")
    pipeline = RAGPipeline()
    
    # Fetch parameters from environment
    repo_id = os.environ.get("HF_REPO_ID", "ai4bharat/MSMARCO-XI")
    config_name = os.environ.get("HF_CONFIG", "hi")
    split_name = os.environ.get("HF_SPLIT", "validation")
    query_column = os.environ.get("HF_QUERY_COLUMN", "Eng_Query")
    
    # Determine file name from CONFIG and SPLIT if HF_FILE_NAME is not set
    file_name = os.environ.get("HF_FILE_NAME", "")
    if not file_name:
        lang_map = {
            "as": "asm", "bn": "ben", "gu": "guj", "hi": "hin", "kn": "kan",
            "ml": "mal", "mr": "mar", "ne": "nep", "or": "ori", "pa": "pan",
            "sa": "san", "ta": "tam", "te": "tel", "ur": "urd"
        }
        lang_prefix = lang_map.get(config_name, config_name)
        split_suffix = "val" if split_name == "validation" else "train"
        file_name = f"{split_name}/{lang_prefix}{split_suffix}.parquet"

    # Streaming dataset via HTTP range requests
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{file_name}"
    print(f"Loading benchmark queries from dataset stream URL: {url}...")
    
    import requests
    import pyarrow.parquet as pq
    
    sample_queries = []
    temp_dest = os.path.join(os.path.dirname(__file__), "temp_benchmark.parquet")
    try:
        t0 = time.time()
        print(f"Downloading benchmark Parquet file locally...")
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(temp_dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded benchmark Parquet in {time.time() - t0:.2f}s.")
        
        df_batch = pd.read_parquet(temp_dest, columns=[query_column])
        raw_queries = df_batch[query_column].dropna().unique()
        sample_queries = [str(q) for q in raw_queries[:40]]
    except Exception as e:
        print(f"Error fetching benchmark queries from parquet: {e}. Falling back to dummy query list.")
        sample_queries = ["What is voice RAG?", "How does semantic chunking work?"]
    finally:
        if os.path.exists(temp_dest):
            try:
                os.remove(temp_dest)
            except Exception:
                pass
        
    print(f"Loaded {len(sample_queries)} queries for benchmarking.")
    
    # Lists to store latencies
    latencies = {
        "guard_in": [],
        "retrieval": [],
        "generation": [],
        "guard_out": [],
        "total": []
    }
    
    results = []
    
    print("\nStarting benchmark runs...")
    for idx, query in enumerate(sample_queries):
        safe_q = query[:60].encode('ascii', errors='ignore').decode('ascii')
        print(f"[{idx+1}/{len(sample_queries)}] Query: '{safe_q}...'")
        
        try:
            # Bypassing STT as per specifications
            res = await pipeline.run(text_query=query)
            
            # Extract latency breakdown
            l_ms = res.get("latency_ms", {})
            latencies["guard_in"].append(l_ms.get("guard_in", 0))
            latencies["retrieval"].append(l_ms.get("retrieval", 0))
            latencies["generation"].append(l_ms.get("generation", 0))
            latencies["guard_out"].append(l_ms.get("guard_out", 0))
            latencies["total"].append(l_ms.get("total", 0))
            
            results.append({
                "query": query,
                "answer": res.get("answer", ""),
                "flagged": res.get("flagged", False),
                "grounded": res.get("grounded", True),
                "guard_in_ms": l_ms.get("guard_in", 0),
                "retrieval_ms": l_ms.get("retrieval", 0),
                "generation_ms": l_ms.get("generation", 0),
                "guard_out_ms": l_ms.get("guard_out", 0),
                "total_ms": l_ms.get("total", 0)
            })
        except Exception as e:
            print(f"Query failed during benchmark: {e}")
        
        # Rate limit safety delay for Groq free-tier
        await asyncio.sleep(2.0)
            
    # Calculate statistics
    print("\nBenchmark complete! Calculating latency statistics...")
    
    stats = {}
    for stage, times in latencies.items():
        if times:
            stats[stage] = {
                "P50": float(np.percentile(times, 50)),
                "P70": float(np.percentile(times, 70)),
                "P100": float(np.percentile(times, 100)),
                "avg": float(np.mean(times))
            }
        else:
            stats[stage] = {"P50": 0.0, "P70": 0.0, "P100": 0.0, "avg": 0.0}
            
    # Print summary table to console
    print("\n" + "="*70)
    print(f"{'Stage':<18} | {'Avg (ms)':<10} | {'P50 (ms)':<10} | {'P70 (ms)':<10} | {'P100 (ms)':<10}")
    print("="*70)
    for stage, values in stats.items():
        print(f"{stage:<18} | {values['avg']:<10.2f} | {values['P50']:<10.2f} | {values['P70']:<10.2f} | {values['P100']:<10.2f}")
    print("="*70)
    
    # Save detailed query runs to CSV
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(os.path.dirname(__file__), "latency_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Detailed benchmark results saved to: {results_csv_path}")

if __name__ == "__main__":
    asyncio.run(main())
