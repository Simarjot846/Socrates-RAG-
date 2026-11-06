# Fix: Restore Qdrant + Re-deploy for Interview

Your free Qdrant cluster (`carrer-companion`) was deleted due to inactivity.
Follow these steps to get `socrates-rag.vercel.app` fully working again.

---

## Step 1 — Create a new Qdrant free cluster (5 min)

1. Go to **[cloud.qdrant.io](https://cloud.qdrant.io)** and sign in.
2. Click **"Create cluster"** → choose **Free tier** → pick any region (US West is fastest for Render).
3. Once created, open the cluster and copy:
   - **Cluster URL** — looks like `https://xxxx-xxxx.us-west-2-0.aws.cloud.qdrant.io:6333`
   - **API Key** — generate one under the cluster's **API Keys** tab.

---

## Step 2 — Update your local `.env`

Open `backend/.env` and replace the old values:

```
QDRANT_URL=https://<YOUR-NEW-CLUSTER-URL>:6333
QDRANT_API_KEY=<YOUR-NEW-API-KEY>
```

---

## Step 3 — Re-index the data locally (15–30 min)

This script downloads the MSMARCO-XI dataset, chunks it 3 ways, embeds with
Cohere, and uploads everything to your new Qdrant cluster.

```powershell
cd backend
# activate your venv first
.\.venv\Scripts\Activate.ps1

# install deps if needed
pip install -r requirements.txt pandas pyarrow

# run the indexer
python reindex_cohere.py
```

Watch for `Done! Total chunks indexed: XXXXX` at the end.
This is a one-time step — the data lives in Qdrant from here on.

> **Tip:** `DATASET_SUBSET_LIMIT` in `.env` controls rows per language (currently 50).
> Keep it at 50 for a fast index (~5 min). Raise to 150 for a richer demo.

---

## Step 4 — Update Render environment variables

1. Go to **[dashboard.render.com](https://dashboard.render.com)** → your `rag-backend` service.
2. Click **Environment** → edit these two vars:
   | Key | Value |
   |-----|-------|
   | `QDRANT_URL` | your new cluster URL |
   | `QDRANT_API_KEY` | your new API key |
3. Click **Save Changes** — Render will auto-redeploy.

> **Important:** Also confirm these other secrets are set in Render's Environment tab:
> `GROQ_API_KEY`, `COHERE_API_KEY`, `SARVAM_API_KEY`, `HF_TOKEN`

---

## Step 5 — Verify the deployment

Once Render finishes deploying (watch the logs tab), open:

```
https://<your-render-service>.onrender.com/api/status
```

You should see:
```json
{"status": "ready", "db_count": 12345, ...}
```

Then open **[socrates-rag.vercel.app](https://socrates-rag.vercel.app)** — the
status badge in the header should show `Index: XXXXX Chunks`.

---

## What was fixed in this PR

| Problem | Fix |
|---------|-----|
| Qdrant cluster deleted | Create new cluster + re-index (this guide) |
| `GROQ_MODEL=openai/gpt-oss-20b` (invalid) | Changed to `llama-3.1-8b-instant` |
| Guard + Retrieve ran sequentially (~400 ms wasted) | Now run concurrently via `asyncio.gather` |
| Output guard checked 3 chunks, used 512 tokens | Now checks only top-1 chunk, `max_tokens=4` |
| Render free tier cold-starts (~30–60 s delay) | Self-ping loop every 10 min keeps it warm |
| Frontend hit cold backend on first query | Warm-up `fetch(/api/ping)` fires on app load |

---

## Expected latency after fix

| Stage | Before | After |
|-------|--------|-------|
| Embed + Guard-In | ~300 ms | ~250 ms |
| Retrieval | ~300 ms sequential | runs in parallel with guard |
| Generation | ~400 ms | ~300 ms (smaller context) |
| Guard-Out | ~300 ms | ~80 ms (1 chunk, 4 tokens) |
| **Total (text query)** | **~1300 ms** | **~650 ms** |

Cold starts are eliminated by the keep-alive loop, so the first demo query
responds in under 1 second.
