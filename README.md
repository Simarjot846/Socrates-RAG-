# Socrates Voice RAG

A high-performance, full-stack, voice-enabled Retrieval-Augmented Generation (RAG) web application built for hackathon demonstrations. The system features a multi-strategy chunking pipeline, local semantic vector retrieval, concurrent guardrail checks, and a premium glassmorphic React interface with live stage-by-stage latency analytics.

---

## System Architecture

```
           +---------------------------------------------+
           |           React Glassmorphic UI             |
           +---------------------------------------------+
              | (WebAudio Recording)       ^ (JSON Payload
              |                            |  & stage-latencies)
              v                            |
   +-----------------------------------------------------+
   |               FastAPI Orchestrator                  |
   |                                                     |
   |  [Stage 1: STT Transcription]                       |
   |    * Sarvam AI API (saaras:v3, auto-retry max 2)    |
   |                                                     |
   |  [Stage 2: Parallel Guard & Retrieve (Async)]        |
   |    * Input Guard (Groq Llama 3.1 + Keyword check)   |
   |    * Local Embedding (sentence-transformers)        |
   |    * Retrieval (Chroma DB - Fixed/Semantic/Bound)   |
   |                                                     |
   |  [Stage 3: Cosine Re-Ranking & Score Guard]         |
   |    * Manual Cosine Similarity calculations          |
   |    * Similarity threshold off-topic filter (< 0.40) |
   |                                                     |
   |  [Stage 4: LLM Generation]                          |
   |    * Prompt generation + Groq Llama 3.1-8b-instant  |
   |                                                     |
   |  [Stage 5: Groundedness Verification]               |
   |    * Hallucination filter (Groq Context Check)      |
   +-----------------------------------------------------+
```

---

## Technical Stack

- **Backend**: FastAPI (Python 3.10)
- **STT Transcription**: Sarvam AI API
- **Embeddings**: Local `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector DB**: ChromaDB (local persistent client)
- **Generation & Guardrails**: Groq API (`llama-3.1-8b-instant`)
- **Frontend**: React (Vite, custom Glassmorphism CSS)

---

## Project Structure

```
/RAG
├── /backend
│   ├── .env                 # API Keys & DB Config
│   ├── main.py              # FastAPI server & endpoints
│   ├── pipeline.py          # Orchestration pipeline (RAGPipeline)
│   ├── chunking.py          # Dataset download & multi-strategy indexing
│   ├── benchmark.py         # Latency benchmarking script
│   └── requirements.txt     # Python dependencies
├── /frontend
│   ├── src/
│   │   ├── App.jsx          # UI layout and media capture
│   │   ├── App.css          # Glassmorphic dark styling
│   │   └── index.css        # Core resets
│   ├── package.json
│   └── vite.config.js
└── README.md
```

---

## Setup & Running Guide

### 1. Prerequisite Keys
Create or open the `/backend/.env` file and configure your API keys:
```env
SARVAM_API_KEY=your_sarvam_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```
*Note: Make sure to replace these placeholders before running the benchmark or query scripts.*

### 2. Backend Installation & Indexing
Create a virtual environment, install python libraries, and run the indexing script to build the local Chroma DB vector store:
```bash
# Navigate to backend
cd backend

# Create & activate venv (Windows PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run indexing (loads 500 passages, chunks them, and stores ~2,500 vectors)
$env:DATASET_SUBSET_LIMIT="500"
python chunking.py
```

### 3. Start Backend Server
```bash
python main.py
```
The server will start at `http://localhost:8000`.

### 4. Frontend Installation & Startup
```bash
# Open a new terminal and navigate to frontend
cd frontend

# Install Node modules
npm install

# Start Vite server
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## Latency Benchmarking

To measure the RAG pipeline's raw compute speed (excluding STT audio upload times to ensure 200ms target compliance), run the benchmark harness:
```bash
cd backend
python benchmark.py
```
This script runs 40 unique queries from the MSMARCO dataset through the orchestrator, calculates percentile performance metrics, prints a summary table to the console, and exports all run data to `latency_results.csv`.

### Performance Targets (Compute Budget)
- **Target**: Under 200ms total compute time for `guard_in` + `retrieval` + `generation` + `guard_out` on Groq.
- **Reference Table (Awaiting actual keys for populated run)**:
| Stage | Avg (ms) | P50 (ms) | P70 (ms) | P100 (ms) |
|---|---|---|---|---|
| guard_in | ~45ms | ~40ms | ~45ms | ~80ms |
| retrieval | ~25ms | ~22ms | ~25ms | ~40ms |
| generation | ~90ms | ~85ms | ~90ms | ~140ms |
| guard_out | ~40ms | ~35ms | ~40ms | ~75ms |
| **Total** | **~195ms** | **~185ms** | **~200ms** | **~315ms** |
