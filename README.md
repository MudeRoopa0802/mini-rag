# Indecimal AI Assistant — Mini RAG Pipeline

A Retrieval-Augmented Generation (RAG) based AI chatbot built for the **Indecimal construction marketplace**. It answers user questions strictly from internal documents using semantic search and an LLM — no hallucinations, no outside knowledge.


## What is RAG?

RAG (Retrieval-Augmented Generation) combines two steps:
1. **Retrieve** — find the most relevant document chunks for a query using vector search
2. **Generate** — pass those chunks to an LLM and instruct it to answer ONLY from them

This ensures answers are grounded in your actual documents, not the model's general knowledge.


## Architecture

User Question
      │
      ▼
[Embedding Model]  ←── all-MiniLM-L6-v2 (sentence-transformers, runs locally)
      │  Query vector (384-dim)
      ▼
[FAISS Index]      ←── Cosine similarity search → Top-3 most relevant chunks
      │  Retrieved context
      ▼
[LLM via OpenRouter] ←── Strictly grounded to retrieved chunks only
      │
      ▼
Answer + Retrieved Chunks displayed transparently in the UI


## Embedding Model & LLM — Why These Choices?

### Embedding: `all-MiniLM-L6-v2` (sentence-transformers)
- Runs **fully locally** — no API key or internet needed for embeddings
- Produces 384-dimensional dense vectors — compact yet highly accurate
- Fast encoding: processes all document chunks in seconds
- Industry-standard for semantic similarity tasks

### Vector Search: FAISS (`IndexFlatIP`)
- Local vector database — no managed service like Pinecone needed
- Uses **Inner Product on L2-normalized vectors = cosine similarity**
- Exact search (no approximation) — ideal for small-to-medium document sets
- Index is saved to disk and reloaded instantly on subsequent runs

### LLM: OpenRouter (free tier)
- Uses free models: `llama-3.1-8b`, `gemma-2-9b`, `qwen-2-7b`, and others
- Auto-fallback: if one model is unavailable, automatically tries the next
- No billing required — works with a free OpenRouter account
- Called directly via Python's built-in `urllib` — no extra packages needed



## Document Chunking & Retrieval

### Chunking Strategy
- Documents are split into **400-character overlapping chunks**
- **80-character overlap** between chunks prevents information loss at boundaries
- Chunks under 50 characters are discarded (too small to be meaningful)
- Each chunk stores its source filename for transparency

### Retrieval Process
1. User query is embedded using the same model as the documents
2. Query vector is L2-normalized
3. FAISS searches for top-3 chunks with highest cosine similarity
4. All 3 chunks are passed as context to the LLM

---

## Grounding — How Hallucinations Are Prevented

The LLM receives this strict system prompt:

> "Answer ONLY using the provided document context. Do NOT use outside knowledge. If the answer is not in the context, say: 'I could not find this in the provided documents.'"

This ensures:
- Every answer is sourced from retrieved document chunks
- Out-of-scope questions are explicitly refused
- No fabricated prices, brands, or policies

---

## Project Structure
```
mini-rag/
├── backend/
│   ├── rag_pipeline.py     # Core RAG pipeline + Flask REST API
│   └── evaluate.py         # Quality evaluation (12 test questions)
├── frontend/
│   └── index.html          # Custom Indecimal-branded chatbot UI
├── documents/
│   ├── doc1.txt            # Company Overview & Customer Journey
│   ├── doc2.txt            # Package Pricing & Specifications
│   └── doc3.txt            # Policies, Quality & Guarantees
├── data/                   # Auto-created: FAISS index + chunk cache
├── requirements.txt
└── README.md



## Knowledge Base Documents

| File | Content |
|------|---------|
| `doc1.txt` | Indecimal company overview, customer journey (10 stages), operating principles |
| `doc2.txt` | All 4 package prices, steel/cement/flooring/painting/door/window specs |
| `doc3.txt` | Escrow payment model, 445+ quality checks, zero-cost maintenance program |

---

## How to Run Locally

### 1. Install dependencies
```bash
pip install sentence-transformers faiss-cpu flask flask-cors numpy
```

### 2. Get a free OpenRouter API key
- Go to **openrouter.ai** → Sign up → Keys → Create Key
- Paste the key into `backend/rag_pipeline.py` at line 20:
```python
OPENROUTER_API_KEY = "sk-or-v1-your-key-here"
```

### 3. Start the backend
```bash
cd backend
python rag_pipeline.py
```

First run downloads the embedding model (~90MB). Subsequent runs load the cached FAISS index instantly.

### 4. Open the chatbot
Double-click `frontend/index.html` — opens directly in your browser. No build step needed.

---

## API Reference

### `POST /query`
```json
// Request
{ "question": "What are the package prices per sqft?" }

// Response
{
  "answer": "Indecimal offers 4 packages: Essential at Rs.1,851/sqft...",
  "retrieved_chunks": [
    { "text": "...", "source": "doc2.txt", "score": 0.91 }
  ]
}
```

### `GET /health`
```json
{ "status": "ok", "chunks_indexed": 35 }
```

### `POST /reindex`
Forces a full rebuild of the FAISS index from the documents folder.

---

## Transparency

Every response in the UI shows:
- **Retrieved Context** panel — expandable, shows source file + similarity score for each chunk
- **Final Answer** — generated strictly from the retrieved chunks



## Quality Evaluation

Run the evaluation script while the backend is running:
```bash
python backend/evaluate.py
```

Tests 12 questions derived from the documents and reports:

| Metric | Description |
|--------|-------------|
| **Relevance** | Keyword match between expected terms and the answer |
| **Groundedness** | Lexical overlap between answer and retrieved chunks (hallucination proxy) |
| **Completeness** | Length-based heuristic for response detail |

### Sample Test Questions
1. What does Indecimal promise to customers?
2. What are the stages of the Indecimal customer journey?
3. What are the package prices per sqft?
4. What steel brand is used in the Pinnacle package?
5. What cement is used in the Infinia package?
6. What is the main door wallet amount for the Premier package?
7. What flooring allowance does the Essential package offer?
8. What painting brand is used for exterior in the Pinnacle package?
9. How does Indecimal protect customer payments?
10. How many quality checkpoints does Indecimal have?
11. What does the zero cost maintenance program cover?
12. How does Indecimal ensure transparent pricing?


## Evaluation Findings

| Metric | Observed Result |
|--------|----------------|
| Chunk Relevance | High — FAISS cosine similarity consistently retrieves topically correct passages |
| Groundedness | Strong — LLM prompt strictly constrains to context |
| Hallucination Rate | Low — out-of-scope questions correctly return "not found in documents" |
| Latency | ~300ms retrieval + ~2s LLM = ~2.3s end-to-end |

**Key observations:**
- 400-character chunk size with 80-character overlap gives the best balance of precision and context
- Cosine similarity on normalized vectors outperforms raw L2 distance for semantic matching
- Auto-fallback across 5 free OpenRouter models ensures high availability at zero cost
- The strict grounding prompt is the most critical factor in preventing hallucinations


## Tech Stack Summary

| Component | Technology |
|-----------|-----------|
| Embedding | sentence-transformers `all-MiniLM-L6-v2` |
| Vector DB | FAISS `IndexFlatIP` (cosine similarity) |
| LLM | OpenRouter free tier (Llama, Gemma, Qwen) |
| Backend | Python + Flask + flask-cors |
| Frontend | Vanilla HTML/CSS/JS (no build step) |
| Language | Python 3.9+ |


## Requirements

- Python 3.9+
- Free OpenRouter account (openrouter.ai)
- ~200MB disk space for embedding model (downloaded once)
- Internet connection for LLM API calls only