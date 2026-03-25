import os
import json
import pickle
import urllib.request
import numpy as np
from pathlib import Path
from typing import List, Dict
from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    print("Run: pip install sentence-transformers faiss-cpu")

# ── Config ────────────────────────────────────────────────────────────────────
DOCUMENTS_DIR   = Path(__file__).parent.parent / "documents"
DATA_DIR        = Path(__file__).parent.parent / "data"
INDEX_PATH      = DATA_DIR / "faiss.index"
CHUNKS_PATH     = DATA_DIR / "chunks.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K           = 3
CHUNK_SIZE      = 400
CHUNK_OVERLAP   = 80

# ── Your OpenRouter Key ───────────────────────────────────────────────────────
OPENROUTER_API_KEY = "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"

# ── Auto-fallback: tries each model until one works ───────────────────────────
FREE_MODELS = [
    "openrouter/free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-4b-it:free",
    "nvidia/llama-3.1-nemotron-nano-8b-v1:free",
    "qwen/qwen3-8b:free",
]

app = Flask(__name__)
CORS(app)

_model  = None
_index  = None
_chunks: List[Dict] = []


# ── 1. Document Processing ────────────────────────────────────────────────────

def chunk_text(text: str, source: str) -> List[Dict]:
    chunks, start = [], 0
    while start < len(text):
        end   = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 50:
            chunks.append({"text": chunk, "source": source})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def load_documents() -> List[Dict]:
    all_chunks = []
    for path in sorted(DOCUMENTS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        all_chunks.extend(chunk_text(text, path.name))
    print(f"[Loader] {len(all_chunks)} chunks from {DOCUMENTS_DIR}")
    return all_chunks


# ── 2. Vector Indexing ────────────────────────────────────────────────────────

def build_index(chunks: List[Dict], model) -> object:
    texts      = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"[Index] Built FAISS index with {index.ntotal} vectors")
    return index


def load_or_build_index():
    model = SentenceTransformer(EMBEDDING_MODEL)
    if INDEX_PATH.exists() and CHUNKS_PATH.exists():
        index  = faiss.read_index(str(INDEX_PATH))
        chunks = pickle.loads(CHUNKS_PATH.read_bytes())
        print(f"[Index] Loaded existing index ({index.ntotal} vectors)")
    else:
        chunks = load_documents()
        index  = build_index(chunks, model)
    return index, chunks, model


# ── 3. Retrieval ──────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = TOP_K) -> List[Dict]:
    q_emb = _model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = _index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append({**_chunks[idx], "score": float(score)})
    return results


# ── 4. Answer Generation (auto-fallback across free models) ──────────────────

def call_openrouter(model: str, system_msg: str, user_msg: str) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer":  "http://localhost:8000",
            "X-Title":       "Indecimal RAG",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]


def generate_answer(query: str, context_chunks: List[Dict]) -> str:
    context_text = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )

    system_msg = (
        "You are a helpful AI assistant for Indecimal, a home construction company. "
        "Answer ONLY using the provided document context. "
        "Do NOT use outside knowledge. "
        "If the answer is not in the context, say: 'I could not find this in the provided documents.' "
        "Be concise, friendly, and mention specific prices or brands from the documents when available."
    )
    user_msg = (
        f"RETRIEVED CONTEXT:\n{context_text}\n\n"
        f"QUESTION: {query}\n\n"
        f"Answer based only on the context above:"
    )

    for model in FREE_MODELS:
        try:
            print(f"[OpenRouter] Trying: {model}")
            answer = call_openrouter(model, system_msg, user_msg)
            print(f"[OpenRouter] Success: {model}")
            return answer
        except Exception as e:
            print(f"[OpenRouter] {model} failed: {e}")
            continue

    return "All free models are currently unavailable. Please try again in a minute."


# ── 5. Flask Routes ───────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "chunks_indexed": len(_chunks)})


@app.route("/query", methods=["POST"])
def query():
    data     = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400
    chunks = retrieve(question)
    answer = generate_answer(question, chunks)
    return jsonify({"answer": answer, "retrieved_chunks": chunks})


@app.route("/reindex", methods=["POST"])
def reindex():
    global _index, _chunks
    doc_chunks = load_documents()
    _index  = build_index(doc_chunks, _model)
    _chunks = doc_chunks
    return jsonify({"status": "reindexed", "chunks": len(_chunks)})


# ── Startup ───────────────────────────────────────────────────────────────────

def init():
    global _model, _index, _chunks
    _index, _chunks, _model = load_or_build_index()


if __name__ == "__main__":
    init()
    port = int(os.getenv("PORT", 8000))
    print(f"\n  Indecimal RAG API running on http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)