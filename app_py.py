import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# -----------------------------
# CONFIG
# -----------------------------
# Put index.faiss and chunks.pkl in the SAME folder as app.py (Render/local)
INDEX_PATH = os.getenv("INDEX_PATH", "index.faiss")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "chunks.pkl")

# API key: set this in Render "Environment Variables"
# If API_KEY is not set, endpoint will still work (no auth).
API_KEY = os.getenv("API_KEY", "").strip()

# -----------------------------
# LOAD FILES (FAIL FAST)
# -----------------------------
if not os.path.exists(INDEX_PATH):
    raise RuntimeError(
        f"Missing FAISS index file: {INDEX_PATH}\n"
        f"Fix: Ensure '{INDEX_PATH}' is deployed alongside app.py, "
        f"or set INDEX_PATH env var to the correct location."
    )

if not os.path.exists(CHUNKS_PATH):
    raise RuntimeError(
        f"Missing chunks pickle file: {CHUNKS_PATH}\n"
        f"Fix: Ensure '{CHUNKS_PATH}' is deployed alongside app.py, "
        f"or set CHUNKS_PATH env var to the correct location."
    )

index = faiss.read_index(INDEX_PATH)

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

if not isinstance(chunks, list) or len(chunks) == 0:
    raise RuntimeError("chunks.pkl loaded but is empty or not a list.")

# -----------------------------
# MODEL
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# API
# -----------------------------
app = FastAPI(title="Punjab Board Exam QA API")

class Question(BaseModel):
    question: str
    k: int = 5

@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_file": INDEX_PATH,
        "chunks": len(chunks),
        "auth_enabled": bool(API_KEY),
    }

@app.post("/ask")
def ask_api(data: Question, x_api_key: str = Header(default="")):
    # --- API KEY CHECK ---
    if API_KEY:
        if not x_api_key or x_api_key.strip() != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")

    q = (data.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question is required")

    k = int(data.k) if data.k else 5
    k = max(1, min(k, 20))  # safety cap

    # --- SEARCH ---
    q_emb = embedder.encode([q])
    q_emb = np.asarray(q_emb, dtype=np.float32)  # FAISS expects float32
    _, idx = index.search(q_emb, k)

    if idx.size == 0 or idx[0][0] < 0:
        return {"answer": "This question is not answered in the given textbook."}

    ctx = chunks[int(idx[0][0])]
    text = (ctx.get("text", "") if isinstance(ctx, dict) else str(ctx)).strip()
    book = ctx.get("book", "Unknown Book") if isinstance(ctx, dict) else "Unknown Book"

    if not text:
        return {"answer": "This question is not answered in the given textbook."}

    # limit answer length for API
    answer = text[:500] + ("..." if len(text) > 500 else "")

    return {
        "answer": f"• {answer}",
        "reference": f"Punjab Textbook Board | {book}",
    }

