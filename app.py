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
INDEX_PATH = os.getenv("INDEX_PATH", "index.faiss")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "chunks.pkl")
API_KEY = os.getenv("API_KEY", "").strip()

# -----------------------------
# GLOBALS (LAZY LOADING)
# -----------------------------
index = None
chunks = None
embedder = None

def load_resources():
    """
    Load heavy resources ONLY when needed (Render FREE plan safe)
    """
    global index, chunks, embedder

    if embedder is None:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")

    if index is None:
        if not os.path.exists(INDEX_PATH):
            raise RuntimeError(f"Missing FAISS index file: {INDEX_PATH}")
        index = faiss.read_index(INDEX_PATH)

    if chunks is None:
        if not os.path.exists(CHUNKS_PATH):
            raise RuntimeError(f"Missing chunks file: {CHUNKS_PATH}")
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)

        if not isinstance(chunks, list) or len(chunks) == 0:
            raise RuntimeError("chunks.pkl is empty or invalid")

# -----------------------------
# API
# -----------------------------
app = FastAPI(title="Punjab Board Exam QA API")

class Question(BaseModel):
    question: str
    k: int = 5

@app.get("/health")
def health():
    """
    Lightweight health check (NO heavy loading)
    """
    return {
        "status": "ok",
        "auth_enabled": bool(API_KEY)
    }

@app.post("/ask")
def ask_api(
    data: Question,
    x_api_key: str = Header(default=None, alias="x-api-key")
):
    # -----------------------------
    # AUTH (FIXED)
    # -----------------------------
    if API_KEY:
        if x_api_key is None:
            raise HTTPException(
                status_code=401,
                detail="API key missing. Send it in 'x-api-key' header."
            )

        if x_api_key.strip() != API_KEY:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key"
            )

    # -----------------------------
    # VALIDATION
    # -----------------------------
    q = (data.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question is required")

    k = max(1, min(int(data.k), 20))

    # -----------------------------
    # LOAD HEAVY RESOURCES (ON DEMAND)
    # -----------------------------
    load_resources()

    # -----------------------------
    # SEARCH
    # -----------------------------
    q_emb = embedder.encode([q])
    q_emb = np.asarray(q_emb, dtype=np.float32)

    _, idx = index.search(q_emb, k)

    if idx.size == 0 or idx[0][0] < 0:
        return {
            "answer": "This question is not answered in the given textbook."
        }

    ctx = chunks[int(idx[0][0])]
    text = (ctx.get("text", "") if isinstance(ctx, dict) else str(ctx)).strip()
    book = ctx.get("book", "Unknown Book") if isinstance(ctx, dict) else "Unknown Book"

    if not text:
        return {
            "answer": "This question is not answered in the given textbook."
        }

    answer = text[:500] + ("..." if len(text) > 500 else "")

    return {
        "answer": f"• {answer}",
        "reference": f"Punjab Textbook Board | {book}"
    }
