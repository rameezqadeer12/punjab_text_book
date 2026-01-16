import os
import pickle
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama

# ============================================================
# CONFIG
# ============================================================
INDEX_PATH  = os.getenv("INDEX_PATH", "index.faiss")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "chunks.pkl")
MODEL_PATH  = os.getenv("MODEL_PATH", "models/model.gguf")

def get_api_key():
    return os.getenv("API_KEY", "").strip()

# ============================================================
# GLOBALS (LAZY LOADING)
# ============================================================
index = None
chunks = None
embedder = None
llm = None

# ============================================================
# LOADERS
# ============================================================
def load_resources():
    global index, chunks, embedder, llm

    try:
        if embedder is None:
            embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

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

        if llm is None:
            if not os.path.exists(MODEL_PATH):
                raise RuntimeError(f"Missing model file: {MODEL_PATH}")
            llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=2048,
                n_threads=4,
                n_batch=128,
                verbose=False
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resource loading failed: {str(e)}")

# ============================================================
# RETRIEVER
# ============================================================
def retrieve(query, k=15, final_k=4):
    q_emb = embedder.encode([query])
    q_emb = np.asarray(q_emb, dtype=np.float32)

    D, I = index.search(q_emb, k)

    ranked = []
    for idx in I[0]:
        if idx < 0:
            continue
        b = chunks[int(idx)]
        text = b.get("text", "") if isinstance(b, dict) else str(b)
        score = 0
        if "theorem" in text.lower(): score += 2
        if "example" in text.lower(): score += 1
        if len(re.findall(r"\d", text)) > 10: score += 1
        ranked.append((score, b))

    ranked.sort(key=lambda x: -x[0])
    return [b for _, b in ranked[:final_k]]

# ============================================================
# PROMPT BUILDER
# ============================================================
def build_exam_prompt(question, ctx_text):
    return f"""
You are a senior mathematics examiner.

Below is official Punjab board textbook content.
Use it ONLY to identify the law or concept.

TEXTBOOK:
----------------
{ctx_text}
----------------

Question:
{question}

Answer strictly in exam style using this structure:

1. Statement of law
2. Given / Assumptions
3. Step-by-step solution
4. Verification / proof
5. Final conclusion

Rules:
- Do not copy matrices from the textbook
- Choose your own valid matrices
- Show full working
- Use clean mathematical formatting
- If the law is not found, say: "Not found in the given textbook."

ANSWER:
"""

# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(title="Punjab Board Hybrid Exam Solver API")

class Question(BaseModel):
    question: str
    k: int = 10

@app.get("/")
def root():
    return {"message": "Punjab Hybrid Exam Solver API is running"}

@app.get("/health")
def health():
    return {"status": "ok", "auth_enabled": bool(get_api_key())}

@app.post("/ask")
def ask_api(data: Question, x_api_key: str = Header(default=None, alias="x-api-key")):

    # ---------- AUTH ----------
    api_key = get_api_key()
    if api_key:
        if x_api_key is None:
            raise HTTPException(status_code=401, detail="API key missing")
        if x_api_key.strip() != api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

    # ---------- VALIDATION ----------
    q = (data.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question is required")

    k = max(5, min(int(data.k), 30))

    # ---------- LOAD RESOURCES ----------
    load_resources()

    # ---------- RETRIEVE ----------
    ctx_blocks = retrieve(q, k=k, final_k=4)
    if not ctx_blocks:
        return {"answer": "Not found in the given textbook."}

    book_text = "\n\n".join(
        (b.get("text", "") if isinstance(b, dict) else str(b))[:600]
        for b in ctx_blocks
    )

    if len(book_text.strip()) < 50:
        return {"answer": "Not found in the given textbook."}

    # ---------- GENERATE ----------
    prompt = build_exam_prompt(q, book_text)
    out = llm(prompt, max_tokens=700, temperature=0.2)
    answer = out["choices"][0]["text"].strip()

    return {
        "answer": answer,
        "source": "Punjab Board Textbooks (Hybrid RAG + LLaMA)"
    }
