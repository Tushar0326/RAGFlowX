from fileinput import filename
from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import uuid
import json
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
from app.events import inngest, DOCUMENT_UPLOADED
import workflows.ingestion 



load_dotenv()
client = OpenAI()





app = FastAPI(title="RAGFlowX")

UPLOAD_DIR = "data/raw"
CHUNK_DIR = "data/chunks"
EMBED_DIR = "data/embeddings"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHUNK_DIR, exist_ok=True)
os.makedirs(EMBED_DIR, exist_ok=True)

# Load embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.get("/")
def root():
    return {"message": "RAGFlowX is running"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files supported")

    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    return {"filename": filename}

@app.post("/chunk/{filename}")
def chunk_document(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    def chunk_text(text, size=500, overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks

    chunks = chunk_text(text)

    base = filename.replace(".txt", "")
    chunk_files = []

    for i, chunk in enumerate(chunks):
        name = f"{base}_chunk_{i}.txt"
        path = os.path.join(CHUNK_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(chunk)
        chunk_files.append(name)

    return {"chunks": chunk_files}

# -------- NEW: EMBEDDINGS -------- #

@app.post("/embed/{filename}")
def embed_document(filename: str):
    base = filename.replace(".txt", "")
    embeddings = []

    for file in os.listdir(CHUNK_DIR):
        if file.startswith(base):
            with open(os.path.join(CHUNK_DIR, file), "r", encoding="utf-8") as f:
                text = f.read()

            vector = model.encode(text).tolist()

            embeddings.append({
                "chunk_file": file,
                "embedding": vector
            })

    if not embeddings:
        raise HTTPException(status_code=400, detail="No chunks found. Run chunk first.")

    save_path = os.path.join(EMBED_DIR, f"{base}_embeddings.json")
    with open(save_path, "w") as f:
        json.dump(embeddings, f)

    return {
        "message": "Embeddings created",
        "total_chunks": len(embeddings),
        "file": save_path
    }

@app.post("/ask")
def ask_question(query: str, filename: str, top_k: int = 3):
    base = filename.replace(".txt", "")
    embedding_file = os.path.join(EMBED_DIR, f"{base}_embeddings.json")

    if not os.path.exists(embedding_file):
        raise HTTPException(status_code=400, detail="Run embed first")

    with open(embedding_file, "r") as f:
        stored_embeddings = json.load(f)

    query_vector = model.encode(query).tolist()

    scores = []
    for item in stored_embeddings:
        score = cosine_similarity(query_vector, item["embedding"])
        scores.append((item["chunk_file"], score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

    context = "\n\n".join([load_chunk_text(f) for f, _ in scores])

    prompt = f"""
You are a helpful assistant.
Answer the question strictly using the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return {
        "question": query,
        "answer": response.choices[0].message.content,
        "sources": [f for f, _ in scores]
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    ...
    return {"filename": filename}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files supported")

    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # 🔔 Emit Inngest event
    await inngest.send(
        {
            "name": DOCUMENT_UPLOADED,
            "data": {
                "filename": filename,
                "path": file_path
            }
        }
    )

    return {
        "message": "File uploaded",
        "filename": filename,
        "event": DOCUMENT_UPLOADED
    }
