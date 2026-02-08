from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import uuid

app = FastAPI(title="RAGFlowX")

UPLOAD_DIR = "data/raw"
CHUNK_DIR = "data/chunks"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHUNK_DIR, exist_ok=True)

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

@app.get("/read/{filename}")
def read_document(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return {
        "preview": text[:500],
        "length": len(text)
    }

# -------- NEW CODE BELOW -------- #

def chunk_text(text: str, chunk_size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks

@app.post("/chunk/{filename}")
def chunk_document(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)

    base_name = filename.replace(".txt", "")
    chunk_files = []

    for idx, chunk in enumerate(chunks):
        chunk_filename = f"{base_name}_chunk_{idx}.txt"
        chunk_path = os.path.join(CHUNK_DIR, chunk_filename)

        with open(chunk_path, "w", encoding="utf-8") as cf:
            cf.write(chunk)

        chunk_files.append(chunk_filename)

    return {
        "message": "Chunking completed",
        "total_chunks": len(chunk_files),
        "chunk_files": chunk_files
    }
