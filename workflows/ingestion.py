from inngest import inngest_function
from app.events import inngest, DOCUMENT_UPLOADED
import os
import json
from sentence_transformers import SentenceTransformer

UPLOAD_DIR = "data/raw"
CHUNK_DIR = "data/chunks"
EMBED_DIR = "data/embeddings"

os.makedirs(CHUNK_DIR, exist_ok=True)
os.makedirs(EMBED_DIR, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_text(text, size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


@inngest_function(
    inngest=inngest,
    name="document.ingestion.pipeline",
    event=DOCUMENT_UPLOADED,
    retries=3,
)
def ingestion_pipeline(ctx):
    """
    Background pipeline:
    1. Read document
    2. Chunk text
    3. Generate embeddings
    """

    filename = ctx.event.data["filename"]
    file_path = ctx.event.data["path"]

    # Step 1: Read document
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Step 2: Chunking
    chunks = chunk_text(text)
    base = filename.replace(".txt", "")
    chunk_files = []

    for i, chunk in enumerate(chunks):
        name = f"{base}_chunk_{i}.txt"
        path = os.path.join(CHUNK_DIR, name)
        with open(path, "w", encoding="utf-8") as cf:
            cf.write(chunk)
        chunk_files.append(name)

    # Step 3: Embeddings
    embeddings = []
    for file in chunk_files:
        with open(os.path.join(CHUNK_DIR, file), "r", encoding="utf-8") as cf:
            content = cf.read()

        vector = model.encode(content).tolist()
        embeddings.append({
            "chunk_file": file,
            "embedding": vector
        })

    with open(os.path.join(EMBED_DIR, f"{base}_embeddings.json"), "w") as ef:
        json.dump(embeddings, ef)

    return {
        "status": "completed",
        "chunks": len(chunk_files),
        "embeddings": len(embeddings)
    }
