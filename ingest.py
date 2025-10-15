# ingest.py
import json
import os
from tqdm import tqdm
from utils import pdf_to_pages, read_text_file, chunk_text_pages, ensure_dir, secure_save_filename

DATA_DIR = "data"
DOCS_DIR = os.path.join(DATA_DIR, "docs")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.jsonl")

def ingest_existing_docs():
    ensure_dir(DATA_DIR)
    ensure_dir(DOCS_DIR)
    docs_files = [f for f in os.listdir(DOCS_DIR) if not f.startswith(".")]
    print(f"Found {len(docs_files)} files in {DOCS_DIR}")
    entries = []
    for fname in docs_files:
        path = os.path.join(DOCS_DIR, fname)
        print("Processing:", fname)
        if fname.lower().endswith(".pdf"):
            pages = pdf_to_pages(path)
        elif fname.lower().endswith((".txt", ".md")):
            pages = read_text_file(path)
        else:
            print("Skipping unsupported:", fname)
            continue
        chunks = chunk_text_pages(pages, chunk_size_words=400, overlap=80)
        for i, c in enumerate(chunks):
            entry = {
                "doc": fname,
                "page": c["page"],
                "chunk_id": i,
                "text": c["text"][:3000]  # cap for safety
            }
            entries.append(entry)
    # write to jsonl (overwrite)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as out:
        for e in entries:
            out.write(json.dumps(e, ensure_ascii=False) + "\n")
    print("Wrote", len(entries), "chunks to", CHUNKS_PATH)

if __name__ == "__main__":
    ingest_existing_docs()
