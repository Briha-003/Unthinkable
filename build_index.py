# build_index.py
import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm

DATA_DIR = "data"
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.jsonl")
INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    os.makedirs(INDEX_DIR, exist_ok=True)
    print("Loading embedding model:", EMB_MODEL)
    model = SentenceTransformer(EMB_MODEL)
    texts = []
    metas = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            metas.append(obj)

    if len(texts) == 0:
        print("No chunks to index. Run ingest.py first or upload files.")
        return

    batch_size = 32
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings).astype("float32")
    dim = embeddings.shape[1]
    print("Embeddings shape:", embeddings.shape)

    # Build FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))

    # Save metas and embeddings
    with open(os.path.join(INDEX_DIR, "metas.jsonl"), "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    np.save(os.path.join(INDEX_DIR, "embeddings.npy"), embeddings)
    print("Index and metadata saved to", INDEX_DIR)

if __name__ == "__main__":
    main()
