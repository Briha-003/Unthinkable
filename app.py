# app.py
import os
import json
import time
from flask import Flask, request, jsonify, render_template, send_from_directory
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from utils import ensure_dir, secure_save_filename
from werkzeug.utils import secure_filename

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_DIR = "data"
DOCS_DIR = os.path.join(DATA_DIR, "docs")
INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.jsonl")
FEEDBACK_PATH = os.path.join(DATA_DIR, "feedback.jsonl")

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-small"  # can be upgraded to flan-t5-base if system allows

ensure_dir(DATA_DIR)
ensure_dir(DOCS_DIR)
ensure_dir(INDEX_DIR)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB upload limit

# -----------------------------
# LOAD MODELS
# -----------------------------
print("Loading embedding model...")
embedder = SentenceTransformer(EMB_MODEL)

index = None
all_chunks = []
embeddings = None


def load_index():
    """Load FAISS index and chunk metadata"""
    global index, all_chunks, embeddings
    idx_path = os.path.join(INDEX_DIR, "index.faiss")
    meta_path = os.path.join(INDEX_DIR, "metas.jsonl")
    emb_path = os.path.join(INDEX_DIR, "embeddings.npy")

    if os.path.exists(idx_path) and os.path.exists(meta_path):
        print("Loading FAISS index...")
        index = faiss.read_index(idx_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            all_chunks = [json.loads(line) for line in f]
        if os.path.exists(emb_path):
            embeddings = np.load(emb_path)
        print(f"✅ Loaded index with {len(all_chunks)} chunks.")
    else:
        print("⚠️ No index found. Please run ingestion or upload files via UI.")


load_index()

# -----------------------------
# LOAD GENERATOR MODEL
# -----------------------------
print("Loading generator model (may take time)...")
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
device = "cpu"
gen_model.to(device)
print("✅ Generator loaded.")


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def compute_confidence_from_distances(dists):
    """Convert FAISS distances to similarity-like confidence scores."""
    sims = [float(1.0 / (1.0 + float(d))) for d in dists]
    return sims


def retrieve_with_scores(query, top_k=4):
    """Retrieve top_k most similar chunks from FAISS."""
    if index is None:
        raise RuntimeError("No FAISS index loaded. Please run ingestion first.")

    q_emb = embedder.encode([query])
    D, I = index.search(q_emb, top_k)

    contexts = []
    for rank, idx in enumerate(I[0]):
        if idx == -1 or idx >= len(all_chunks):
            continue
        chunk = all_chunks[idx]
        sim = float(1.0 / (1.0 + float(D[0][rank])))

        c = {
            "rank": rank + 1,
            "doc": chunk.get("doc", "unknown"),
            "page": chunk.get("page", "-"),
            "chunk_id": chunk.get("chunk_id", idx),
            "text": chunk.get("text", ""),
            "_similarity": sim,
            "_faiss_dist": float(D[0][rank]),
            "_url": f"/docs/{os.path.basename(chunk.get('doc', '#'))}#page={chunk.get('page', '-')}",
        }
        contexts.append(c)

    # Filter out irrelevant chunks
    contexts = [c for c in contexts if c["_similarity"] >= 0.35]
    return contexts, D[0][: len(contexts)]


def synthesize_answer(
    question: str,
    contexts,
    refine_instruction: str = None,
    previous_answer: str = None,
    max_length=512,
):
    """
    Generates or refines an answer using Flan-T5 with structured, factual prompts.
    Adds support for analogies, bullet points, and simplification if requested.
    """

    context_text = "\n\n".join(
        [
            f"Doc: {c['doc']} | Page: {c['page']} | Chunk {c['chunk_id']}:\n{c['text']}"
            for c in contexts
        ]
    )

    if previous_answer and refine_instruction:
        # REFINEMENT PROMPT
        base_prompt = (
            "You are a helpful assistant refining an existing answer using only the provided context.\n"
            "Follow these rules:\n"
            "1. Correct factual errors using ONLY the context below.\n"
            "2. If the context lacks information, explicitly say so.\n"
            "3. Follow the user's refinement instruction carefully.\n"
            "4. Present the refined answer in clear, structured bullet points where suitable.\n"
            "5. If asked to simplify, use relatable analogies and everyday examples.\n"
            "6. Keep the tone factual, concise, and friendly.\n"
            "7. Include citations in parentheses (DocName, Page).\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            f"Previous answer:\n{previous_answer}\n\n"
            f"Refinement instruction: {refine_instruction}\n\n"
            "Now write the improved answer below:\n"
        )
    else:
        # INITIAL PROMPT
        base_prompt = (
            "You are an intelligent assistant that answers questions using only the provided context.\n"
            "Guidelines:\n"
            "1. Answer factually using ONLY the context provided.\n"
            "2. If information is missing, state 'I don’t know based on the given context.'\n"
            "3. Provide your answer in well-structured bullet points for clarity.\n"
            "4. If asked to simplify, use simple analogies to explain complex ideas.\n"
            "5. Add citations in parentheses (DocName, Page).\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\nAnswer:"
        )

    # Generate
    inputs = tokenizer(base_prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.6,
            top_p=0.9,
            num_beams=4,
            early_stopping=True,
            repetition_penalty=1.15,
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return answer


def generate_tldr(full_answer: str, max_length=40):
    """Generate one-line TL;DR summary."""
    prompt = f"Write a one-line TL;DR summary of the following answer:\n\n{full_answer}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    out = gen_model.generate(**inputs, max_length=max_length, num_beams=2, early_stopping=True)
    return tokenizer.decode(out[0], skip_special_tokens=True)


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def index_page():
    return render_template("index.html")


@app.route("/docs/<path:filename>")
def serve_document(filename):
    """Serve uploaded PDFs or docs."""
    return send_from_directory(DOCS_DIR, filename, as_attachment=False)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Handle PDF/doc uploads and rebuild FAISS index."""
    uploaded = request.files.getlist("files")
    if not uploaded:
        return jsonify({"error": "No files uploaded"}), 400

    saved = []
    for f in uploaded:
        fname = secure_save_filename(f.filename)
        out_path = os.path.join(DOCS_DIR, fname)
        f.save(out_path)
        saved.append(fname)

    try:
        import ingest as ingest_mod
        import build_index as build_mod

        ingest_mod.ingest_existing_docs()
        build_mod.main()
        load_index()
    except Exception as e:
        return jsonify({"error": f"Uploaded but failed to index: {e}"}), 500

    return jsonify({"saved": saved})


@app.route("/api/query", methods=["POST"])
def api_query():
    """Main RAG QA endpoint"""
    payload = request.json or {}
    question = payload.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    top_k = int(payload.get("top_k", 4))
    refine_instruction = payload.get("refine_instruction")
    previous_answer = payload.get("previous_answer")

    t0 = time.perf_counter()
    contexts, dists = retrieve_with_scores(question, top_k=top_k)
    sims = compute_confidence_from_distances(dists)
    confidence = float(sum(sims) / len(sims)) if sims else 0.0

    # Generate or refine
    answer = synthesize_answer(
        question,
        contexts,
        refine_instruction=refine_instruction,
        previous_answer=previous_answer,
        max_length=512,
    )
    tldr = generate_tldr(answer, max_length=40)
    elapsed = time.perf_counter() - t0

    # attach similarity to each context
    for i, c in enumerate(contexts):
        c["_similarity"] = sims[i] if i < len(sims) else 0.0
        c["_url"] = f"/docs/{c['doc']}#page={c['page']}"

    resp = {
        "answer": answer,
        "tldr": tldr,
        "contexts": contexts,
        "confidence": confidence,
        "time": elapsed,
    }
    return jsonify(resp)


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    """Save feedback from user."""
    data = request.json or {}
    record = {
        "ts": time.time(),
        "question": data.get("question"),
        "answer": data.get("answer"),
        "feedback": data.get("feedback"),
        "comment": data.get("comment", ""),
        "contexts": data.get("contexts", []),
    }
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return jsonify({"ok": True})


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    app.run(debug=False, use_reloader=False, port=5000)
