
---

# Knowledge-base RAG Chatbot

*A Local Retrieval-Augmented Generation (RAG) System using Free LLMs and FAISS*

---

### Demo & Documentation

1) Watch Demo Video: [https://drive.google.com/your-demo-video-link](https://drive.google.com/file/d/1SlzhfOEl-c75NlnLcxhZoqtxeQEvlVVs/view?usp=drive_link)
2) Read Project Report: [https://drive.google.com/your-project-report-link](https://drive.google.com/file/d/1qYhZsDSxsU1H8B2xJrm2Vrir8BQ6ByXq/view?usp=sharing)

---

## Overview

The Knowledge-base RAG Chatbot is an intelligent document-aware assistant that allows users to upload their own PDFs, text, or DOC files, and then ask natural language questions about the content.

It uses:

* Sentence Transformers for embeddings
* FAISS for vector similarity search
* Flan-T5 (open-source) for response generation
* Flask + Streamlit hybrid web interface for interactivity

All of this runs locally on CPU, using free models and no paid APIs, making it ideal for demonstrations, research, and internal use.

---

## Key Features

* PDF / DOC Uploads – Upload multiple documents via the web interface.
* RAG-based Question Answering – Uses FAISS retrieval + LLM generation for contextual answers.
* Source Citation – Each answer cites the source document and page number.
* Confidence Scoring – Displays similarity-based confidence score per answer.
* Document Browser – Click to open or view retrieved passages and source context.
* Interactive Refinement – “Refine Answer” button allows users to rephrase or simplify responses.
* TL;DR Summary – Automatically generates one-line summaries for long answers.
* User Feedback Logging – Collects thumbs up/down and comments for evaluation and improvement.
* Completely Offline & Free – Runs on CPU without any paid API keys or cloud dependencies.

---

## Architecture

User → Flask API → SentenceTransformer Embeddings → FAISS Index → Context Chunks
→ Generator (Flan-T5) → Synthesized Answer + TL;DR + Confidence + Citations

---

## Tech Stack

Backend: Flask
Frontend: HTML, CSS, JS, Streamlit
Embeddings: sentence-transformers/all-MiniLM-L6-v2
Vector Store: FAISS
LLM Generator: google/flan-t5-base
Summarization: Flan-T5
Feedback Storage: Local JSON logs
Environment: Python 3.10+

---

## Project Structure
```
kb_rag_chatbot/
├─ app.py – Flask app (core backend logic)
├─ ingest.py – PDF/DOC loader & text chunking
├─ build_index.py – Builds FAISS index from embeddings
├─ utils.py – Helper functions for saving/loading
├─ data/
│   ├─ docs/ – Uploaded documents
│   ├─ faiss_index/ – FAISS index and embeddings
│   ├─ chunks.jsonl – Chunk metadata (Created after files are uploaded)
│   └─ feedback.jsonl – Logged user feedback (Created after feedback is sent)
├─ static/
│   ├─ style.css – Frontend styling
│   └─ script.js – Frontend logic
├─ templates/
│   └─ index.html – Main web UI
├─ requirements.txt
└─ README.md
```
---

## Installation & Setup Guide (Windows)

Minimum requirements:

* Python 3.10+
* 8 GB RAM
* ~10 GB disk space

1. Clone the Repository

   ```
   cd "D:\kb_rag_chatbot"
   git clone https://github.com/Briha-003/Unthinkable.git
   cd Unthinkable
   ```

2. Create Virtual Environment

   ```
   python -m venv venv
   ```

3. Activate Environment

   ```
   venv\Scripts\Activate.ps1
   ```

   If you see a PowerShell restriction error, run:

   ```
   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
   ```

   Then re-run:

   ```
   venv\Scripts\Activate.ps1
   ```

4. Upgrade pip

   ```
   python -m pip install --upgrade pip
   ```

5. Install Required Packages

   ```
   pip install flask==3.0.3
   pip install sentence-transformers==2.7.0
   pip install transformers==4.44.2
   pip install torch==2.3.1
   pip install faiss-cpu==1.11.0
   pip install numpy==1.26.4
   pip install markupsafe==2.1.5
   pip install werkzeug==3.0.3
   pip install gunicorn==22.0.0
   ```

---

## Run the Chatbot

Start the Flask app:

```
python app.py
```

Then open:

```
http://127.0.0.1:5000/
```

You’ll see the chatbot interface - upload files, ask questions, and explore the retrieved answers.

---

## Refinement & Feedback

* Click “Refine Answer” to enter a custom instruction, e.g. “Make it concise”, “Explain in reference to..”, "Give in details.."
* The model improves its previous answer rather than regenerating from scratch.
* You can rate each answer and leave feedback; this data is stored in `data/feedback.jsonl`.

---

## Advanced Configuration

GEN_MODEL – Change between flan-t5-small or flan-t5-base (in app.py)
EMB_MODEL – Change embedding model for better retrieval (in app.py)
MAX_CONTENT_LENGTH – Max upload size 
top_k – Number of chunks to retrieve (in static/script.js)

---

## 🧾 Logs & Feedback

User feedback is saved in `data/feedback.jsonl` as:

```
{
  "ts": 1728838927.12,
  "question": "What is RAG?",
  "answer": "RAG stands for Retrieval-Augmented Generation...",
  "feedback": "up",
  "comment": "Clear and concise.",
  "contexts": [...]
}
```

This allows for evaluation and retraining on user interactions.

---

## Troubleshooting

* ModuleNotFoundError: flask → Run `pip install flask` inside venv
* No index found → Run `python ingest.py` and `python build_index.py`
* Upload failed: TypeError: Failed to fetch → Ensure Flask debug mode is off (debug=False)
* markupsafe._speedups → `pip install --force-reinstall markupsafe==2.1.5`
* torch or faiss install fails → Try `pip install --upgrade pip setuptools wheel` and retry

---

## License

This project is released under the MIT License.
You are free to use, modify, and distribute this code with attribution.

---

## Acknowledgements

* SentenceTransformers: [https://www.sbert.net/](https://www.sbert.net/)
* FAISS by Facebook AI: [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
* Flan-T5 by Google: [https://huggingface.co/google/flan-t5-base](https://huggingface.co/google/flan-t5-base)
* Streamlit: [https://streamlit.io](https://streamlit.io)
* Flask: [https://flask.palletsprojects.com](https://flask.palletsprojects.com)

---
