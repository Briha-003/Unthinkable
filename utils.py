# utils.py
import re
import pdfplumber
from typing import List, Dict, Tuple
import os
from werkzeug.utils import secure_filename

def clean_text(t: str) -> str:
    t = t.replace("\r\n", "\n")
    t = re.sub(r"\n{2,}", "\n\n", t)
    return t.strip()

def pdf_to_pages(path: str) -> List[str]:
    """Return a list of page texts (page 1 => index 0)"""
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            pages.append(clean_text(txt))
    return pages

def read_text_file(path: str) -> List[str]:
    """Return a single-element list treated as 'page 1' for text files."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [clean_text(f.read())]

def chunk_text_pages(pages: List[str], chunk_size_words: int = 400, overlap: int = 80) -> List[Dict]:
    """
    Create chunks, but keep track of original page numbers for each chunk.
    Returns list of dicts: {'text':..., 'page': <1-based page number>}
    """
    all_chunks = []
    for i, page_text in enumerate(pages):
        words = page_text.split()
        j = 0
        while j < len(words):
            chunk_words = words[j:j+chunk_size_words]
            chunk_text = " ".join(chunk_words)
            all_chunks.append({
                "text": chunk_text,
                "page": i+1  # 1-based
            })
            j += chunk_size_words - overlap
        # if page short, still produce one chunk even for empty page
        if len(words) == 0:
            all_chunks.append({
                "text": "",
                "page": i+1
            })
    return all_chunks

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def secure_save_filename(filename: str) -> str:
    return secure_filename(filename)
