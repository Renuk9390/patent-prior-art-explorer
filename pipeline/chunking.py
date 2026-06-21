"""
Text chunking utilities.

Two chunking paths feed the same downstream pipeline:
  - The curated demo corpus (corpus/patents.py) is already split into
    short, hand-picked paragraphs, so each paragraph is used as a chunk
    directly -- no further splitting needed.
  - User-uploaded PDFs are raw, unstructured text, so `chunk_text()`
    below greedily groups sentences into chunks of a target length.

Kept dependency-free (just the standard library `re` module, plus
`pypdf` for PDF parsing) on purpose: fewer dependencies means fewer
things that can fail to install on a free-tier Space, and a sentence
splitter this simple is easy for anyone picking up the project later to
read and modify in five minutes.
"""

from __future__ import annotations

import re

# Splits after sentence-ending punctuation, only when followed by a
# capital letter or an opening parenthesis -- avoids splitting on
# abbreviations like "e.g." or decimal numbers like "35 U.S.C." in most
# practical cases without needing a full NLP sentence tokenizer.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def split_sentences(text: str) -> list[str]:
    """A lightweight, dependency-free sentence splitter."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, target_words: int = 90, max_words: int = 160) -> list[str]:
    """Greedily group sentences into chunks of roughly `target_words`.

    Used for arbitrary uploaded documents, where paragraph boundaries
    from PDF text extraction are often unreliable or missing entirely.
    """
    sentences = split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())

        if current and current_words + sentence_words > max_words:
            chunks.append(" ".join(current))
            current = []
            current_words = 0

        current.append(sentence)
        current_words += sentence_words

        if current_words >= target_words:
            chunks.append(" ".join(current))
            current = []
            current_words = 0

    if current:
        chunks.append(" ".join(current))

    # Drop tiny fragments (e.g. a stray heading or page number) that
    # aren't useful as standalone search results.
    return [c for c in chunks if len(c.split()) >= 8]


def extract_pdf_text(file_path: str) -> str:
    """Extract raw text from a PDF file, page by page.

    Returns an empty string (never raises) for image-only / scanned
    PDFs with no extractable text layer -- the caller is responsible
    for surfacing a friendly message in that case, since OCR is
    intentionally out of scope for this lightweight demo.
    """
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(page_text)
    return "\n\n".join(pages)
