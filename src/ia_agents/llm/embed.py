import os, json
import re
import unicodedata
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Literal
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

def embed_google(texts: list[str]) -> list[list[float]]:
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    resp = client.models.embed_content(
        model=os.environ.get("GOOGLE_EMBEDDING_MODEL_ID", "text-embedding-004"),
        contents=texts,
    )
    return [e.values for e in resp.embeddings]

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text) # NFKC: Normalization Form Compatibility Composition. There 4 dif options in the Unicode normalization standard: https://unicode.org/reports/tr15/
    text = re.sub(r"[ \t]+", " ", text) # Replaces each consecutive group of spaces or tabs with one space
    text = re.sub(r"\n{3,}", "\n\n", text) # Replaces three or more consecutive newline characters with two, preserving a single blank line between paragraphs
    
    ### Add more rules relative to the text to be processed...

    return text.strip()

def chunk_by_characters(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into fixed-size, overlapping character chunks without respecting sentence boundaries."""
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks

def chunk_by_paragraphs(text: str, max_chars: int = 500) -> list[str]:
    """Group paragraphs separated by blank lines up to the maximum size."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 1 <= max_chars:
            current = (current + "\n\n" + paragraph).strip()
        else:
            if current:
                chunks.append(current)
            current = paragraph
    if current:
        chunks.append(current)
    return chunks

# Other chunking methods: https://medium.com/the-ai-forum/semantic-chunking-for-rag-f4733025d5f5

@dataclass
class Chunk:
    text: str
    document_id: str
    position: int # chunk order within the document
    source: str
    ingested_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    chunk_id: str = field(init=False)

    def __post_init__(self):
        # Deterministic ID based on document ID, position, and content.
        content_hash = hashlib.sha256(self.text.encode()).hexdigest()[:8]
        self.chunk_id = f"{self.document_id}-{self.position:03d}-{content_hash}"

def build_chunks(
    text: str,
    document_id: str,
    source: str,
    chunk_size: int = 300,
    overlap: int = 50,
    strategy: Literal["characters", "paragraphs"] = "characters",
    max_chars: int = 500,
) -> list[Chunk]:
    """Build chunks using the selected strategy.

    chunk_size and overlap apply only to characters. max_chars applies only to
    paragraphs, which remain intact even when longer than max_chars.
    """
    if strategy == "characters":
        texts = chunk_by_characters(text, chunk_size, overlap)
    elif strategy == "paragraphs":
        texts = chunk_by_paragraphs(text, max_chars=max_chars)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}")
    return [
        Chunk(text=t, document_id=document_id, position=i, source=source)
        for i, t in enumerate(texts)
    ]
