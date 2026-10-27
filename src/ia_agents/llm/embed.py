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
    if not texts:
        return []
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    resp = client.models.embed_content(
        model=os.environ.get("GOOGLE_EMBEDDING_MODEL_ID", "gemini-embedding-2"),
        # Explicit Content boundaries prevent Embedding 2 from aggregating
        # all strings into a single embedding.
        contents=[types.Content(parts=[types.Part(text=text)]) for text in texts],
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
    section_id: str | None = None
    section_title: str | None = None
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
    """Build section-aware chunks using the selected strategy.

    chunk_size and overlap apply only to characters. max_chars applies only to
    paragraphs, which remain intact even when longer than max_chars. Numbered
    headings start new sections, and chunks never cross section boundaries.
    """
    if strategy not in ("characters", "paragraphs"):
        raise ValueError(f"Unknown chunking strategy: {strategy!r}")

    heading_pattern = re.compile(r"(?m)^(\d+)\.\s+(.+?)\s*$")
    headings = list(heading_pattern.finditer(text))
    sections: list[tuple[str, str, str]] = []
    if headings:
        if text[:headings[0].start()].strip():
            sections.append(("introduction", "Introduction", text[:headings[0].start()].strip()))
        for index, heading in enumerate(headings):
            end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
            section_id = f"section_{int(heading.group(1)):02d}"
            section_title = heading.group(2).strip()
            section_text = text[heading.start():end].strip()
            sections.append((section_id, section_title, section_text))
    else:
        sections.append(("section_00", "Document", text))

    chunks: list[Chunk] = []
    for section_id, section_title, section_text in sections:
        if strategy == "characters":
            texts = chunk_by_characters(section_text, chunk_size, overlap)
        else:
            texts = chunk_by_paragraphs(section_text, max_chars=max_chars)
        position_offset = len(chunks)
        chunks.extend([
            Chunk(
                text=chunk_text,
                document_id=document_id,
                position=position_offset + offset,
                source=source,
                section_id=section_id,
                section_title=section_title,
            )
            for offset, chunk_text in enumerate(texts)
        ])
    return chunks
