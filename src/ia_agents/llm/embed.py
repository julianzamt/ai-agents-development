import os, json
import re
import unicodedata
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Literal, Protocol
from dotenv import load_dotenv
from google import genai
from google.genai import types
import chromadb

load_dotenv()


class EmbeddingProvider(Protocol):
    """Provider-neutral interface for retrieval document and query vectors."""

    def embed_documents(self, texts: list[str], titles: list[str | None] | None = None) -> list[list[float]]: ...

    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...


class GoogleEmbeddingProvider:
    def __init__(self) -> None:
        self.model = os.environ.get("GOOGLE_EMBEDDING_MODEL_ID", "gemini-embedding-2")
        self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    def _embed(self, texts: list[str], purpose: Literal["document", "query"], titles: list[str | None] | None = None):
        if not texts:
            return []
        if self.model.startswith("gemini-embedding-2"):
            if purpose == "query":
                contents = [
                    types.Content(parts=[types.Part(text=f"task: search result | query: {text}")])
                    for text in texts
                ]
            else:
                if titles is not None and len(titles) != len(texts):
                    raise ValueError("Provide one document title per text.")
                document_titles = titles or [None] * len(texts)
                contents = [
                    types.Content(parts=[types.Part(text=f"title: {title or 'none'} | text: {text}")])
                    for text, title in zip(texts, document_titles)
                ]
            config = None
        else:
            contents = [
                types.Content(parts=[types.Part(text=text)])
                for text in texts
            ]
            config = types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY" if purpose == "query" else "RETRIEVAL_DOCUMENT"
            )
        response = self.client.models.embed_content(
            model=self.model,
            contents=contents,
            **({"config": config} if config is not None else {}),
        )
        return [embedding.values for embedding in response.embeddings]

    def embed_documents(self, texts: list[str], titles: list[str | None] | None = None) -> list[list[float]]:
        return self._embed(texts, "document", titles=titles)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "query")


class OpenAIEmbeddingProvider:
    def __init__(self) -> None:
        from openai import OpenAI

        self.model = os.environ.get("OPENAI_EMBEDDING_MODEL_ID", "text-embedding-3-small")
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_documents(self, texts: list[str], titles: list[str | None] | None = None) -> list[list[float]]:
        return self._embed(texts)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)


class OllamaEmbeddingProvider:
    def __init__(self) -> None:
        from ollama import Client

        self.model = os.environ.get("OLLAMA_EMBEDDING_MODEL_ID", "nomic-embed-text")
        self.client = Client(host=os.environ.get("OLLAMA_HOST"))

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embed(model=self.model, input=texts)
        return response.embeddings

    def embed_documents(self, texts: list[str], titles: list[str | None] | None = None) -> list[list[float]]:
        return self._embed(texts)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)


def get_embedding_provider(provider: str | EmbeddingProvider | None = None) -> EmbeddingProvider:
    """Create the configured embedding provider (Google by default)."""
    if provider is not None and not isinstance(provider, str):
        return provider
    provider = (provider or os.environ.get("EMBEDDING_PROVIDER", "google")).lower()
    providers = {
        "google": GoogleEmbeddingProvider,
        "openai": OpenAIEmbeddingProvider,
        "ollama": OllamaEmbeddingProvider,
    }
    try:
        return providers[provider]()
    except KeyError as exc:
        raise ValueError(f"Unsupported embedding provider: {provider!r}") from exc


def embed_documents(texts: list[str], titles: list[str | None] | None = None) -> list[list[float]]:
    if not texts:
        return []
    return get_embedding_provider().embed_documents(texts, titles=titles)


def embed_queries(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return get_embedding_provider().embed_queries(texts)


def search(
    queries: str | list[str],
    collection: chromadb.Collection,
    k: int = 3,
    max_distance: float | None = None,
    where: dict | None = None,
    provider: str | EmbeddingProvider = "google",
) -> list[dict] | list[list[dict]]:
    """Search a Chroma collection with one or more text queries.

    A single string returns a flat list of results. A list of strings returns
    one result list per query. If ``max_distance`` is set, results farther
    away than that threshold are omitted. Distances are returned as Chroma
    reports them and depend on the collection's configured metric.
    """
    if k <= 0:
        return [] if isinstance(queries, str) else [[] for _ in queries]

    single_query = isinstance(queries, str)
    query_texts = [queries] if single_query else queries
    if not query_texts:
        return []

    embedding_provider = get_embedding_provider(provider)
    query_embeddings = embedding_provider.embed_queries(query_texts)
    if len(query_embeddings) != len(query_texts):
        raise ValueError("Google returned a different number of vectors than queries.")

    query_options = {"query_embeddings": query_embeddings, "n_results": k}
    if where is not None:
        query_options["where"] = where
    response = collection.query(**query_options)

    ids_by_query = response.get("ids", [])
    documents_by_query = response.get("documents", [])
    metadatas_by_query = response.get("metadatas", [])
    distances_by_query = response.get("distances", [])
    if not all(len(field) == len(query_texts) for field in (
        ids_by_query, documents_by_query, metadatas_by_query, distances_by_query,
    )):
        raise ValueError("Chroma returned a different number of result groups than queries.")

    results_by_query = []
    for ids, documents, metadatas, distances in zip(
        ids_by_query, documents_by_query, metadatas_by_query, distances_by_query,
    ):
        query_results = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            if max_distance is None or distance <= max_distance:
                query_results.append({
                    "chunk_id": chunk_id,
                    "text": text,
                    "distance": distance,
                    "metadata": metadata or {},
                })
        results_by_query.append(query_results)

    return results_by_query[0] if single_query else results_by_query

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
