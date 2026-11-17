"""Multi-query retrieval: search with LLM-written rephrasings of a question and merge the results."""

import re

import chromadb

from ..llm import LLMClient
from ..llm.responses import response_details
from .embed import search

VARIANTS_SYSTEM_PROMPT = (
    "Rewrite the user's question as {n} different search queries for technical documentation. "
    "Each one stresses a different aspect, term or possible interpretation. "
    "Return one query per line, with no numbering and no explanation."
)

LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def generate_variants(query: str, llm: LLMClient, n: int) -> list[str]:
    """Ask the LLM for up to n rephrasings of the query, without blanks, repeats or the original."""
    result = llm.generate(query, system=VARIANTS_SYSTEM_PROMPT.format(n=n), temperature=0.0)
    text = response_details(llm.provider, result.response)[0]
    seen = {query.strip().casefold()}
    variants: list[str] = []
    for line in text.splitlines():
        variant = LIST_MARKER_RE.sub("", line).strip()
        if variant and variant.casefold() not in seen:
            seen.add(variant.casefold())
            variants.append(variant)
    return variants[:n]


def multi_query_search(
    query: str,
    collection: chromadb.Collection,
    llm: LLMClient,
    n_variants: int,
    k: int,
    where: dict | None = None,
    where_document: dict | None = None,
) -> tuple[list[dict], list[str]]:
    """Search with the query and its variants, keep each chunk's best distance, return the top k.

    Returns the merged chunks (same shape as ``search()``, closest first) and the variants used.
    """
    variants = generate_variants(query, llm, n_variants)
    results = search([query, *variants], collection, k=k, where=where, where_document=where_document)
    best: dict[str, dict] = {}
    for chunks in results:
        for chunk in chunks:
            current = best.get(chunk["chunk_id"])
            if current is None or chunk["distance"] < current["distance"]:
                best[chunk["chunk_id"]] = chunk
    return sorted(best.values(), key=lambda chunk: chunk["distance"])[:k], variants
