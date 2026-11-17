import chromadb

from .embed import search
from .pipeline import DEFAULT_K


def diagnose_retrieval(
    query: str,
    expected_chunk_id: str,
    collection: chromadb.Collection,
    k: int = DEFAULT_K,
) -> dict:
    """check whether the chunk that should answer a query is in the top k.
    Run retrieval for ``query`` and report whether ``expected_chunk_id`` is among the top k.

    Returns ``{"ok", "k", "rank", "distance", "retrieved", "chunks"}``. ``rank`` (1-based) and
    ``distance`` are None when the chunk is missing, which is a retrieval failure: the query
    embedding is too far from the chunk, or k is too small. ``retrieved`` lists the returned
    chunk ids in ranking order; ``chunks`` is the full ``search()`` output.
    """
    chunks = search(query, collection, k=k)
    retrieved = [chunk["chunk_id"] for chunk in chunks]
    rank = retrieved.index(expected_chunk_id) + 1 if expected_chunk_id in retrieved else None
    return {
        "ok": rank is not None,
        "k": k,
        "rank": rank,
        "distance": chunks[rank - 1]["distance"] if rank else None,
        "retrieved": retrieved,
        "chunks": chunks,
    }
