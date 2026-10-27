"""Inspect semantic retrieval results from a persistent ChromaDB collection."""

import argparse
import os
from pathlib import Path

import chromadb

from .llm.embed import search


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queries", nargs="+", help="One or more search queries")
    parser.add_argument(
        "--persist-directory", type=Path, default=Path("chroma_db"),
        help="ChromaDB persistence directory (default: ./chroma_db)",
    )
    parser.add_argument("--collection", default="documents", help="ChromaDB collection name")
    parser.add_argument("-k", type=int, default=3, help="Matches per query (default: 3)")
    parser.add_argument(
        "--max-distance", type=float,
        help="Omit results with a larger Chroma distance",
    )
    parser.add_argument("--section", help="Filter by section ID, e.g. section_04")
    parser.add_argument("--document-id", help="Filter by document ID")
    parser.add_argument(
        "--provider", choices=["google", "openai", "ollama"],
        default=os.environ.get("EMBEDDING_PROVIDER", "google"),
        help="Embedding provider (default: EMBEDDING_PROVIDER or google)",
    )
    args = parser.parse_args(argv)

    if args.k <= 0:
        parser.error("-k must be greater than zero.")
    if args.max_distance is not None and args.max_distance < 0:
        parser.error("--max-distance must be non-negative.")

    client = chromadb.PersistentClient(path=str(args.persist_directory))
    try:
        collection = client.get_collection(name=args.collection)
    except Exception as exc:
        parser.error(f"Could not open collection {args.collection!r}: {exc}")

    filters = []
    if args.section:
        filters.append({"section_id": args.section})
    if args.document_id:
        filters.append({"document_id": args.document_id})
    where = None
    if len(filters) == 1:
        where = filters[0]
    elif filters:
        where = {"$and": filters}

    results_by_query = search(
        args.queries,
        collection,
        k=args.k,
        max_distance=args.max_distance,
        where=where,
        provider=args.provider,
    )
    for query, results in zip(args.queries, results_by_query):
        print(f"\nQuery: {query}")
        if not results:
            print("  No matches (or all matches exceeded --max-distance).")
            continue
        for rank, result in enumerate(results, start=1):
            metadata = result["metadata"]
            print(
                f"\n  {rank}. distance={result['distance']:.4f}"
                f" | id={result['chunk_id']}"
            )
            print(
                f"     section={metadata.get('section_id')}"
                f" | {metadata.get('section_title')}"
            )
            print(f"     source={metadata.get('source')} | document={metadata.get('document_id')}")
            print(f"     {result['text']}")


if __name__ == "__main__":
    main()
