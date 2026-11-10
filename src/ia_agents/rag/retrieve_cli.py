"""Inspect semantic retrieval results from a persistent ChromaDB collection."""

import argparse
from pathlib import Path

import chromadb

from .embed import collection_embedding, search


def parse_filter(text: str) -> tuple[str, str]:
    """Split a KEY=VALUE metadata filter argument."""
    key, separator, value = text.partition("=")
    if not separator or not key.strip():
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    return key.strip(), value


def build_where(filters: list[tuple[str, str]]) -> dict | None:
    """Build a Chroma `where` clause requiring every metadata key to equal its value."""
    clauses = [{key: value} for key, value in filters]
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


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
    parser.add_argument(
        "--filter", dest="filters", action="append", default=[], type=parse_filter,
        metavar="KEY=VALUE",
        help="Keep only chunks whose metadata KEY equals VALUE, e.g. --filter source=guia-apis. "
             "Repeat to require several (all must match).",
    )
    parser.add_argument("--section", help="Filter by section ID, e.g. section_04")
    parser.add_argument("--document-id", help="Filter by document ID")
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

    try:
        recorded = collection_embedding(collection)
    except ValueError as exc:
        parser.error(str(exc))
    print(
        f"Embedding (stored vectors and queries): "
        f"{recorded['embedding_provider']} / {recorded['embedding_model']}"
    )

    filters = list(args.filters)
    if args.section:
        filters.append(("section_id", args.section))
    if args.document_id:
        filters.append(("document_id", args.document_id))
    where = build_where(filters)

    try:
        results_by_query = search(
            args.queries,
            collection,
            k=args.k,
            max_distance=args.max_distance,
            where=where,
        )
    except ValueError as exc:
        parser.error(str(exc))
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
