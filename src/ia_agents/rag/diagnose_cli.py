"""Check whether the chunk that should answer a query is retrieved in the top k."""

import argparse
import sys
from pathlib import Path

import chromadb
from chromadb.errors import ChromaError

from .diagnose import diagnose_retrieval
from .pipeline import DEFAULT_K


def print_result(query: str, expected_chunk_id: str, result: dict) -> None:
    print(f"\nQuery: {query}")
    if result["ok"]:
        print(f"OK: {expected_chunk_id} is rank {result['rank']} of {result['k']} "
              f"(distance={result['distance']:.4f})")
    else:
        print(f"FAIL: {expected_chunk_id} is not in the top {result['k']}")
    for rank, chunk in enumerate(result["chunks"], start=1):
        marker = "*" if chunk["chunk_id"] == expected_chunk_id else " "
        print(f"  {marker}{rank}. distance={chunk['distance']:.4f} id={chunk['chunk_id']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="The question to retrieve for")
    parser.add_argument("chunk_id", help="ID of the chunk expected in the results")
    parser.add_argument(
        "--persist-directory", type=Path, default=Path("chroma_db"),
        help="ChromaDB persistence directory (default: ./chroma_db)",
    )
    parser.add_argument("--collection", default="documents", help="ChromaDB collection name")
    parser.add_argument("-k", type=int, default=DEFAULT_K, help=f"Chunks to retrieve (default: {DEFAULT_K})")
    args = parser.parse_args(argv)

    if args.k <= 0:
        parser.error("-k must be greater than zero.")

    client = chromadb.PersistentClient(path=str(args.persist_directory))
    try:
        collection = client.get_collection(name=args.collection)
    except Exception as exc:
        parser.error(f"Could not open collection {args.collection!r}: {exc}")

    try:
        result = diagnose_retrieval(args.query, args.chunk_id, collection, k=args.k)
    except (ValueError, ChromaError) as exc:
        parser.error(str(exc))
    print_result(args.query, args.chunk_id, result)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
