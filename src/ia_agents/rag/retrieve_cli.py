"""Inspect semantic retrieval results from a persistent ChromaDB collection."""

import argparse
from pathlib import Path

import chromadb
from chromadb.errors import ChromaError

from .embed import collection_embedding, get_chunks, search

FILTER_OPERATORS = {
    "eq": "$eq", "ne": "$ne", "in": "$in", "nin": "$nin",
    "gt": "$gt", "gte": "$gte", "lt": "$lt", "lte": "$lte",
}
LIST_OPERATORS = {"$in", "$nin"}
NUMERIC_OPERATORS = {"$gt", "$gte", "$lt", "$lte"}


def parse_filter(text: str) -> dict:
    """Turn KEY=VALUE or KEY:OP=VALUE into one Chroma metadata clause.

    OP is eq (the default), ne, in, nin, gt, gte, lt or lte. For in and nin,
    VALUE is a comma-separated list. The numeric operators need a numeric VALUE.
    """
    head, separator, value = text.partition("=")
    key, _, operator_name = head.partition(":")
    key = key.strip()
    operator_name = operator_name.strip() or "eq"
    if not separator or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE or KEY:OP=VALUE, got {text!r}")
    if operator_name not in FILTER_OPERATORS:
        raise argparse.ArgumentTypeError(
            f"unknown operator {operator_name!r} in {text!r}; "
            f"use one of {', '.join(FILTER_OPERATORS)}"
        )
    operator = FILTER_OPERATORS[operator_name]
    if operator == "$eq":
        return {key: value}
    if operator in LIST_OPERATORS:
        return {key: {operator: [item.strip() for item in value.split(",")]}}
    if operator in NUMERIC_OPERATORS:
        try:
            number = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"{operator_name} needs a number, got {value!r} in {text!r}"
            ) from None
        return {key: {operator: number}}
    return {key: {operator: value}}


def build_where(clauses: list[dict]) -> dict | None:
    """Combine clauses so that all must match. Returns None when there are none."""
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def print_chunk(result: dict, rank: int | None = None) -> None:
    metadata = result["metadata"]
    prefix = f"\n  {rank}. " if rank is not None else "\n  "
    distance = f"distance={result['distance']:.4f} | " if "distance" in result else ""
    print(f"{prefix}{distance}id={result['chunk_id']}")
    print(f"     section={metadata.get('section_id')} | {metadata.get('section_title')}")
    print(f"     source={metadata.get('source')} | document={metadata.get('document_id')}")
    print(f"     {result['text']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "queries", nargs="*",
        help="One or more search queries (leave empty when using --chunk-id)",
    )
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
        metavar="KEY[:OP]=VALUE",
        help="Keep only chunks whose metadata KEY matches, e.g. --filter source=guia-apis. "
             "OP is eq (default), ne, in, nin, gt, gte, lt or lte; for in/nin give a "
             "comma-separated VALUE, e.g. --filter tema:in=apis,rag. Repeat to require "
             "several (all must match).",
    )
    parser.add_argument("--section", help="Filter by section ID, e.g. section_04")
    parser.add_argument("--document-id", help="Filter by document ID")
    parser.add_argument("--contains", metavar="TEXT", help="Keep only chunks whose text contains TEXT")
    parser.add_argument(
        "--regex", metavar="PATTERN",
        help="Keep only chunks whose text matches the regular expression PATTERN",
    )
    parser.add_argument(
        "--chunk-id", dest="chunk_ids", action="append", default=[], metavar="ID",
        help="Fetch a chunk by its exact ID instead of searching (repeatable)",
    )
    args = parser.parse_args(argv)

    if args.k <= 0:
        parser.error("-k must be greater than zero.")
    if args.max_distance is not None and args.max_distance < 0:
        parser.error("--max-distance must be non-negative.")
    if args.queries and args.chunk_ids:
        parser.error("Use either search queries or --chunk-id, not both.")
    if not args.queries and not args.chunk_ids:
        parser.error("Give one or more search queries, or --chunk-id.")

    client = chromadb.PersistentClient(path=str(args.persist_directory))
    try:
        collection = client.get_collection(name=args.collection)
    except Exception as exc:
        parser.error(f"Could not open collection {args.collection!r}: {exc}")

    filters = list(args.filters)
    if args.section:
        filters.append({"section_id": args.section})
    if args.document_id:
        filters.append({"document_id": args.document_id})
    where = build_where(filters)

    text_filters = []
    if args.contains is not None:
        text_filters.append({"$contains": args.contains})
    if args.regex is not None:
        text_filters.append({"$regex": args.regex})
    where_document = build_where(text_filters)

    if args.chunk_ids:
        try:
            chunks = get_chunks(
                collection, ids=args.chunk_ids, where=where, where_document=where_document,
            )
        except (ValueError, ChromaError) as exc:
            parser.error(str(exc))
        if not chunks:
            print("No chunks matched.")
        for result in chunks:
            print_chunk(result)
        return

    try:
        recorded = collection_embedding(collection)
    except ValueError as exc:
        parser.error(str(exc))
    print(
        f"Embedding (stored vectors and queries): "
        f"{recorded['embedding_provider']} / {recorded['embedding_model']}"
    )

    try:
        results_by_query = search(
            args.queries,
            collection,
            k=args.k,
            max_distance=args.max_distance,
            where=where,
            where_document=where_document,
        )
    except (ValueError, ChromaError) as exc:
        parser.error(str(exc))
    for query, results in zip(args.queries, results_by_query):
        print(f"\nQuery: {query}")
        if not results:
            print("  No matches (or all matches exceeded --max-distance).")
            continue
        for rank, result in enumerate(results, start=1):
            print_chunk(result, rank)


if __name__ == "__main__":
    main()
