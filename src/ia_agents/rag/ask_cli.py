"""Answer questions from a persistent ChromaDB collection with the RAG pipeline, citing the chunks used."""

import argparse
from pathlib import Path

import chromadb
from chromadb.errors import ChromaError

from .pipeline import (
    DEFAULT_K, DEFAULT_LOG_PATH, DEFAULT_MAX_DISTANCE, DEFAULT_MAX_PROMPT_CHARS, rag,
)
from .retrieve_cli import build_where, parse_filter


def print_result(query: str, result: dict) -> None:
    record = result["record"]
    print(f"\nQuery: {query}\nAnswer: {result['answer']}")
    if result["cited"]:
        print("Sources:")
        for source in result["cited"]:
            print(f"  [{source['ref']}] {source['chunk_id']} ({source['section_title']})")
    if record["invalid_citations"]:
        print(f"Warning: cited evidence that was not provided: {record['invalid_citations']}")
    print(
        f"[outcome={record['outcome']} | retrieved={len(record['retrieved'])} | "
        f"passed threshold={len(record['passed_threshold'])} | in context={len(record['in_context'])} | "
        f"cited={len(record['cited'])} | context chars={record['context_chars']}]"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queries", nargs="+", help="One or more questions")
    parser.add_argument(
        "--persist-directory", type=Path, default=Path("chroma_db"),
        help="ChromaDB persistence directory (default: ./chroma_db)",
    )
    parser.add_argument("--collection", default="documents", help="ChromaDB collection name")
    parser.add_argument("-k", type=int, default=DEFAULT_K, help=f"Chunks to retrieve (default: {DEFAULT_K})")
    parser.add_argument(
        "--max-distance", type=float, default=DEFAULT_MAX_DISTANCE,
        help=f"Relevance threshold: drop chunks with a larger Chroma distance (default: {DEFAULT_MAX_DISTANCE})",
    )
    parser.add_argument(
        "--max-prompt-chars", type=int, default=DEFAULT_MAX_PROMPT_CHARS,
        help=f"Budget for the whole prompt: system prompt, question and evidence "
             f"(default: {DEFAULT_MAX_PROMPT_CHARS})",
    )
    parser.add_argument(
        "--filter", dest="filters", action="append", default=[], type=parse_filter,
        metavar="KEY[:OP]=VALUE",
        help="Keep only chunks whose metadata matches; same syntax as `retrieve --filter`. Repeatable.",
    )
    parser.add_argument("--document-id", help="Search only this document, e.g. Web/JavaScript/Guide/Closures")
    parser.add_argument("--contains", metavar="TEXT", help="Keep only chunks whose text contains TEXT")
    parser.add_argument(
        "--log", type=Path, default=DEFAULT_LOG_PATH,
        help=f"JSONL file to append one record per question to (default: {DEFAULT_LOG_PATH})",
    )
    parser.add_argument("--no-log", action="store_true", help="Do not write the query log")
    args = parser.parse_args(argv)

    if args.k <= 0:
        parser.error("-k must be greater than zero.")
    if args.max_distance < 0:
        parser.error("--max-distance must be non-negative.")
    if args.max_prompt_chars <= 0:
        parser.error("--max-prompt-chars must be greater than zero.")

    client = chromadb.PersistentClient(path=str(args.persist_directory))
    try:
        collection = client.get_collection(name=args.collection)
    except Exception as exc:
        parser.error(f"Could not open collection {args.collection!r}: {exc}")

    filters = list(args.filters)
    if args.document_id:
        filters.append({"document_id": args.document_id})
    where = build_where(filters)
    where_document = {"$contains": args.contains} if args.contains is not None else None

    for query in args.queries:
        try:
            result = rag(
                query, collection, k=args.k, max_distance=args.max_distance,
                max_prompt_chars=args.max_prompt_chars,
                log_path=None if args.no_log else args.log,
                where=where, where_document=where_document,
            )
        except (ValueError, ChromaError) as exc:
            parser.error(str(exc))
        print_result(query, result)


if __name__ == "__main__":
    main()
