"""Ingest a UTF-8 text file, or every .md/.txt file under a directory, into a local ChromaDB collection."""

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError

from .embed import collection_embedding, embedding_metadata, get_embedding_provider
from .preprocess import build_chunks, parse_front_matter


UPSERT_BATCH = 1000  # stay well below ChromaDB's maximum batch size
TEXT_SUFFIXES = (".md", ".txt")


def format_chunks_report(chunks, settings: str) -> str:
    """Readable text report: settings, size statistics, then every chunk in order."""
    sizes = [len(chunk.text) for chunk in chunks]
    sections = Counter((chunk.section_id, chunk.section_title) for chunk in chunks)
    lines = [
        f"Settings: {settings}",
        f"Chunks: {len(chunks)}",
        f"Characters per chunk: min={min(sizes)} median={statistics.median(sizes):g} max={max(sizes)}",
        "Chunks per section:",
        *(f"  {section_id} | {title}: {count}" for (section_id, title), count in sections.items()),
    ]
    for chunk in chunks:
        lines += [
            "", "=" * 72,
            f"{chunk.chunk_id} | {len(chunk.text)} characters",
            f"Section: {chunk.section_id} | {chunk.section_title}",
            f"Source: {chunk.source}",
            "-" * 72, chunk.text,
        ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="UTF-8 text file, or a directory searched recursively for .md and .txt files")
    parser.add_argument(
        "--persist-directory", type=Path, default=Path("chroma_db"),
        help="ChromaDB persistence directory (default: ./chroma_db)",
    )
    parser.add_argument("--collection", default="documents", help="ChromaDB collection name")
    parser.add_argument("--document-id", help="Document ID (default: front matter slug for markdown, else input filename without extension); single file only")
    parser.add_argument("--source", help="Source metadata (default: input path); single file only")
    parser.add_argument(
        "--strategy", choices=["characters", "paragraphs", "markdown"], default="markdown",
        help="markdown: heading-aware chunks for Markdown (default); characters/paragraphs: plain text",
    )
    parser.add_argument("--chunk-size", type=int, default=300, help="Character chunk size")
    parser.add_argument("--overlap", type=int, default=50, help="Character chunk overlap")
    parser.add_argument(
        "--max-chars", type=int,
        help="Paragraphs: grouping limit, oversized paragraphs stay intact (default 500). "
             "Markdown: hard cap per chunk, except code blocks, which are never cut (default 2000)",
    )
    parser.add_argument("--target-chars", type=int, default=1200, help="Markdown: size sections over the cap are packed toward")
    parser.add_argument("--min-chars", type=int, default=300, help="Markdown: sections below this are merged with a neighbour")
    parser.add_argument("--print-chunks", action="store_true", help="Print chunk metadata and text")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Write the chunks to a readable file and stop: no embedding provider, no ChromaDB",
    )
    parser.add_argument(
        "--dump-file", type=Path,
        help="Dry-run output file (default: <document-id>.chunks.txt, or <directory>.chunks.txt, in the current directory)",
    )
    args = parser.parse_args(argv)

    if args.dump_file is not None and not args.dry_run:
        parser.error("--dump-file requires --dry-run.")
    if args.strategy == "characters" and not (0 <= args.overlap < args.chunk_size):
        parser.error("Require chunk-size > 0 and 0 <= overlap < chunk-size.")
    if args.max_chars is not None and args.max_chars <= 0:
        parser.error("max-chars must be greater than zero.")
    max_chars = args.max_chars if args.max_chars is not None else (2000 if args.strategy == "markdown" else 500)
    if args.strategy == "markdown" and not (0 < args.min_chars <= args.target_chars <= max_chars):
        parser.error("Require 0 < min-chars <= target-chars <= max-chars.")
    directory_mode = args.input.is_dir()
    if directory_mode:
        if args.document_id or args.source:
            parser.error("--document-id and --source apply to a single file, not a directory.")
        files = sorted(
            path for path in args.input.rglob("*")
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
        )
        if not files:
            parser.error(f"No {' or '.join(TEXT_SUFFIXES)} files under {args.input}.")
    else:
        files = [args.input]

    chunks = []
    documents: dict[str, Path] = {}
    for path in files:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            parser.error(f"{path}: {exc}")
        if not text:
            if directory_mode:
                print(f"Skipping empty file: {path}", file=sys.stderr)
                continue
            parser.error("Input contains no text.")

        document_id = args.document_id or (
            path.relative_to(args.input).with_suffix("").as_posix() if directory_mode else path.stem
        )
        if args.strategy == "markdown" and not args.document_id:
            # Corpus files are all named index.md: the front matter slug identifies the document.
            document_id = parse_front_matter(text)[0].get("slug", document_id)
        if document_id in documents:
            parser.error(f"Document ID {document_id!r} is used by both {documents[document_id]} and {path}.")
        documents[document_id] = path
        chunks.extend(build_chunks(
            text, document_id=document_id,
            source=args.source or str(path), strategy=args.strategy,
            chunk_size=args.chunk_size, overlap=args.overlap, max_chars=max_chars,
            target_chars=args.target_chars, min_chars=args.min_chars,
        ))
    if not chunks:
        parser.error("Input produced no chunks.")
    files_note = f" from {len(documents)} files" if directory_mode else ""
    if args.print_chunks:
        for chunk in chunks:
            print(f"\n{'=' * 72}\n{chunk.chunk_id} | {len(chunk.text)} characters")
            print(f"Section: {chunk.section_id} | {chunk.section_title}")
            print(f"Source: {chunk.source} | Ingested: {chunk.ingested_at.isoformat()}")
            print(f"{'-' * 72}\n{chunk.text}")

    if args.dry_run:
        # Slugs contain "/": never let a document ID create folders.
        default_name = args.input.name if directory_mode else chunks[0].document_id.replace("/", "_")
        dump_file = args.dump_file or Path(f"{default_name}.chunks.txt")
        settings = {
            "characters": f"strategy=characters chunk_size={args.chunk_size} overlap={args.overlap}",
            "paragraphs": f"strategy=paragraphs max_chars={max_chars}",
            "markdown": f"strategy=markdown min_chars={args.min_chars} target_chars={args.target_chars} max_chars={max_chars}",
        }[args.strategy]
        report = format_chunks_report(chunks, f"{settings} files={len(documents)}" if directory_mode else settings)
        try:
            dump_file.parent.mkdir(parents=True, exist_ok=True)
            dump_file.write_text(report, encoding="utf-8")
        except OSError as exc:
            parser.error(str(exc))
        print(f"Dry run: wrote {len(chunks)} chunks{files_note} to {dump_file} (nothing embedded or stored)")
        return

    provider = get_embedding_provider()
    expected = embedding_metadata(provider)
    args.persist_directory.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(args.persist_directory))
    # Check an existing collection before paying for embeddings; vectors from
    # different providers or models must never share a collection.
    try:
        existing = client.get_collection(name=args.collection)
    except NotFoundError:
        existing = None
    if existing is not None:
        try:
            recorded = collection_embedding(existing)
        except ValueError as exc:
            parser.error(str(exc))
        if recorded != expected:
            parser.error(
                f"Collection {args.collection!r} was ingested with {recorded}, but the configured "
                f"embedding is {expected}. Use the same provider and model, or a new collection."
            )

    vectors = provider.embed_documents(
        [chunk.text for chunk in chunks],
        titles=[chunk.section_title for chunk in chunks],
    )
    if len(vectors) != len(chunks):
        raise ValueError("The embedding provider returned a different number of vectors than chunks.")
    # Metadata applies only on creation; Chroma ignores it for existing collections.
    collection = client.get_or_create_collection(name=args.collection, metadata=expected)
    for start in range(0, len(chunks), UPSERT_BATCH):
        batch = slice(start, start + UPSERT_BATCH)
        collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks[batch]],
            embeddings=vectors[batch],
            documents=[chunk.text for chunk in chunks[batch]],
            metadatas=[
                {
                    "document_id": chunk.document_id,
                    "position": chunk.position,
                    "source": chunk.source,
                    "ingested_at": chunk.ingested_at.isoformat(),
                    "section_id": chunk.section_id,
                    "section_title": chunk.section_title,
                }
                for chunk in chunks[batch]
            ],
        )
    print(
        f"Saved {len(chunks)} chunks{files_note} and embeddings to {args.persist_directory} "
        f"(collection: {args.collection})\n"
        f"Embedding: {provider.name} / {provider.model}"
    )


if __name__ == "__main__":
    main()
