"""Ingest a UTF-8 text file into a local ChromaDB collection."""

import argparse
import statistics
from collections import Counter
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError

from .embed import collection_embedding, embedding_metadata, get_embedding_provider
from .preprocess import build_chunks, clean_text


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
    parser.add_argument("input", type=Path, help="UTF-8 text file")
    parser.add_argument(
        "--persist-directory", type=Path, default=Path("chroma_db"),
        help="ChromaDB persistence directory (default: ./chroma_db)",
    )
    parser.add_argument("--collection", default="documents", help="ChromaDB collection name")
    parser.add_argument("--document-id", help="Document ID (default: input filename without extension)")
    parser.add_argument("--source", help="Source metadata (default: input path)")
    parser.add_argument("--strategy", choices=["characters", "paragraphs"], default="characters")
    parser.add_argument("--chunk-size", type=int, default=300, help="Character chunk size")
    parser.add_argument("--overlap", type=int, default=50, help="Character chunk overlap")
    parser.add_argument("--max-chars", type=int, default=500, help="Paragraph grouping limit; oversized paragraphs stay intact")
    parser.add_argument("--print-chunks", action="store_true", help="Print chunk metadata and text")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Write the chunks to a readable file and stop: no embedding provider, no ChromaDB",
    )
    parser.add_argument(
        "--dump-file", type=Path,
        help="Dry-run output file (default: <document-id>.chunks.txt in the current directory)",
    )
    args = parser.parse_args(argv)

    if args.dump_file is not None and not args.dry_run:
        parser.error("--dump-file requires --dry-run.")
    if args.strategy == "characters" and not (0 <= args.overlap < args.chunk_size):
        parser.error("Require chunk-size > 0 and 0 <= overlap < chunk-size.")
    if args.strategy == "paragraphs" and args.max_chars <= 0:
        parser.error("max-chars must be greater than zero.")
    try:
        text = clean_text(args.input.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        parser.error(str(exc))
    if not text:
        parser.error("Input contains no text after cleaning.")

    chunks = build_chunks(
        text, document_id=args.document_id or args.input.stem,
        source=args.source or str(args.input), strategy=args.strategy,
        chunk_size=args.chunk_size, overlap=args.overlap, max_chars=args.max_chars,
    )
    if args.print_chunks:
        for chunk in chunks:
            print(f"\n{'=' * 72}\n{chunk.chunk_id} | {len(chunk.text)} characters")
            print(f"Section: {chunk.section_id} | {chunk.section_title}")
            print(f"Source: {chunk.source} | Ingested: {chunk.ingested_at.isoformat()}")
            print(f"{'-' * 72}\n{chunk.text}")

    if args.dry_run:
        dump_file = args.dump_file or Path(f"{chunks[0].document_id}.chunks.txt")
        settings = (f"strategy={args.strategy} chunk_size={args.chunk_size} overlap={args.overlap}"
                    if args.strategy == "characters" else f"strategy={args.strategy} max_chars={args.max_chars}")
        report = format_chunks_report(chunks, settings)
        try:
            dump_file.parent.mkdir(parents=True, exist_ok=True)
            dump_file.write_text(report, encoding="utf-8")
        except OSError as exc:
            parser.error(str(exc))
        print(f"Dry run: wrote {len(chunks)} chunks to {dump_file} (nothing embedded or stored)")
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
    collection.upsert(
        ids=[chunk.chunk_id for chunk in chunks],
        embeddings=vectors,
        documents=[chunk.text for chunk in chunks],
        metadatas=[
            {
                "document_id": chunk.document_id,
                "position": chunk.position,
                "source": chunk.source,
                "ingested_at": chunk.ingested_at.isoformat(),
                "section_id": chunk.section_id,
                "section_title": chunk.section_title,
            }
            for chunk in chunks
        ],
    )
    print(
        f"Saved {len(chunks)} chunks and embeddings to {args.persist_directory} "
        f"(collection: {args.collection})\n"
        f"Embedding: {provider.name} / {provider.model}"
    )


if __name__ == "__main__":
    main()
