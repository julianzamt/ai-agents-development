"""Ingest a UTF-8 text file into a local ChromaDB collection."""

import argparse
from pathlib import Path

import chromadb

from .llm.embed import build_chunks, clean_text, embed_documents


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
    args = parser.parse_args(argv)

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

    vectors = embed_documents(
        [chunk.text for chunk in chunks],
        titles=[chunk.section_title for chunk in chunks],
    )
    if len(vectors) != len(chunks):
        raise ValueError("Google returned a different number of vectors than chunks.")
    args.persist_directory.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(args.persist_directory))
    collection = client.get_or_create_collection(name=args.collection)
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
        f"(collection: {args.collection})"
    )


if __name__ == "__main__":
    main()
