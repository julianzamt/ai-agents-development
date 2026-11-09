"""Chunk a text file, or every .md/.txt file under a directory, then embed and store it in a local ChromaDB collection.

To inspect the chunks without embedding anything, use the `chunk` command.
"""

import argparse
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError

from .chunk_cli import add_chunking_arguments, load_chunks
from .embed import collection_embedding, embedding_metadata, get_embedding_provider

UPSERT_BATCH = 1000  # stay well below ChromaDB's maximum batch size


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_chunking_arguments(parser)
    parser.add_argument(
        "--persist-directory", type=Path, default=Path("chroma_db"),
        help="ChromaDB persistence directory (default: ./chroma_db)",
    )
    parser.add_argument("--collection", default="documents", help="ChromaDB collection name")
    parser.add_argument("--print-chunks", action="store_true", help="Print chunk metadata and text")
    args = parser.parse_args(argv)

    loaded = load_chunks(parser, args)
    chunks = loaded.chunks
    if args.print_chunks:
        for chunk in chunks:
            print(f"\n{'=' * 72}\n{chunk.chunk_id} | {len(chunk.text)} characters")
            print(f"Section: {chunk.section_id} | {chunk.section_title}")
            print(f"Source: {chunk.source} | Ingested: {chunk.ingested_at.isoformat()}")
            print(f"{'-' * 72}\n{chunk.text}")

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
        f"Saved {len(chunks)} chunks{loaded.files_note} and embeddings to {args.persist_directory} "
        f"(collection: {args.collection})\n"
        f"Embedding: {provider.name} / {provider.model}"
    )


if __name__ == "__main__":
    main()
