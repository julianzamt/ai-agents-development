"""Chunk a text file, or every .md/.txt file under a directory, and write a readable report.

No embedding provider and no ChromaDB: this is the chunking half of `ingest`, to inspect
what would be embedded.
"""

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

from .preprocess import Chunk, build_chunks, parse_front_matter

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


def add_chunking_arguments(parser: argparse.ArgumentParser) -> None:
    """The input and chunking options shared by `chunk` and `ingest`."""
    parser.add_argument(
        "input", type=Path,
        help="UTF-8 text file, or a directory searched recursively for .md and .txt files",
    )
    parser.add_argument(
        "--document-id",
        help="Document ID (default: front matter slug for markdown, else input filename without extension); single file only",
    )
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


class Loaded(NamedTuple):
    chunks: list[Chunk]
    documents: dict[str, Path]  # document ID -> file
    settings: str
    directory_mode: bool

    @property
    def files_note(self) -> str:
        return f" from {len(self.documents)} files" if self.directory_mode else ""


def load_chunks(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Loaded:
    """Validate the shared options, read the input file(s) and chunk them. Errors go through parser.error."""
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

    chunks: list[Chunk] = []
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

    settings = {
        "characters": f"strategy=characters chunk_size={args.chunk_size} overlap={args.overlap}",
        "paragraphs": f"strategy=paragraphs max_chars={max_chars}",
        "markdown": f"strategy=markdown min_chars={args.min_chars} target_chars={args.target_chars} max_chars={max_chars}",
    }[args.strategy]
    if directory_mode:
        settings += f" files={len(documents)}"
    return Loaded(chunks, documents, settings, directory_mode)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_chunking_arguments(parser)
    parser.add_argument(
        "--output", type=Path,
        help="Report file (default: <directory>.chunks.txt next to the input directory, "
             "or <document-id>.chunks.txt in the current directory for a single file)",
    )
    parser.add_argument("--stdout", action="store_true", help="Print the report instead of writing a file")
    args = parser.parse_args(argv)
    if args.stdout and args.output is not None:
        parser.error("--stdout and --output are mutually exclusive.")

    loaded = load_chunks(parser, args)
    report = format_chunks_report(loaded.chunks, loaded.settings)
    if args.stdout:
        print(report, end="")
        return

    if args.output is not None:
        output = args.output
    elif loaded.directory_mode:
        # Next to the directory, never inside it: ingest reads every .txt file under a directory.
        directory = args.input.resolve()
        output = directory.parent / f"{directory.name}.chunks.txt"
    else:
        # Slugs contain "/": never let a document ID create folders.
        output = Path(f"{loaded.chunks[0].document_id.replace('/', '_')}.chunks.txt")
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
    except OSError as exc:
        parser.error(str(exc))
    print(f"Wrote {len(loaded.chunks)} chunks{loaded.files_note} to {output}")


if __name__ == "__main__":
    main()
