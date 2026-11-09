"""Prepare text for ingestion: chunking. No embedding provider or network needed."""

import re
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Literal


def chunk_by_characters(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into fixed-size, overlapping character chunks without respecting sentence boundaries."""
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks

def chunk_by_paragraphs(text: str, max_chars: int = 500) -> list[str]:
    """Group paragraphs separated by blank lines up to the maximum size."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 1 <= max_chars:
            current = (current + "\n\n" + paragraph).strip()
        else:
            if current:
                chunks.append(current)
            current = paragraph
    if current:
        chunks.append(current)
    return chunks

# Other chunking methods: https://medium.com/the-ai-forum/semantic-chunking-for-rag-f4733025d5f5

# --- Markdown strategy ---------------------------------------------------------
#
# Heading-aware chunking for Markdown. Text is parsed into blocks (paragraphs/lists,
# fenced code, tables, headings), grouped into sections at "##" and "###" headings, and
# every chunk is prefixed with its heading path ("Title > H2 > H3") so it keeps its
# context. Code fences and tables are never cut; tiny sections are merged with a
# neighbour; oversized ones are split between blocks.

FRONT_MATTER_RE = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.S)
FENCE_OPEN_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$")
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
SECTION_LEVELS = (2, 3)  # headings that start a new section


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Return the simple `key: value` front matter fields and the text without that block."""
    match = FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    fields = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip():
            fields[key.strip()] = value.strip()
    return fields, text[match.end():].lstrip("\n")


@dataclass
class _Block:
    kind: str  # "prose" | "code" | "table" | "heading" (a #### or deeper heading) | "section" (a ## or ###)
    text: str
    level: int = 0


@dataclass
class _Section:
    path: tuple[str, ...]
    level: int  # 1 for the lead text before the first ## heading
    blocks: list[_Block]

    @property
    def size(self) -> int:
        return len(_join(self.blocks))


def _join(blocks) -> str:
    return "\n\n".join(block.text for block in blocks)


def _parse_blocks(text: str) -> list[_Block]:
    """Split Markdown into blocks. Blank lines inside a fence do not split it."""
    blocks: list[_Block] = []
    current: list[str] = []
    kind = "prose"
    fence: str | None = None

    def flush():
        nonlocal current
        if current:
            blocks.append(_Block(kind, "\n".join(current).strip("\n")))
        current = []

    for line in text.split("\n"):
        if fence is not None:
            current.append(line)
            stripped = line.strip()
            if stripped and set(stripped) == {fence[0]} and len(stripped) >= len(fence):
                flush()
                fence = None
            continue
        opening = FENCE_OPEN_RE.match(line)
        heading = HEADING_RE.match(line)
        is_table_line = line.lstrip().startswith("|")
        if opening:
            flush()
            kind, fence = "code", opening.group(1)
            current.append(line)
        elif heading:
            flush()
            level = len(heading.group(1))
            if level in SECTION_LEVELS:
                blocks.append(_Block("section", heading.group(2), level))
            else:
                blocks.append(_Block("heading", line.strip(), level))
        elif not line.strip():
            flush()
        else:
            wanted = "table" if is_table_line else "prose"
            if kind != wanted:
                flush()
            kind = wanted
            current.append(line)
    flush()  # an unclosed fence is kept whole
    return [block for block in blocks if block.text.strip()]


def _split_sections(blocks: list[_Block], title: str | None) -> list[_Section]:
    root = (title,) if title else ()
    sections = [_Section(root, 1, [])]
    h2: str | None = None
    for block in blocks:
        if block.kind != "section":
            sections[-1].blocks.append(block)
            continue
        if block.level == 2:
            h2 = block.text
            path = (*root, h2)
        else:
            path = (*root, h2, block.text) if h2 else (*root, block.text)
        sections.append(_Section(path, block.level, []))
    return [section for section in sections if section.blocks]


def _related(earlier: _Section, later: _Section) -> bool:
    """Siblings, or `later` nested inside `earlier`: safe to merge without mixing topics."""
    return later.path[:-1] == earlier.path[:-1] or later.path[:len(earlier.path)] == earlier.path


def _merge(first: _Section, second: _Section) -> _Section:
    """Concatenate two sections; the second one's heading is kept inline in the body."""
    heading = _Block("heading", "#" * second.level + " " + second.path[-1], second.level)
    return _Section(first.path, first.level, [*first.blocks, heading, *second.blocks])


def _fits(section: _Section, max_chars: int) -> bool:
    """Whether the section would be a single chunk, counting its heading-path prefix."""
    return len(" > ".join(section.path)) + 2 + section.size <= max_chars


def _merge_small_sections(sections: list[_Section], min_chars: int, max_chars: int) -> list[_Section]:
    merged: list[_Section] = []
    index = 0
    while index < len(sections):
        section = sections[index]
        following = sections[index + 1] if index + 1 < len(sections) else None
        if section.size < min_chars:
            if (following is not None and _related(section, following)
                    and _fits(_merge(section, following), max_chars)):
                sections[index + 1] = _merge(section, following)  # may still be small: merges again
                index += 1
                continue
            if merged and _related(merged[-1], section) and _fits(_merge(merged[-1], section), max_chars):
                merged[-1] = _merge(merged[-1], section)
                index += 1
                continue
        merged.append(section)
        index += 1
    return merged


def _split_oversized_prose(text: str, limit: int) -> list[str]:
    """Split a long paragraph or list: by line first, then by sentence, then on whitespace."""
    pieces: list[str] = []
    for line in text.split("\n"):
        if len(line) <= limit:
            pieces.append(line)
            continue
        for sentence in SENTENCE_END_RE.split(line):
            while len(sentence) > limit:
                cut = sentence.rfind(" ", 0, limit)
                if cut <= 0:
                    cut = limit
                pieces.append(sentence[:cut])
                sentence = sentence[cut:].lstrip()
            pieces.append(sentence)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        joined = f"{current}\n{piece}" if current else piece
        if current and len(joined) > limit:
            chunks.append(current)
            current = piece
        else:
            current = joined
    if current:
        chunks.append(current)
    return chunks


def _split_oversized_table(text: str, limit: int) -> list[str]:
    """Split a Markdown table between rows, repeating its header and separator in every piece."""
    lines = text.split("\n")
    if len(lines) < 4:
        return [text]
    header, rows = "\n".join(lines[:2]), lines[2:]
    chunks: list[str] = []
    current: list[str] = []
    for row in rows:
        if current and len(header) + sum(len(r) + 1 for r in current) + len(row) > limit:
            chunks.append(header + "\n" + "\n".join(current))
            current = []
        current.append(row)
    chunks.append(header + "\n" + "\n".join(current))
    return chunks


def _build_units(blocks: list[_Block], limit: int) -> list[str]:
    """Blocks glued into indivisible units: a deep heading stays with what follows it, and
    code or a table stays with the paragraph that introduces it (when the pair fits)."""
    units: list[tuple[str, str]] = []  # (kind of the last block, text)
    carry = ""
    for block in blocks:
        if block.kind == "heading":
            carry = f"{carry}\n\n{block.text}".strip()
            continue
        text = f"{carry}\n\n{block.text}".strip() if carry else block.text
        carry = ""
        if (block.kind in ("code", "table") and units and units[-1][0] == "prose"
                and len(units[-1][1]) + 2 + len(text) <= limit):
            units[-1] = (block.kind, f"{units[-1][1]}\n\n{text}")
        else:
            units.append((block.kind, text))
    if carry:
        units.append(("prose", carry))
    return [text for _, text in units]


def _pack(units: list[str], target: int, min_chars: int, max_chars: int) -> list[str]:
    """Greedily pack units toward `target`; a chunk under `min_chars` may grow up to `max_chars`."""
    chunks: list[str] = []
    current = ""
    for unit in units:
        if not current:
            current = unit
            continue
        joined = f"{current}\n\n{unit}"
        if len(joined) <= target or (len(current) < min_chars and len(joined) <= max_chars):
            current = joined
        else:
            chunks.append(current)
            current = unit
    if current:
        if chunks and len(current) < min_chars and len(chunks[-1]) + 2 + len(current) <= max_chars:
            chunks[-1] = f"{chunks[-1]}\n\n{current}"
        else:
            chunks.append(current)
    return chunks


def chunk_by_markdown(
    text: str, target_chars: int = 1200, max_chars: int = 2000, min_chars: int = 300,
) -> list[tuple[int, str, str]]:
    """Return (section_index, heading_path, chunk_text) triples for a Markdown document.

    A section that fits in max_chars is one chunk. Larger ones are packed toward
    target_chars between blocks; code fences and tables are never cut (a table over the
    limit is split between rows, a code block over it stays whole). Sections under
    min_chars are merged into a related neighbour. Each chunk starts with its heading path.
    """
    fields, body = parse_front_matter(text)
    sections = _split_sections(_parse_blocks(body), fields.get("title"))
    sections = _merge_small_sections(sections, min_chars, max_chars)

    result: list[tuple[int, str, str]] = []
    for section_index, section in enumerate(sections, start=1):
        heading_path = " > ".join(section.path)
        prefix = f"{heading_path}\n\n" if heading_path else ""
        section_text = _join(section.blocks)
        if len(prefix) + len(section_text) <= max_chars:
            bodies = [section_text]
        else:
            room = max(max_chars - len(prefix), 1)
            units: list[str] = []
            for unit in _build_units(section.blocks, room):
                if len(unit) <= room or "```" in unit or "~~~" in unit:
                    units.append(unit)  # code stays whole, even when over the limit
                elif unit.lstrip().startswith("|"):
                    units.extend(_split_oversized_table(unit, room))
                else:
                    units.extend(_split_oversized_prose(unit, room))
            bodies = _pack(units, max(target_chars - len(prefix), 1), max(min_chars - len(prefix), 0), room)
        result.extend((section_index, heading_path, prefix + chunk_body) for chunk_body in bodies)
    return result


@dataclass
class Chunk:
    text: str
    document_id: str
    position: int # chunk order within the document
    source: str
    section_id: str | None = None
    section_title: str | None = None
    ingested_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    chunk_id: str = field(init=False)

    def __post_init__(self):
        # Deterministic ID based on document ID, position, and content.
        content_hash = hashlib.sha256(self.text.encode()).hexdigest()[:8]
        self.chunk_id = f"{self.document_id}-{self.position:03d}-{content_hash}"

def build_chunks(
    text: str,
    document_id: str,
    source: str,
    chunk_size: int = 300,
    overlap: int = 50,
    strategy: Literal["characters", "paragraphs", "markdown"] = "characters",
    max_chars: int | None = None,
    target_chars: int = 1200,
    min_chars: int = 300,
) -> list[Chunk]:
    """Build section-aware chunks using the selected strategy.

    characters: chunk_size and overlap apply; numbered headings start new sections.
    paragraphs: max_chars (default 500) applies, and paragraphs stay intact even when
    longer; numbered headings start new sections.
    markdown: heading-aware chunks (see chunk_by_markdown); max_chars defaults to 2000,
    target_chars and min_chars apply, and the front matter is read, not embedded.
    Chunks never cross section boundaries.
    """
    if strategy not in ("characters", "paragraphs", "markdown"):
        raise ValueError(f"Unknown chunking strategy: {strategy!r}")

    if strategy == "markdown":
        max_chars = 2000 if max_chars is None else max_chars
        if not (0 < min_chars <= target_chars <= max_chars):
            raise ValueError("Markdown chunking requires 0 < min_chars <= target_chars <= max_chars.")
        chunks: list[Chunk] = []
        for section_index, heading_path, chunk_text in chunk_by_markdown(text, target_chars, max_chars, min_chars):
            chunks.append(Chunk(
                text=chunk_text,
                document_id=document_id,
                position=len(chunks),
                source=source,
                section_id=f"section_{section_index:02d}",
                section_title=heading_path or "Document",
            ))
        return chunks
    max_chars = 500 if max_chars is None else max_chars

    heading_pattern = re.compile(r"(?m)^(\d+)\.\s+(.+?)\s*$")
    headings = list(heading_pattern.finditer(text))
    sections: list[tuple[str, str, str]] = []
    if headings:
        if text[:headings[0].start()].strip():
            sections.append(("introduction", "Introduction", text[:headings[0].start()].strip()))
        for index, heading in enumerate(headings):
            end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
            section_id = f"section_{int(heading.group(1)):02d}"
            section_title = heading.group(2).strip()
            section_text = text[heading.start():end].strip()
            sections.append((section_id, section_title, section_text))
    else:
        sections.append(("section_00", "Document", text))

    chunks: list[Chunk] = []
    for section_id, section_title, section_text in sections:
        if strategy == "characters":
            texts = chunk_by_characters(section_text, chunk_size, overlap)
        else:
            texts = chunk_by_paragraphs(section_text, max_chars=max_chars)
        position_offset = len(chunks)
        chunks.extend([
            Chunk(
                text=chunk_text,
                document_id=document_id,
                position=position_offset + offset,
                source=source,
                section_id=section_id,
                section_title=section_title,
            )
            for offset, chunk_text in enumerate(texts)
        ])
    return chunks
