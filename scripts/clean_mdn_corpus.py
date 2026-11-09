#!/usr/bin/env python3
"""Clean the MDN JavaScript Guide Markdown in corpus/raw/ and write it to corpus/clean/.

Fenced code blocks are copied byte for byte; every rule applies to prose only, and
inline code spans are masked while rules run. Only .md files are written (images
carry no text).

Usage: python scripts/clean_mdn_corpus.py [--src corpus/raw] [--dst corpus/clean]
"""

import argparse
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})")  # any indent: fences nest inside list items
FRONT_MATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.S)
INLINE_CODE_RE = re.compile(r"(`+)(?!`).+?(?<!`)\1(?!`)", re.S)
MASK = "\x00{}\x00"
MASK_RE = re.compile(r"\x00(\d+)\x00")

# Macros that only render navigation or live demos: the whole macro (and its line) goes.
DROP_MACROS = ("PreviousNext", "Previous", "Next", "EmbedLiveSample", "InteractiveExample")
DROP_MACRO_LINE_RE = re.compile(r"^[ \t]*\{\{(?:%s)\([^}]*\)\}\}[ \t]*\n?" % "|".join(DROP_MACROS), re.M)
MACRO_RE = re.compile(r"\{\{\s*(\w+)\s*(?:\((.*?)\))?\s*\}\}", re.S)
MACRO_ARG_RE = re.compile(r'"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\'')

TABLE_RE = re.compile(r"<table\b.*?</table>", re.S | re.I)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->\s*?", re.S)
# Only known HTML tags are stripped, so prose like "a < b" or <T> placeholders survive.
HTML_TAG_RE = re.compile(
    r"</?(?:p|div|span|a|em|strong|b|i|code|br|sup|sub|ul|ol|li|dl|dt|dd|table|caption|thead|tbody|tr|th|td)\b[^>]*>",
    re.I,
)
IMAGE_RE = re.compile(r"!\[\s*([^\]]*?)\s*\]\([^)]*\)", re.S)
LINK_RE = re.compile(r"\[([^\]]+)\]\((?:[^()\s]|\([^)]*\))+(?:\s+\"[^\"]*\")?\)")
# Sections that are only link lists: after links are flattened they are bare titles, so they only add noise.
DROP_SECTION_RE = re.compile(r"^(#{1,6})[ \t]+see also[ \t]*$", re.I)
# Code blocks MDN marks "hidden" are demo scaffolding the reader never sees.
HIDDEN_FENCE_RE = re.compile(r"^[ \t]*(?:`{3,}|~{3,})[ \t]*\w[^\n]*\bhidden\b")
HEADING_LEVEL_RE = re.compile(r"^(#{1,6})[ \t]")
ALERT_RE = re.compile(r"^(\s*>\s*)\[!(NOTE|WARNING|TIP|CAUTION|IMPORTANT)\][ \t]*$", re.M | re.I)


# --- splitting into code fences and prose ------------------------------------

def split_fences(text: str) -> list[tuple[bool, str]]:
    """Return (is_code, text) segments covering the input exactly."""
    segments: list[tuple[bool, str]] = []
    buffer: list[str] = []
    fence: str | None = None
    for line in text.splitlines(keepends=True):
        match = FENCE_RE.match(line)
        if fence is None and match:
            if buffer:
                segments.append((False, "".join(buffer)))
            buffer, fence = [line], match.group(1)
        elif fence is not None:
            buffer.append(line)
            closing = FENCE_RE.match(line)
            if closing and closing.group(1)[0] == fence[0] and len(closing.group(1)) >= len(fence) and not line.strip(" \n`~"):
                segments.append((True, "".join(buffer)))
                buffer, fence = [], None
        else:
            buffer.append(line)
    if buffer:
        segments.append((fence is not None, "".join(buffer)))
    return segments


def drop_sections(text: str, stats: dict[str, int]) -> str:
    """Remove "See also" sections: the heading and everything up to the next heading of the same or higher level."""
    kept: list[str] = []
    dropping_level = 0
    for is_code, segment in split_fences(text):
        if is_code:
            if not dropping_level:
                kept.append(segment)
            continue
        for line in segment.splitlines(keepends=True):
            heading = HEADING_LEVEL_RE.match(line)
            if dropping_level and heading and len(heading.group(1)) <= dropping_level:
                dropping_level = 0
            if not dropping_level and DROP_SECTION_RE.match(line):
                dropping_level = len(HEADING_LEVEL_RE.match(line).group(1))
                stats["sections_dropped"] = stats.get("sections_dropped", 0) + 1
            if not dropping_level:
                kept.append(line)
    return "".join(kept)


def drop_hidden_blocks(text: str, stats: dict[str, int]) -> str:
    """Remove fenced code blocks whose info string contains the word "hidden"."""
    kept = []
    for is_code, segment in split_fences(text):
        if is_code and HIDDEN_FENCE_RE.match(segment):
            stats["hidden_blocks_dropped"] = stats.get("hidden_blocks_dropped", 0) + 1
            continue
        kept.append(segment)
    return "".join(kept)


def drop_noise(text: str, stats: dict[str, int]) -> str:
    """Structural removals applied before any prose rule: "See also" sections and hidden code blocks."""
    return drop_hidden_blocks(drop_sections(text, stats), stats)


# --- macros ------------------------------------------------------------------

def _macro_args(raw: str | None) -> list[str]:
    return [a if a is not None else b for a, b in MACRO_ARG_RE.findall(raw or "")]


def _replace_macro(match: re.Match) -> str:
    name, args = match.group(1), _macro_args(match.group(2))
    lname = name.lower()
    if lname in ("jsxref", "domxref"):
        if not args:
            return ""
        label = args[1] if len(args) > 1 else args[0].replace("/", ".")
        return f"`{label}`"
    if lname == "glossary":
        return (args[1] if len(args) > 1 else args[0]) if args else ""
    if lname == "htmlelement":
        return f"`<{args[0]}>`" if args else ""
    if lname in ("httpmethod", "httpheader"):
        return f"`{args[0]}`" if args else ""
    if lname == "optional_inline":
        return "(optional)"
    return match.group(0)  # unknown macro: leave it so the report flags it


# --- HTML tables -> Markdown tables ------------------------------------------

class _TableParser(HTMLParser):
    """Collect caption and rows; inline formatting inside cells becomes Markdown."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.caption = ""
        self.rows: list[list[tuple[bool, str]]] = []
        self._cell: list[str] | None = None
        self._cell_is_header = False
        self._row: list[tuple[bool, str]] | None = None
        self._in_caption = False
        self._caption: list[str] = []
        self._code_depth = 0

    def _sink(self) -> list[str] | None:
        return self._caption if self._in_caption else self._cell

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell, self._cell_is_header = [], tag == "th"
        elif tag == "caption":
            self._in_caption = True
        elif (sink := self._sink()) is not None:
            if tag == "code":
                self._code_depth += 1
                sink.append("`")
            elif tag in ("strong", "b"):
                sink.append("**")
            elif tag in ("em", "i"):
                sink.append("*")
            elif tag in ("br", "p", "li", "div"):
                sink.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append((self._cell_is_header, self._cell_text()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None
        elif tag == "caption":
            self._in_caption = False
            self.caption = re.sub(r"\s+", " ", "".join(self._caption)).strip()
        elif (sink := self._sink()) is not None:
            if tag == "code":
                self._code_depth = max(0, self._code_depth - 1)
                sink.append("`")
            elif tag in ("strong", "b"):
                sink.append("**")
            elif tag in ("em", "i"):
                sink.append("*")
            elif tag in ("p", "li", "div"):
                sink.append(" ")

    def handle_data(self, data):
        if (sink := self._sink()) is not None:
            sink.append(data)

    def _cell_text(self) -> str:
        text = re.sub(r"\s+", " ", "".join(self._cell or [])).strip()
        return text.replace("|", "\\|")


def html_table_to_markdown(html: str) -> str:
    parser = _TableParser()
    parser.feed(html)
    rows = [row for row in parser.rows if row]
    if not rows:
        return parser.caption
    width = max(len(row) for row in rows)
    header = rows[0] if all(is_header for is_header, _ in rows[0]) else None
    body = rows[1:] if header else rows
    head_cells = [text for _, text in header] if header else [""] * width

    def line(cells: list[str]) -> str:
        return "| " + " | ".join(cells + [""] * (width - len(cells))) + " |"

    out = [line(head_cells), "| " + " | ".join(["---"] * width) + " |"]
    out += [line([text for _, text in row]) for row in body]
    table = "\n".join(out)
    return f"**{parser.caption}**\n\n{table}" if parser.caption else table


# --- prose cleaning ----------------------------------------------------------

def clean_prose(text: str, stats: dict[str, int]) -> str:
    masked: list[str] = []

    def mask(match: re.Match) -> str:
        masked.append(match.group(0))
        return MASK.format(len(masked) - 1)

    def count(rule: str, pattern: re.Pattern, value: str) -> str:
        stats[rule] = stats.get(rule, 0) + len(pattern.findall(value))
        return value

    text = text.replace(" ", " ").replace("\r\n", "\n")
    text = INLINE_CODE_RE.sub(mask, text)

    count("html_comments", HTML_COMMENT_RE, text)
    text = HTML_COMMENT_RE.sub("", text)

    count("macros_dropped", DROP_MACRO_LINE_RE, text)
    text = DROP_MACRO_LINE_RE.sub("", text)
    count("macros_resolved", MACRO_RE, text)
    text = MACRO_RE.sub(_replace_macro, text)

    # Table cells hold macros-turned-backticks and HTML; convert after macros, before tag stripping.
    count("tables", TABLE_RE, text)
    text = TABLE_RE.sub(lambda m: html_table_to_markdown(m.group(0)), text)

    count("images", IMAGE_RE, text)
    text = IMAGE_RE.sub(lambda m: f"[Image: {' '.join(m.group(1).split())}]" if m.group(1) else "", text)
    count("links", LINK_RE, text)
    text = LINK_RE.sub(r"\1", text)

    count("html_tags", HTML_TAG_RE, text)
    text = HTML_TAG_RE.sub("", text)

    text = ALERT_RE.sub(lambda m: f"{m.group(1)}**{m.group(2).capitalize()}:**", text)

    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return MASK_RE.sub(lambda m: masked[int(m.group(1))], text)


def clean_document(raw: str) -> tuple[str, dict[str, int]]:
    stats: dict[str, int] = {}
    front = ""
    match = FRONT_MATTER_RE.match(raw)
    if match:
        fields = dict(
            line.split(": ", 1) for line in match.group(1).splitlines() if ": " in line
        )
        kept = [f"{key}: {fields[key]}" for key in ("title", "slug") if key in fields]
        front = "---\n" + "\n".join(kept) + "\n---\n\n"
        raw = raw[match.end():]

    raw = drop_noise(raw, stats)
    parts = [text if is_code else clean_prose(text, stats) for is_code, text in split_fences(raw)]
    body = "".join(parts)
    body = re.sub(r"\n{3,}", "\n\n", body).strip("\n")
    return front + body + "\n", stats


# --- checks ------------------------------------------------------------------

LEFTOVERS = {
    "macro": re.compile(r"\{\{"),
    "html tag": re.compile(r"</?(?:table|thead|tbody|tr|td|th|p|div|em|strong|code|br|a|sup|li)\b", re.I),
    "html comment": re.compile(r"<!--"),
    "markdown link": re.compile(r"\]\(\S"),
    "see also section": re.compile(r"^#{1,6}[ \t]+see also[ \t]*$", re.I | re.M),
}


def leftovers(text: str) -> list[str]:
    """Prose problems that should not survive cleaning (code fences and inline code are exempt)."""
    problems = []
    for is_code, segment in split_fences(text):
        if is_code:
            if HIDDEN_FENCE_RE.match(segment):
                problems.append("hidden code block")
            continue
        segment = INLINE_CODE_RE.sub("", segment)
        for name, pattern in LEFTOVERS.items():
            if pattern.search(segment):
                problems.append(name)
    return sorted(set(problems))


def code_blocks(text: str) -> list[str]:
    return [segment for is_code, segment in split_fences(text) if is_code]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=ROOT / "corpus" / "raw")
    parser.add_argument("--dst", type=Path, default=ROOT / "corpus" / "clean")
    args = parser.parse_args(argv)

    files = sorted(args.src.rglob("*.md"))
    if not files:
        parser.error(f"No Markdown files under {args.src}")

    totals: dict[str, int] = {}
    failures = 0
    for path in files:
        raw = path.read_text(encoding="utf-8")
        cleaned, stats = clean_document(raw)
        for rule, n in stats.items():
            totals[rule] = totals.get(rule, 0) + n

        problems = leftovers(cleaned)
        if code_blocks(drop_noise(raw, {})) != code_blocks(cleaned):
            problems.append("code fences changed")
        if problems:
            failures += 1
            print(f"FAIL {path.relative_to(args.src)}: {', '.join(problems)}", file=sys.stderr)

        out = args.dst / path.relative_to(args.src)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(cleaned, encoding="utf-8")

    print(f"Cleaned {len(files)} files -> {args.dst}")
    for rule, n in sorted(totals.items()):
        print(f"  {rule}: {n}")
    if failures:
        print(f"{failures} file(s) failed the checks", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
