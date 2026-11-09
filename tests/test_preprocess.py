"""Chunking tests; no embedding provider involved."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from ia_agents.rag import chunk_cli
from ia_agents.rag.preprocess import build_chunks, chunk_by_markdown, parse_front_matter

CLEAN_CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "clean"


class PreprocessTests(unittest.TestCase):
    def test_unknown_strategy(self):
        with self.assertRaisesRegex(ValueError, "Unknown chunking strategy"):
            build_chunks("text", "doc", "source", strategy="unknown")

    def test_chunks_do_not_cross_section_boundaries(self):
        text = "Intro text.\n\n1. First section\nAlpha beta gamma.\n\n2. Second section\nDelta epsilon zeta."
        chunks = build_chunks(
            text, "doc", "fixture", chunk_size=100, overlap=20,
            strategy="characters",
        )
        self.assertEqual([chunk.section_id for chunk in chunks], ["introduction", "section_01", "section_02"])
        self.assertEqual([chunk.section_title for chunk in chunks], [
            "Introduction", "First section", "Second section",
        ])
        self.assertTrue(all("Delta epsilon zeta" not in chunk.text for chunk in chunks[:2]))

    def test_unheaded_text_gets_document_section_metadata(self):
        chunks = build_chunks("Plain text without headings.", "doc", "fixture")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].section_id, "section_00")
        self.assertEqual(chunks[0].section_title, "Document")


def markdown_chunks(text: str, **sizes) -> list[str]:
    return [chunk for _, _, chunk in chunk_by_markdown(text, **sizes)]


def fence_lines(text: str) -> int:
    return sum(1 for line in text.split("\n") if line.lstrip().startswith(("```", "~~~")))


class MarkdownChunkingTests(unittest.TestCase):
    SMALL = dict(target_chars=120, max_chars=200, min_chars=30)

    def test_front_matter_is_read_but_not_embedded(self):
        text = "---\ntitle: Closures\nslug: Web/Guide/Closures\n---\n\nBody text."
        fields, body = parse_front_matter(text)
        self.assertEqual(fields, {"title": "Closures", "slug": "Web/Guide/Closures"})
        self.assertEqual(body, "Body text.")
        self.assertEqual(markdown_chunks(text), ["Closures\n\nBody text."])

    def test_sections_split_at_h2_and_h3_with_heading_path(self):
        text = (
            "---\ntitle: Guide\n---\n\n"
            "## Alpha\n\n" + "a" * 60 + "\n\n"
            "### Beta\n\n" + "b" * 60 + "\n\n"
            "## Gamma\n\n" + "c" * 60
        )
        chunks = chunk_by_markdown(text, target_chars=100, max_chars=100, min_chars=10)
        self.assertEqual([path for _, path, _ in chunks], ["Guide > Alpha", "Guide > Alpha > Beta", "Guide > Gamma"])
        self.assertTrue(all(chunk.startswith(path + "\n\n") for _, path, chunk in chunks))

    def test_h4_headings_stay_inside_the_section(self):
        text = "## A\n\n" + "x" * 50 + "\n\n#### Deep\n\n" + "y" * 50
        chunks = chunk_by_markdown(text, target_chars=500, max_chars=500, min_chars=10)
        self.assertEqual(len(chunks), 1)
        self.assertIn("#### Deep", chunks[0][2])

    def test_deep_heading_is_not_stranded_at_the_end_of_a_chunk(self):
        text = "## A\n\n" + "x" * 90 + "\n\n#### Deep\n\n" + "y" * 90
        for chunk in markdown_chunks(text, target_chars=120, max_chars=140, min_chars=10):
            self.assertFalse(chunk.rstrip().endswith("#### Deep"), chunk)

    def test_fence_is_never_split_and_hides_headings_and_blank_lines(self):
        code = "```js\nconst a = 1;\n\n## not a heading\n\nconst b = 2;\n```"
        text = "## Code\n\nIntro paragraph.\n\n" + code + "\n\n" + "tail " * 60
        chunks = markdown_chunks(text, target_chars=80, max_chars=120, min_chars=10)
        holder = [chunk for chunk in chunks if "const a = 1;" in chunk]
        self.assertEqual(len(holder), 1)
        self.assertIn(code, holder[0])
        self.assertNotIn("not a heading", "".join(path for _, path, _ in chunk_by_markdown(text, **self.SMALL)))
        self.assertTrue(all(fence_lines(chunk) % 2 == 0 for chunk in chunks))

    def test_oversized_code_block_stays_whole(self):
        code = "```js\n" + "\n".join(f"line_{i}();" for i in range(60)) + "\n```"
        chunks = markdown_chunks("## Big\n\n" + code, **self.SMALL)
        self.assertEqual(sum(code in chunk for chunk in chunks), 1)
        self.assertGreater(max(len(chunk) for chunk in chunks), self.SMALL["max_chars"])

    def test_indented_fence_inside_a_list_is_atomic(self):
        code = "    ```js\n    const x = 1;\n\n    const y = 2;\n    ```"
        text = "## Lists\n\n- item:\n\n" + code + "\n\n" + "tail " * 80
        chunks = markdown_chunks(text, **self.SMALL)
        self.assertTrue(all(fence_lines(chunk) % 2 == 0 for chunk in chunks))
        self.assertEqual(sum("const x = 1;" in chunk and "const y = 2;" in chunk for chunk in chunks), 1)

    def test_code_stays_with_the_paragraph_that_introduces_it(self):
        text = "## Intro\n\n" + "filler " * 20 + "\n\nConsider this example:\n\n```js\nrun();\n```\n\n" + "after " * 20
        chunks = markdown_chunks(text, target_chars=130, max_chars=200, min_chars=20)
        holder = [chunk for chunk in chunks if "run();" in chunk]
        self.assertEqual(len(holder), 1)
        self.assertIn("Consider this example:", holder[0])

    def test_table_is_not_cut_mid_row_and_oversized_table_repeats_header(self):
        rows = [f"| r{i} | meaning of r{i} |" for i in range(30)]
        table = "\n".join(["| Char | Meaning |", "| --- | --- |", *rows])
        chunks = markdown_chunks("## Tbl\n\n" + table, **self.SMALL)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertIn("| Char | Meaning |\n| --- | --- |", chunk)
            self.assertTrue(all(line.startswith("|") for line in chunk.split("\n\n", 1)[1].split("\n")))
        self.assertEqual(sorted(row for chunk in chunks for row in chunk.split("\n") if row.startswith("| r")), sorted(rows))

    def test_small_section_merges_into_its_next_sibling(self):
        text = "## Tiny\n\nshort.\n\n## Normal\n\n" + "n" * 100
        chunks = chunk_by_markdown(text, target_chars=150, max_chars=200, min_chars=30)
        self.assertEqual(len(chunks), 1)
        _, path, chunk = chunks[0]
        self.assertEqual(path, "Tiny")
        self.assertIn("short.", chunk)
        self.assertIn("## Normal", chunk)  # the merged section keeps its own heading inline

    def test_trailing_small_section_merges_into_the_previous_sibling(self):
        text = "## Normal\n\n" + "n" * 100 + "\n\n## See also\n\nlinks"
        chunks = markdown_chunks(text, target_chars=150, max_chars=200, min_chars=30)
        self.assertEqual(len(chunks), 1)
        self.assertIn("links", chunks[0])

    def test_small_section_is_not_merged_when_the_result_exceeds_max(self):
        text = "## Tiny\n\nshort.\n\n## Normal\n\n" + "n" * 180
        chunks = chunk_by_markdown(text, target_chars=150, max_chars=200, min_chars=30)
        self.assertEqual([path for _, path, _ in chunks], ["Tiny", "Normal"])

    def test_small_section_is_not_merged_across_unrelated_parents(self):
        text = "## A\n\n### A1\n\n" + "a" * 100 + "\n\n### A2\n\ntiny\n\n## B\n\n### B1\n\n" + "b" * 100
        chunks = chunk_by_markdown(text, target_chars=150, max_chars=400, min_chars=30)
        self.assertNotIn("tiny", "".join(chunk for _, path, chunk in chunks if path.endswith("B1")))

    def test_oversized_prose_is_split_with_the_prefix_repeated(self):
        sentences = " ".join(f"Sentence number {i} says something useful." for i in range(30))
        chunks = markdown_chunks("## Long\n\n" + sentences, **self.SMALL)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.startswith("Long\n\n") and len(chunk) <= self.SMALL["max_chars"] for chunk in chunks))
        self.assertEqual(" ".join(chunk.split("\n\n", 1)[1] for chunk in chunks).split(), sentences.split())

    def test_text_without_headings_or_front_matter(self):
        chunks = build_chunks("Plain paragraph.\n\nAnother one.", "doc", "src", strategy="markdown")
        self.assertEqual([chunk.section_title for chunk in chunks], ["Document"])
        self.assertEqual(chunks[0].text, "Plain paragraph.\n\nAnother one.")

    def test_build_chunks_metadata_and_validation(self):
        text = "---\ntitle: T\n---\n\n## One\n\n" + "x" * 400 + "\n\n## Two\n\n" + "y" * 400
        chunks = build_chunks(text, "doc", "src", strategy="markdown", max_chars=500, target_chars=450, min_chars=50)
        self.assertEqual([c.section_id for c in chunks], ["section_01", "section_02"])
        self.assertEqual([c.section_title for c in chunks], ["T > One", "T > Two"])
        self.assertEqual([c.position for c in chunks], [0, 1])
        with self.assertRaisesRegex(ValueError, "min_chars <= target_chars <= max_chars"):
            build_chunks(text, "doc", "src", strategy="markdown", max_chars=100, target_chars=450)

    def test_chunk_command_uses_the_front_matter_slug_as_document_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dump = Path(tmp, "index.md"), Path(tmp, "out.txt")
            source.write_text("---\ntitle: T\nslug: Web/Guide/T\n---\n\n## One\n\nhello world", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                chunk_cli.main([str(source), "--output", str(dump)])
            report = dump.read_text(encoding="utf-8")
        self.assertIn("Web/Guide/T-000-", report)
        self.assertIn("strategy=markdown", report)

    @unittest.skipUnless(CLEAN_CORPUS.is_dir(), "corpus/clean has not been generated")
    def test_cleaned_corpus_invariants(self):
        files = sorted(CLEAN_CORPUS.rglob("*.md"))
        self.assertTrue(files)
        for path in files:
            text = path.read_text(encoding="utf-8")
            fields, _ = parse_front_matter(text)
            chunks = build_chunks(text, fields["slug"], str(path), strategy="markdown")
            self.assertTrue(chunks, path)
            for chunk in chunks:
                label = f"{path.relative_to(CLEAN_CORPUS)} {chunk.chunk_id}"
                self.assertEqual(fence_lines(chunk.text) % 2, 0, f"cut code fence: {label}")
                self.assertNotIn("page-type:", chunk.text, label)
                self.assertTrue(chunk.text.startswith(fields["title"]), f"missing heading path: {label}")
                if "```" not in chunk.text:
                    self.assertLessEqual(len(chunk.text), 2000, label)


if __name__ == "__main__":
    unittest.main()
