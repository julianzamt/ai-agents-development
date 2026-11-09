"""The chunk command writes a readable report; no embedding provider and no ChromaDB."""

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from ia_agents.rag import chunk_cli
from ia_agents.rag.chunk_cli import main

REPO = Path(__file__).resolve().parent.parent
COMMITTED_REPORT = REPO / "corpus" / "clean.chunks.txt"


def make_corpus(root: Path) -> Path:
    corpus = root / "corpus"
    for name, slug, body in (("a", "Guide/A", "## One\n\nAlpha text."), ("b/c", "Guide/B", "## Two\n\nBeta text.")):
        path = corpus / name / "index.md"
        path.parent.mkdir(parents=True)
        path.write_text(f"---\ntitle: {name}\nslug: {slug}\n---\n\n{body}", encoding="utf-8")
    (corpus / "image.svg").write_text("<svg/>", encoding="utf-8")
    return corpus


class ChunkCliTests(unittest.TestCase):
    def test_does_not_import_the_embedding_or_storage_stack(self):
        self.assertFalse(hasattr(chunk_cli, "chromadb"))
        self.assertFalse(hasattr(chunk_cli, "get_embedding_provider"))

    def test_writes_readable_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "corpus.txt"
            source.write_text("1. First\n\nAlpha paragraph.\n\n2. Second\n\nBeta paragraph.", encoding="utf-8")
            output = root / "out" / "chunks.txt"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                main([str(source), "--output", str(output), "--strategy", "paragraphs", "--document-id", "example"])
            report = output.read_text(encoding="utf-8")
            self.assertIn("Wrote 2 chunks", stdout.getvalue())
            self.assertIn("Chunks: 2", report)
            self.assertIn("Characters per chunk: min=", report)
            self.assertIn("section_01 | First: 1", report)
            self.assertIn("Alpha paragraph.", report)
            self.assertIn("Beta paragraph.", report)

    def test_stdout_prints_the_report_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "corpus.txt"
            source.write_text("A document.", encoding="utf-8")
            work = Path(directory) / "work"
            work.mkdir()
            previous = os.getcwd()
            os.chdir(work)
            try:
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    main([str(source), "--stdout"])
            finally:
                os.chdir(previous)
            self.assertIn("A document.", stdout.getvalue())
            self.assertEqual(list(work.iterdir()), [])

    def test_stdout_and_output_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "corpus.txt"
            source.write_text("A document.", encoding="utf-8")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main([str(source), "--stdout", "--output", str(Path(directory) / "x.txt")])

    def test_directory_writes_one_report_for_all_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus, output = make_corpus(root), root / "all.txt"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                main([str(corpus), "--output", str(output)])
            report = output.read_text(encoding="utf-8")
            self.assertIn("Chunks: 2", report)
            self.assertIn("files=2", report)
            self.assertIn("Guide/A-000-", report)
            self.assertIn("Guide/B-000-", report)
            self.assertIn("from 2 files", stdout.getvalue())

    def test_default_output_name_and_location(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = make_corpus(root)
            work = root / "work"
            work.mkdir()
            previous = os.getcwd()
            os.chdir(work)
            try:
                with redirect_stdout(io.StringIO()):
                    main([str(corpus / "a" / "index.md")])
                    main([str(corpus)])
            finally:
                os.chdir(previous)
            # A single file: slug-safe name in the cwd. A directory: next to it, not inside it.
            self.assertEqual([path.name for path in work.iterdir()], ["Guide_A.chunks.txt"])
            self.assertTrue((root / "corpus.chunks.txt").is_file())
            self.assertFalse(list(corpus.rglob("*.chunks.txt")))

    def test_rejects_bad_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = make_corpus(root)
            empty = root / "empty_dir"
            empty.mkdir()
            duplicate = root / "dup"
            duplicate.mkdir()
            for name in ("one.md", "two.md"):
                (duplicate / name).write_text("---\nslug: Same\n---\n\ntext", encoding="utf-8")
            blank = root / "blank.txt"
            blank.write_text(" \n", encoding="utf-8")
            cases = {
                "document-id with directory": [str(corpus), "--document-id", "x"],
                "source with directory": [str(corpus), "--source", "x"],
                "no files": [str(empty)],
                "duplicate id": [str(duplicate)],
                "empty file": [str(blank)],
                "bad overlap": [str(blank), "--strategy", "characters", "--overlap", "300"],
                "bad sizes": [str(corpus), "--min-chars", "5000"],
            }
            for label, argv in cases.items():
                with self.subTest(label), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(argv)

    def test_skips_empty_files_in_a_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = make_corpus(root)
            (corpus / "empty.md").write_text(" \n", encoding="utf-8")
            stderr = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                main([str(corpus), "--output", str(root / "r.txt")])
            self.assertIn("Skipping empty file", stderr.getvalue())

    @unittest.skipUnless(COMMITTED_REPORT.is_file(), "corpus/clean.chunks.txt has not been generated")
    def test_committed_report_is_up_to_date(self):
        previous = os.getcwd()
        os.chdir(REPO)  # the report records source paths relative to the repository root
        try:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                main(["corpus/clean", "--stdout"])
        finally:
            os.chdir(previous)
        self.assertEqual(
            stdout.getvalue(), COMMITTED_REPORT.read_text(encoding="utf-8"),
            "corpus/clean.chunks.txt is stale: run `chunk corpus/clean` from the repository root and commit it",
        )


if __name__ == "__main__":
    unittest.main()
