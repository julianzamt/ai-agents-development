"""Exercise the ingestion command without external API calls."""

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import Mock, patch

import chromadb
from chromadb.errors import NotFoundError

from ia_agents.rag.ingest_cli import main

EMBEDDING = {"embedding_provider": "google", "embedding_model": "test-model"}


def fake_provider(vectors=None, name="google", model="test-model"):
    """An embedding provider double; by default it returns one vector per text."""
    provider = Mock()
    provider.name, provider.model = name, model
    if vectors is None:
        provider.embed_documents.side_effect = lambda texts, titles=None: [
            [float(i), 0.5] for i, _ in enumerate(texts)
        ]
    else:
        provider.embed_documents.return_value = vectors
    return provider


def patch_provider(provider):
    return patch("ia_agents.rag.ingest_cli.get_embedding_provider", return_value=provider)


class IngestCliTests(unittest.TestCase):
    def test_both_strategies_save_vectors_and_metadata(self):
        for strategy in ("characters", "paragraphs"):
            with self.subTest(strategy=strategy), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "corpus.txt"
                source.write_text("  First paragraph.\n\nSecond paragraph.  ", encoding="utf-8")
                persistence = root / "nested" / "chroma_db"
                stdout = io.StringIO()
                provider = fake_provider()
                with patch_provider(provider), redirect_stdout(stdout):
                    main([str(source), "--persist-directory", str(persistence), "--strategy", strategy,
                          "--chunk-size", "24", "--overlap", "5", "--max-chars", "24",
                          "--document-id", "example", "--source", "fixture", "--print-chunks"])

                client = chromadb.PersistentClient(path=str(persistence))
                collection = client.get_collection("documents")
                self.assertEqual({key: collection.metadata.get(key) for key in EMBEDDING}, EMBEDDING)
                stored = collection.get(include=["documents", "metadatas", "embeddings"])
                self.assertGreater(len(stored["ids"]), 1)
                provider.embed_documents.assert_called_once_with(
                    stored["documents"],
                    titles=[metadata["section_title"] for metadata in stored["metadatas"]],
                )
                self.assertIn(stored["ids"][0], stdout.getvalue())
                self.assertIn("Embedding: google / test-model", stdout.getvalue())
                for i, (chunk_id, text, metadata, vector) in enumerate(zip(
                    stored["ids"], stored["documents"], stored["metadatas"], stored["embeddings"],
                )):
                    self.assertTrue(chunk_id.startswith(f"example-{i:03d}-"))
                    self.assertEqual(metadata["document_id"], "example")
                    self.assertEqual(metadata["source"], "fixture")
                    self.assertEqual(metadata["position"], i)
                    self.assertIn("ingested_at", metadata)
                    self.assertEqual(metadata["section_id"], "section_00")
                    self.assertEqual(metadata["section_title"], "Document")
                    self.assertEqual(list(vector), [float(i), 0.5])
                    self.assertTrue(text)
                if strategy == "paragraphs":
                    self.assertEqual(stored["documents"], ["First paragraph.", "Second paragraph."])

    def test_default_persistence_directory_and_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "corpus.txt"
            source.write_text("A document.", encoding="utf-8")
            persistence = Path(directory) / "chroma_db"
            with patch("ia_agents.rag.ingest_cli.Path", wraps=Path) as path_class, \
                    patch_provider(fake_provider([[1.0, 0.0]])), \
                    patch("ia_agents.rag.ingest_cli.chromadb.PersistentClient") as client_class, \
                    redirect_stdout(io.StringIO()):
                client_class.return_value.get_collection.side_effect = NotFoundError("missing")
                # Exercise parser defaults while redirecting the relative default to a temp folder.
                path_class.side_effect = lambda value=".": persistence if value == "chroma_db" else Path(value)
                main([str(source)])
                client_class.assert_called_once_with(path=str(persistence))
                client_class.return_value.get_or_create_collection.assert_called_once_with(
                    name="documents", metadata=EMBEDDING,
                )

    def test_dry_run_flags_moved_to_the_chunk_command(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "corpus.txt"
            source.write_text("A document.", encoding="utf-8")
            for flag in (["--dry-run"], ["--dump-file", str(Path(directory) / "x.txt")]):
                with self.subTest(flag=flag), patch("ia_agents.rag.ingest_cli.get_embedding_provider") as get_provider:
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        main([str(source), *flag])
                    get_provider.assert_not_called()

    def test_invalid_input_does_not_create_a_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "empty.txt"
            source.write_text(" \n\t", encoding="utf-8")
            for extra in ([], ["--overlap", "300"], ["--strategy", "paragraphs", "--max-chars", "0"]):
                with self.subTest(extra=extra), patch("ia_agents.rag.ingest_cli.get_embedding_provider") as get_provider:
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        main([str(source), *extra])
                    get_provider.assert_not_called()

    def test_vector_count_mismatch_does_not_write_to_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.txt"
            source.write_text("A document.", encoding="utf-8")
            persistence = Path(directory) / "db"
            with patch_provider(fake_provider([])):
                with self.assertRaisesRegex(ValueError, "number of vectors"):
                    main([str(source), "--persist-directory", str(persistence)])
            with self.assertRaises(NotFoundError):
                chromadb.PersistentClient(path=str(persistence)).get_collection("documents")

    def test_reingestion_upserts_same_chunk_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.txt"
            source.write_text("A short document.", encoding="utf-8")
            persistence = root / "db"
            with patch_provider(fake_provider([[1.0, 0.0]])), redirect_stdout(io.StringIO()):
                main([str(source), "--persist-directory", str(persistence)])
                main([str(source), "--persist-directory", str(persistence)])
            collection = chromadb.PersistentClient(path=str(persistence)).get_collection("documents")
            self.assertEqual(collection.count(), 1)

    def test_rejects_collection_from_a_different_embedding_before_embedding(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.txt"
            source.write_text("A short document.", encoding="utf-8")
            persistence = Path(directory) / "db"
            with patch_provider(fake_provider([[1.0, 0.0]])), redirect_stdout(io.StringIO()):
                main([str(source), "--persist-directory", str(persistence)])
            for other in (fake_provider(name="openai", model="test-model"),
                          fake_provider(name="google", model="other-model")):
                with self.subTest(provider=other.name, model=other.model), patch_provider(other):
                    stderr = io.StringIO()
                    with redirect_stderr(stderr), self.assertRaises(SystemExit):
                        main([str(source), "--persist-directory", str(persistence)])
                    self.assertIn("was ingested with", stderr.getvalue())
                    other.embed_documents.assert_not_called()

    def test_rejects_collection_without_recorded_embedding(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.txt"
            source.write_text("A short document.", encoding="utf-8")
            persistence = Path(directory) / "db"
            chromadb.PersistentClient(path=str(persistence)).create_collection("documents")
            provider = fake_provider([[1.0, 0.0]])
            stderr = io.StringIO()
            with patch_provider(provider), redirect_stderr(stderr), self.assertRaises(SystemExit):
                main([str(source), "--persist-directory", str(persistence)])
            self.assertIn("does not record its embedding", stderr.getvalue())
            provider.embed_documents.assert_not_called()


    def make_corpus(self, root: Path) -> Path:
        corpus = root / "corpus"
        for name, slug, body in (("a", "Guide/A", "## One\n\nAlpha text."), ("b/c", "Guide/B", "## Two\n\nBeta text.")):
            path = corpus / name / "index.md"
            path.parent.mkdir(parents=True)
            path.write_text(f"---\ntitle: {name}\nslug: {slug}\n---\n\n{body}", encoding="utf-8")
        (corpus / "image.svg").write_text("<svg/>", encoding="utf-8")
        return corpus

    def test_directory_mode_embeds_all_files_in_one_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus, persistence = self.make_corpus(root), root / "db"
            (corpus / "empty.md").write_text(" \n", encoding="utf-8")
            provider, stdout, stderr = fake_provider(), io.StringIO(), io.StringIO()
            with patch_provider(provider), redirect_stdout(stdout), redirect_stderr(stderr):
                main([str(corpus), "--persist-directory", str(persistence)])
            provider.embed_documents.assert_called_once()
            stored = chromadb.PersistentClient(path=str(persistence)).get_collection("documents").get(include=["metadatas"])
            self.assertEqual({m["document_id"] for m in stored["metadatas"]}, {"Guide/A", "Guide/B"})
            self.assertEqual({Path(m["source"]).name for m in stored["metadatas"]}, {"index.md"})
            self.assertEqual(len(stored["ids"]), 2)
            self.assertIn("from 2 files", stdout.getvalue())
            self.assertIn("Skipping empty file", stderr.getvalue())

    def test_directory_mode_uses_relative_path_when_there_is_no_slug(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "docs"
            (corpus / "sub").mkdir(parents=True)
            (corpus / "sub" / "notes.txt").write_text("Some notes.", encoding="utf-8")
            persistence = root / "db"
            with patch_provider(fake_provider()), redirect_stdout(io.StringIO()):
                main([str(corpus), "--persist-directory", str(persistence)])
            stored = chromadb.PersistentClient(path=str(persistence)).get_collection("documents").get(include=["metadatas"])
            self.assertEqual(stored["metadatas"][0]["document_id"], "sub/notes")

    def test_directory_mode_rejects_bad_input_before_creating_a_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = self.make_corpus(root)
            empty = root / "empty_dir"
            empty.mkdir()
            duplicate = root / "dup"
            duplicate.mkdir()
            for name in ("one.md", "two.md"):
                (duplicate / name).write_text("---\nslug: Same\n---\n\ntext", encoding="utf-8")
            cases = {
                "document-id": [str(corpus), "--document-id", "x"],
                "source": [str(corpus), "--source", "x"],
                "no files": [str(empty)],
                "duplicate id": [str(duplicate)],
            }
            for label, argv in cases.items():
                with self.subTest(label), patch("ia_agents.rag.ingest_cli.get_embedding_provider") as get_provider:
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        main(argv)
                    get_provider.assert_not_called()


if __name__ == "__main__":
    unittest.main()
