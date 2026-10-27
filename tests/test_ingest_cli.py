"""Exercise the ingestion command without external API calls."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

import chromadb

from ia_agents.ingest_cli import main


class IngestCliTests(unittest.TestCase):
    def test_both_strategies_save_vectors_and_metadata(self):
        for strategy in ("characters", "paragraphs"):
            with self.subTest(strategy=strategy), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "corpus.txt"
                source.write_text("  First  paragraph.\n\nSecond paragraph.  ", encoding="utf-8")
                persistence = root / "nested" / "chroma_db"
                stdout = io.StringIO()
                with patch("ia_agents.ingest_cli.embed_google") as embed, redirect_stdout(stdout):
                    embed.side_effect = lambda texts: [[float(i), 0.5] for i, _ in enumerate(texts)]
                    main([str(source), "--persist-directory", str(persistence), "--strategy", strategy,
                          "--chunk-size", "24", "--overlap", "5", "--max-chars", "24",
                          "--document-id", "example", "--source", "fixture", "--print-chunks"])

                client = chromadb.PersistentClient(path=str(persistence))
                collection = client.get_collection("documents")
                stored = collection.get(include=["documents", "metadatas", "embeddings"])
                self.assertGreater(len(stored["ids"]), 1)
                embed.assert_called_once_with(stored["documents"])
                self.assertIn(stored["ids"][0], stdout.getvalue())
                for i, (chunk_id, text, metadata, vector) in enumerate(zip(
                    stored["ids"], stored["documents"], stored["metadatas"], stored["embeddings"],
                )):
                    self.assertTrue(chunk_id.startswith(f"example-{i:03d}-"))
                    self.assertEqual(metadata["document_id"], "example")
                    self.assertEqual(metadata["source"], "fixture")
                    self.assertEqual(metadata["position"], i)
                    self.assertIn("ingested_at", metadata)
                    self.assertEqual(list(vector), [float(i), 0.5])
                    self.assertTrue(text)
                if strategy == "paragraphs":
                    self.assertEqual(stored["documents"], ["First paragraph.", "Second paragraph."])

    def test_default_persistence_directory_and_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "corpus.txt"
            source.write_text("A document.", encoding="utf-8")
            persistence = Path(directory) / "chroma_db"
            with patch("ia_agents.ingest_cli.Path", wraps=Path) as path_class, \
                    patch("ia_agents.ingest_cli.embed_google", return_value=[[1.0, 0.0]]), \
                    patch("ia_agents.ingest_cli.chromadb.PersistentClient") as client_class, \
                    redirect_stdout(io.StringIO()):
                # Exercise parser defaults while redirecting the relative default to a temp folder.
                path_class.side_effect = lambda value=".": persistence if value == "chroma_db" else Path(value)
                main([str(source)])
                client_class.assert_called_once_with(path=str(persistence))
                client_class.return_value.get_or_create_collection.assert_called_once_with(name="documents")

    def test_invalid_input_does_not_call_google(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "empty.txt"
            source.write_text(" \n\t", encoding="utf-8")
            for extra in ([], ["--overlap", "300"], ["--strategy", "paragraphs", "--max-chars", "0"]):
                with self.subTest(extra=extra), patch("ia_agents.ingest_cli.embed_google") as embed:
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        main([str(source), *extra])
                    embed.assert_not_called()

    def test_vector_count_mismatch_does_not_write_to_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.txt"
            source.write_text("A document.", encoding="utf-8")
            with patch("ia_agents.ingest_cli.embed_google", return_value=[]), \
                    patch("ia_agents.ingest_cli.chromadb.PersistentClient") as client_class:
                with self.assertRaisesRegex(ValueError, "number of vectors"):
                    main([str(source), "--persist-directory", str(Path(directory) / "db")])
                client_class.assert_not_called()

    def test_reingestion_upserts_same_chunk_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.txt"
            source.write_text("A short document.", encoding="utf-8")
            persistence = root / "db"
            with patch("ia_agents.ingest_cli.embed_google", return_value=[[1.0, 0.0]]), \
                    redirect_stdout(io.StringIO()):
                main([str(source), "--persist-directory", str(persistence)])
                main([str(source), "--persist-directory", str(persistence)])
            collection = chromadb.PersistentClient(path=str(persistence)).get_collection("documents")
            self.assertEqual(collection.count(), 1)


if __name__ == "__main__":
    unittest.main()
