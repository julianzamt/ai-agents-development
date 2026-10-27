"""Exercise the ingestion command without external API calls."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

from ia_agents.ingest_cli import main


class IngestCliTests(unittest.TestCase):
    def test_both_strategies_save_vectors_and_metadata(self):
        for strategy in ("characters", "paragraphs"):
            with self.subTest(strategy=strategy), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "corpus.txt"
                source.write_text("  First  paragraph.\n\nSecond paragraph.  ", encoding="utf-8")
                output = Path(directory) / "nested" / "vectors.json"
                stdout = io.StringIO()
                with patch("ia_agents.ingest_cli.embed_google") as embed, redirect_stdout(stdout):
                    embed.side_effect = lambda texts: [[float(i), 0.5] for i, _ in enumerate(texts)]
                    main([str(source), "--output", str(output), "--strategy", strategy,
                          "--chunk-size", "24", "--overlap", "5", "--max-chars", "24",
                          "--document-id", "example", "--source", "fixture", "--print-chunks"])
                records = json.loads(output.read_text(encoding="utf-8"))
                self.assertGreater(len(records), 1)
                embed.assert_called_once_with([record["text"] for record in records])
                self.assertIn(records[0]["chunk_id"], stdout.getvalue())
                for i, record in enumerate(records):
                    self.assertEqual(record["document_id"], "example")
                    self.assertEqual(record["source"], "fixture")
                    self.assertEqual(record["position"], i)
                    self.assertEqual(record["vector"], [float(i), 0.5])
                    self.assertIn("ingested_at", record)
                if strategy == "paragraphs":
                    self.assertEqual([r["text"] for r in records], ["First paragraph.", "Second paragraph."])

    def test_invalid_input_does_not_call_google(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "empty.txt"
            source.write_text(" \n\t", encoding="utf-8")
            output = Path(directory) / "output.json"
            for extra in ([], ["--overlap", "300"], ["--strategy", "paragraphs", "--max-chars", "0"]):
                with self.subTest(extra=extra), patch("ia_agents.ingest_cli.embed_google") as embed:
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        main([str(source), "--output", str(output), *extra])
                    embed.assert_not_called()
                    self.assertFalse(output.exists())

    def test_vector_count_mismatch_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.txt"
            source.write_text("A document.", encoding="utf-8")
            output = Path(directory) / "output.json"
            output.write_text("existing", encoding="utf-8")
            with patch("ia_agents.ingest_cli.embed_google", return_value=[]):
                with self.assertRaisesRegex(ValueError, "number of vectors"):
                    main([str(source), "--output", str(output)])
            self.assertEqual(output.read_text(), "existing")
