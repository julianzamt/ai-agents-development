"""Class 4 laboratory: offline corpus ingestion, storage, and traceability."""

import hashlib
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from google.genai import types, _transformers

from ia_agents.llm.embed import build_chunks, clean_text, embed_google


CORPUS_PATH = Path(__file__).with_name("test_corpus.txt")


class IngestionLabTests(unittest.TestCase):
    def test_character_ingestion(self):
        self._check_corpus_ingestion("characters")

    def test_paragraph_ingestion(self):
        self._check_corpus_ingestion("paragraphs")

    def test_unknown_strategy(self):
        with self.assertRaisesRegex(ValueError, "Unknown chunking strategy"):
            build_chunks("text", "doc", "source", strategy="unknown")

    def _check_corpus_ingestion(self, strategy):
        raw_text = CORPUS_PATH.read_text(encoding="utf-8")
        self.assertTrue(9_000 <= len(raw_text) <= 11_000)
        text = clean_text(raw_text)
        document_id = "charly-garcia-career"

        # 1,000 characters retain several sentences of biographical context.
        # A 150-character overlap preserves context across fixed boundaries,
        # while keeping every chunk below the lab's 1,500-character target.
        chunks = build_chunks(
            text, document_id, source=CORPUS_PATH.name,
            chunk_size=1_000, overlap=150, strategy=strategy, max_chars=1_000,
        )
        self.assertGreater(len(chunks), 1)
        # Enable readable output with PRINT_CHUNKS=1 when running unittest.
        if os.environ.get("PRINT_CHUNKS") == "1":
            for chunk in chunks:
                print(f"\n{'=' * 72}")
                print(f"{strategy} | Chunk {chunk.position} | {len(chunk.text)} characters")
                print(f"ID: {chunk.chunk_id}")
                print(f"Document: {chunk.document_id} | Source: {chunk.source}")
                print(f"Ingested at: {chunk.ingested_at.isoformat()}")
                print(f"{'-' * 72}\n{chunk.text}")

        # Only the external provider is mocked: cleaning, chunk construction,
        # embed_google's batch request/response handling, and JSON I/O are real.
        # These distinct vectors are fixtures, not semantic embeddings.
        expected_vectors = [[float(i), 0.5, -0.5] for i in range(len(chunks))]
        response = SimpleNamespace(embeddings=[
            SimpleNamespace(values=vector) for vector in expected_vectors
        ])
        with patch.dict("os.environ", {
            "GOOGLE_API_KEY": "offline-test-key",
            "GOOGLE_EMBEDDING_MODEL_ID": "offline-test-model",
        }), patch("ia_agents.llm.embed.genai.Client") as client_class:
            embed_content = client_class.return_value.models.embed_content
            embed_content.return_value = response
            vectors = embed_google([chunk.text for chunk in chunks])
            embed_content.assert_called_once_with(
                model="offline-test-model",
                contents=[types.Content(parts=[types.Part(text=c.text)]) for c in chunks],
            )
            # Exercise the SDK's Embedding 2 normalization: plain strings
            # collapse into one Content, but these boundaries must survive.
            normalized = _transformers.t_contents(embed_content.call_args.kwargs["contents"])
            self.assertEqual(len(normalized), len(chunks))
            self.assertEqual([content.parts[0].text for content in normalized],
                             [chunk.text for chunk in chunks])

        self.assertEqual(len(vectors), len(chunks))
        self.assertEqual(len(vectors), len(chunks))
        for position, (chunk, vector) in enumerate(zip(chunks, vectors)):
            with self.subTest(position=position):
                self.assertTrue(0 < len(chunk.text) < 1_500)
                if strategy == "characters":
                    self.assertEqual(chunk.text, text[position * 850:position * 850 + 1_000])
                else:
                    self.assertLessEqual(len(chunk.text), 1_000)
                self.assertEqual(vector, expected_vectors[position])
                self.assertEqual(chunk.source, CORPUS_PATH.name)
                timestamp = chunk.ingested_at
                self.assertIsNotNone(timestamp.utcoffset())
                content_hash = hashlib.sha256(chunk.text.encode()).hexdigest()[:8]
                expected_id = f"{document_id}-{position:03d}-{content_hash}"
                self.assertEqual(chunk.chunk_id, expected_id)
                self.assertEqual(chunk.document_id, document_id)
                self.assertEqual(chunk.position, position)

        # Recover the complete cleaned document, including its final fragment.
        if strategy == "characters":
            reconstructed = chunks[0].text + "".join(
                chunk.text[150:] for chunk in chunks[1:]
            )
        else:
            reconstructed = "\n\n".join(chunk.text for chunk in chunks)
        self.assertEqual(reconstructed, text)



if __name__ == "__main__":
    unittest.main()
