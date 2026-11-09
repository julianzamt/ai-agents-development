"""Chunking tests; no embedding provider involved."""

import unittest

from ia_agents.rag.preprocess import build_chunks


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


if __name__ == "__main__":
    unittest.main()
