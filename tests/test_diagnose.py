"""Exercise the retrieval diagnosis with a faked search, without external API calls."""

import unittest
from unittest.mock import patch

from ia_agents.rag.diagnose import diagnose_retrieval

EXPECTED = "Web/JavaScript/Guide/Closures-000-abc"


def chunk(chunk_id, distance):
    return {"chunk_id": chunk_id, "text": "x", "distance": distance, "metadata": {}}


class DiagnoseTests(unittest.TestCase):
    def run_diagnose(self, chunks, **kwargs):
        with patch("ia_agents.rag.diagnose.search", return_value=chunks) as fake:
            return diagnose_retrieval("query", EXPECTED, object(), **kwargs), fake

    def test_found(self):
        result, _ = self.run_diagnose([chunk("other", 0.4), chunk(EXPECTED, 0.6)])
        self.assertTrue(result["ok"])
        self.assertEqual((result["rank"], result["distance"]), (2, 0.6))
        self.assertEqual(result["retrieved"], ["other", EXPECTED])

    def test_missing(self):
        result, _ = self.run_diagnose([chunk("other", 0.4)], k=1)
        self.assertFalse(result["ok"])
        self.assertIsNone(result["rank"])
        self.assertIsNone(result["distance"])
        self.assertEqual(result["k"], 1)

    def test_passes_k_to_search(self):
        _, fake = self.run_diagnose([], k=7)
        self.assertEqual(fake.call_args.kwargs["k"], 7)


if __name__ == "__main__":
    unittest.main()
