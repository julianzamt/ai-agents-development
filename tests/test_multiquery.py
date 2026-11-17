"""Exercise multi-query retrieval with a fake LLM and a faked search, without external API calls."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ia_agents.rag.multiquery import generate_variants, multi_query_search
from ia_agents.rag.pipeline import rag


class FakeLLM:
    provider = "ollama"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, prompt, system=None, temperature=None, **options):
        self.calls.append({"prompt": prompt, "system": system})
        return SimpleNamespace(response={"message": {"content": self.replies.pop(0)}})


def chunk(chunk_id, distance):
    return {"chunk_id": chunk_id, "text": f"text {chunk_id}", "distance": distance,
            "metadata": {"document_id": "doc", "section_id": "s", "section_title": "t"}}


class GenerateVariantsTests(unittest.TestCase):
    def test_strips_markers_blanks_repeats_and_the_original(self):
        llm = FakeLLM("1. what is a closure\n- Closure scope\n\n* closure scope\nWhat is a Closure?\n3) lexical environment")
        variants = generate_variants("What is a closure?", llm, 5)
        self.assertEqual(variants, ["what is a closure", "Closure scope", "lexical environment"])

    def test_keeps_at_most_n(self):
        self.assertEqual(generate_variants("q", FakeLLM("a\nb\nc\nd"), 2), ["a", "b"])

    def test_asks_for_n_variants(self):
        llm = FakeLLM("a")
        generate_variants("question", llm, 3)
        self.assertEqual(llm.calls[0]["prompt"], "question")
        self.assertIn("3 different", llm.calls[0]["system"])


class MultiQuerySearchTests(unittest.TestCase):
    def test_merges_keeping_best_distance_and_returns_top_k(self):
        results = [
            [chunk("a", 0.5), chunk("b", 0.9)],
            [chunk("b", 0.4), chunk("c", 0.7)],
        ]
        with patch("ia_agents.rag.multiquery.search", return_value=results) as fake:
            chunks, variants = multi_query_search("q", None, FakeLLM("v1"), 1, k=2, where={"x": 1})
        self.assertEqual([(c["chunk_id"], c["distance"]) for c in chunks], [("b", 0.4), ("a", 0.5)])
        self.assertEqual(variants, ["v1"])
        self.assertEqual(fake.call_args.args[0], ["q", "v1"])
        self.assertEqual(fake.call_args.kwargs["where"], {"x": 1})


class RagMultiQueryTests(unittest.TestCase):
    def test_off_by_default(self):
        llm = FakeLLM("Answer [1].")
        with patch("ia_agents.rag.pipeline.search", return_value=[chunk("a", 0.5)]) as plain, \
                patch("ia_agents.rag.pipeline.multi_query_search") as multi:
            result = rag("q", collection=None, llm=llm, log_path=None)
        multi.assert_not_called()
        plain.assert_called_once()
        self.assertEqual(result["record"]["variants"], [])
        self.assertEqual(result["record"]["settings"]["multi_query"], 0)
        self.assertEqual(len(llm.calls), 1)

    def test_enabled_uses_merged_chunks_and_logs_variants(self):
        llm = FakeLLM("Answer [1].")
        with patch("ia_agents.rag.pipeline.multi_query_search", return_value=([chunk("a", 0.5)], ["v1"])) as multi:
            result = rag("q", collection=None, llm=llm, log_path=None, multi_query=2, k=4)
        self.assertEqual(multi.call_args.args[3:5], (2, 4))
        self.assertEqual(result["record"]["variants"], ["v1"])
        self.assertEqual(result["record"]["settings"]["multi_query"], 2)
        self.assertEqual(result["record"]["provider"], "ollama")

    def test_negative_is_rejected(self):
        with self.assertRaises(ValueError):
            rag("q", collection=None, llm=FakeLLM(), log_path=None, multi_query=-1)


if __name__ == "__main__":
    unittest.main()
