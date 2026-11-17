"""Exercise the evaluation dataset loader and the committed dataset, without external API calls."""

import json
import re
import tempfile
import unittest
from pathlib import Path

from ia_agents.rag.config import load_config
from ia_agents.rag.evaluate import load_eval_dataset

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_REPORT = ROOT / "corpus" / "clean.chunks.txt"
EVAL_PATH = ROOT / load_config()["eval_dataset"]


def load(cases):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "eval.json"
        path.write_text(json.dumps(cases), encoding="utf-8")
        return load_eval_dataset(path)


def case(question="q?", chunks=("a",), must_answer=True, **extra):
    return {"question": question, "relevant_chunks": list(chunks), "must_answer": must_answer, **extra}


class LoaderTests(unittest.TestCase):
    def test_loads_valid_cases(self):
        cases = [case(), case("other?", chunks=(), must_answer=False, kind="out_of_domain")]
        self.assertEqual(load(cases), cases)

    def test_rejects_invalid_datasets(self):
        invalid = {
            "empty list": [],
            "not a list": {"question": "q?"},
            "missing question": [{"relevant_chunks": [], "must_answer": False}],
            "blank question": [case(question=" ")],
            "repeated question": [case(), case()],
            "chunks not a list": [{"question": "q?", "relevant_chunks": "a", "must_answer": True}],
            "repeated chunk": [case(chunks=("a", "a"))],
            "must_answer without chunks": [case(chunks=(), must_answer=True)],
            "chunks without must_answer": [case(must_answer=False)],
            "must_answer not a bool": [case(must_answer=1)],
        }
        for name, cases in invalid.items():
            with self.subTest(name), self.assertRaises(ValueError):
                load(cases)


class CommittedDatasetTests(unittest.TestCase):
    def test_has_enough_cases_of_every_kind(self):
        cases = load_eval_dataset(EVAL_PATH)
        self.assertGreaterEqual(len(cases), 20)
        self.assertEqual({c["kind"] for c in cases},
                         {"direct", "colloquial", "multi_chunk", "out_of_domain", "missing_detail"})

    def test_expected_chunks_exist_in_the_committed_chunk_report(self):
        report = CHUNKS_REPORT.read_text(encoding="utf-8")
        existing = set(re.findall(r"^(\S+) \| \d+ characters$", report, flags=re.M))
        for c in load_eval_dataset(EVAL_PATH):
            for chunk_id in c["relevant_chunks"]:
                self.assertIn(chunk_id, existing, f"stale chunk id for {c['question']!r}; re-check the dataset")


if __name__ == "__main__":
    unittest.main()
