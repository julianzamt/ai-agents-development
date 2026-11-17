"""Exercise the diagnose command without external API calls."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import chromadb

from ia_agents.rag.diagnose import Judgment
from ia_agents.rag.diagnose_cli import main


def make_collection(directory):
    client = chromadb.PersistentClient(path=directory)
    client.create_collection("documents")

METRICS = {"per_case": [{"question": "q?", "relevant": ["a", "b"], "retrieved": ["a", "x"], "hit": 1,
                         "recall": 0.5, "precision": 0.5, "rr": 1.0}],
           "cases": 2, "hit_rate": 1.0, "recall": 0.5, "precision": 0.25, "mrr": 0.75}


def result(ok):
    chunks = [{"chunk_id": "a", "distance": 0.5, "text": "", "metadata": {}}]
    return {"ok": ok, "k": 3, "rank": 1 if ok else None, "distance": 0.5 if ok else None,
            "retrieved": ["a"], "chunks": chunks}


class DiagnoseCliTests(unittest.TestCase):
    def run_main(self, outcome, *extra):
        with tempfile.TemporaryDirectory() as directory:
            make_collection(directory)
            stdout = io.StringIO()
            with patch("ia_agents.rag.diagnose_cli.diagnose_retrieval", return_value=outcome) as fake, \
                    redirect_stdout(stdout), self.assertRaises(SystemExit) as exit_info:
                main(["check", "question", "a", "--persist-directory", directory, *extra])
            return stdout.getvalue(), exit_info.exception.code, fake

    def test_found_exits_zero(self):
        output, code, _ = self.run_main(result(True))
        self.assertEqual(code, 0)
        self.assertIn("OK: a is rank 1 of 3", output)

    def test_missing_exits_one(self):
        output, code, _ = self.run_main(result(False))
        self.assertEqual(code, 1)
        self.assertIn("FAIL: a is not in the top 3", output)

    def test_passes_k(self):
        _, _, fake = self.run_main(result(True), "-k", "7")
        self.assertEqual(fake.call_args.kwargs["k"], 7)

    def test_rejects_non_positive_k(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exit_info:
            main(["check", "question", "a", "-k", "0"])
        self.assertEqual(exit_info.exception.code, 2)


class MetricsTests(unittest.TestCase):
    def run_metrics(self, *extra):
        with tempfile.TemporaryDirectory() as directory:
            make_collection(directory)
            dataset = Path(directory) / "eval.json"
            dataset.write_text(json.dumps(
                [{"question": "q?", "relevant_chunks": ["a"], "must_answer": True}]), encoding="utf-8")
            stdout = io.StringIO()
            with patch("ia_agents.rag.diagnose_cli.retrieval_metrics",
                       side_effect=lambda cases, collection, k: {**METRICS, "k": k}) as fake, \
                    redirect_stdout(stdout):
                main(["metrics", "--dataset", str(dataset), "--persist-directory", directory, *extra])
            return stdout.getvalue(), fake

    def test_prints_one_row_per_k(self):
        output, fake = self.run_metrics("-k", "1", "3")
        self.assertEqual([call.kwargs["k"] for call in fake.call_args_list], [1, 3])
        self.assertIn("0.750", output)
        self.assertEqual(len(output.strip().splitlines()), 4)

    def test_verbose_lists_each_case(self):
        output, _ = self.run_metrics("-k", "2", "-v")
        self.assertIn("HIT  recall=0.50", output)
        self.assertIn("*1. a", output)
        self.assertIn("missing: b", output)

    def test_rejects_non_positive_k(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exit_info:
            main(["metrics", "-k", "0"])
        self.assertEqual(exit_info.exception.code, 2)


class AnsweringTests(unittest.TestCase):
    REPORTS = {
        "cost": {"cases": 2, "input_tokens": 1200, "output_tokens": 80, "cost": 0.0048,
                 "per_case": [{"question": "q1?", "input_tokens": 700, "output_tokens": 50, "cost": 0.003}]},
        "latency": {"cases": 2, "mean": 2.5, "median": 2.0, "max": 4.0,
                    "per_case": [{"question": "q1?", "latency": 1.25}]},
        "stages": {"cases": 3, "errors": {"retrieval": 1, "context": 0, "generation": 1}, "no_error": 1,
                   "per_case": [{"question": "q1?", "error": {"stage": "retrieval", "detail": "no relevant chunk in the top 3"}},
                                {"question": "q2?", "error": None}]},
        "judge": {"cases": 3, "answered": 2, "faithfulness": 4.5, "relevance": 3.0,
                  "per_case": [
                      {"question": "q1?", "outcome": "answered", "answer": "x",
                       "judgment": Judgment(faithfulness=5, relevance=4, justification="Supported.")},
                      {"question": "q2?", "outcome": "abstained", "answer": "y", "judgment": None}]},
    }
    REPORT_FUNCTIONS = {"cost": "cost_report", "latency": "latency_report",
                        "stages": "stage_report", "judge": "judge_runs"}

    def run_command(self, command, *extra, report=None):
        with tempfile.TemporaryDirectory() as directory:
            make_collection(directory)
            dataset = Path(directory) / "eval.json"
            dataset.write_text(json.dumps(
                [{"question": "q?", "relevant_chunks": ["a"], "must_answer": True}]), encoding="utf-8")
            stdout = io.StringIO()
            reporter = self.REPORT_FUNCTIONS[command]
            with patch("ia_agents.rag.diagnose_cli.run_cases", return_value=["run"]) as runs, \
                    patch(f"ia_agents.rag.diagnose_cli.{reporter}", return_value=report or self.REPORTS[command]) as fake, \
                    patch("ia_agents.rag.diagnose_cli.LLMClient"), redirect_stdout(stdout):
                main([command, "--dataset", str(dataset), "--persist-directory", directory, *extra])
            return stdout.getvalue(), runs, fake

    def test_cost_prints_tokens_and_cost(self):
        output, _, fake = self.run_command("cost")
        self.assertIn("tokens: 1200 in, 80 out", output)
        self.assertIn("cost: $0.00480 total, $0.00240 per case", output)
        self.assertEqual(fake.call_args.args[0], ["run"])
        self.assertIn("input_per_million", fake.call_args.args[1])

    def test_cost_without_pricing(self):
        output, _, _ = self.run_command("cost", report={**self.REPORTS["cost"], "cost": None})
        self.assertIn("cost: n/a", output)

    def test_latency(self):
        output, _, _ = self.run_command("latency", "-v")
        self.assertIn("mean: 2.50s  median: 2.00s  max: 4.00s", output)
        self.assertIn("1.25s  q1?", output)

    def test_stages(self):
        output, _, _ = self.run_command("stages", "-v")
        self.assertIn("retrieval=1, context=0, generation=1 | no error: 1", output)
        self.assertIn("RETRIEVAL: no relevant chunk in the top 3  q1?", output)
        self.assertIn("ok  q2?", output)

    def test_judge(self):
        output, _, fake = self.run_command("judge", "-v")
        self.assertIn("faithfulness: 4.50/5  relevance: 3.00/5", output)
        self.assertIn("faithfulness=5 relevance=4  q1?", output)
        self.assertIn("not judged (abstained)  q2?", output)
        self.assertNotIn("cost", output)
        self.assertEqual(len(fake.call_args.args), 2)

    def test_passes_settings_to_the_run(self):
        _, runs, _ = self.run_command("latency", "-k", "2", "--max-distance", "0.9", "--max-prompt-chars", "5000")
        kwargs = runs.call_args.kwargs
        self.assertEqual((kwargs["k"], kwargs["max_distance"], kwargs["max_prompt_chars"]), (2, 0.9, 5000))

    def test_rejects_non_positive_k(self):
        for command in ("cost", "latency", "stages", "judge"):
            with self.subTest(command), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exit_info:
                main([command, "-k", "0"])
            self.assertEqual(exit_info.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
