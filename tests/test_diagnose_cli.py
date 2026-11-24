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

METRICS = {"per_case": [{"question": "q?", "kind": "direct", "relevant": ["a", "b"], "retrieved": ["a", "x"],
                         "distances": [0.8, 1.3], "hit": 1, "recall": 0.5, "precision": 0.5, "rr": 1.0},
                        {"question": "fine?", "kind": "direct", "relevant": ["a"], "retrieved": ["a", "x"],
                         "distances": [0.7, 0.9], "hit": 1, "recall": 1.0, "precision": 0.5, "rr": 1.0}],
           "unanswerable": [{"question": "near?", "kind": "missing_detail", "retrieved": ["y"],
                             "distances": [1.1]},
                            {"question": "far?", "kind": "out_of_domain", "retrieved": ["z"],
                             "distances": [1.5]}],
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
        table = output.split("Distances")[0].strip().splitlines()
        self.assertEqual([line.split()[0] for line in table[3:]], ["k=1", "k=3"])

    def test_verbose_lists_each_case(self):
        output, _ = self.run_metrics("-k", "2", "-v")
        self.assertIn("HIT  recall=0.50", output)
        self.assertIn("*1. 0.8000   a", output)
        self.assertIn("missing: b", output)

    def test_only_failures_hides_fully_retrieved_cases(self):
        output, _ = self.run_metrics("--only-failures")
        self.assertIn("q?", output)
        self.assertNotIn("fine?", output)
        self.assertIn("[direct]", output)

    def test_shows_distances_against_the_threshold(self):
        output, _ = self.run_metrics("-v", "--max-distance", "1.2")
        self.assertIn("1.3000 x", output)
        self.assertIn("relevant chunks retrieved:  n=2", output)
        self.assertIn("beyond threshold: 0", output)
        self.assertIn("within threshold: 1", output)
        self.assertIn("far?", output)

    def test_only_failures_lists_unanswerable_cases_within_the_threshold(self):
        output, _ = self.run_metrics("--only-failures", "--max-distance", "1.2")
        self.assertIn("near?", output)
        self.assertNotIn("far?", output)

    def test_prints_the_threshold_and_its_source(self):
        output, _ = self.run_metrics("--max-distance", "0.9")
        self.assertIn("Distance threshold (max_distance): 0.9 (from --max-distance)", output)

    def test_prints_a_table_per_kind(self):
        by_kind = {"direct": {"cases": 2, "hit_rate": 1.0, "recall": 0.625, "precision": 0.5, "mrr": 1.0}}
        with patch.dict(METRICS, {"by_kind": by_kind}):
            output, _ = self.run_metrics("-k", "1", "3")
        self.assertIn("By kind:", output)
        self.assertEqual(output.count("0.625"), 2)

    def test_rejects_non_positive_k(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exit_info:
            main(["metrics", "-k", "0"])
        self.assertEqual(exit_info.exception.code, 2)


class ColorTests(unittest.TestCase):
    def test_colors_only_on_a_terminal_without_no_color(self):
        from ia_agents.rag.diagnose_cli import paint
        with patch("sys.stdout.isatty", return_value=True), patch.dict("os.environ", {}, clear=True):
            self.assertEqual(paint("HIT", "green"), "\033[32mHIT\033[0m")
        with patch("sys.stdout.isatty", return_value=True), patch.dict("os.environ", {"NO_COLOR": "1"}):
            self.assertEqual(paint("HIT", "green"), "HIT")
        with patch("sys.stdout.isatty", return_value=False):
            self.assertEqual(paint("HIT", "green"), "HIT")


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
                       "judgment": Judgment(faithfulness=5, relevance=4, justification="Supported."),
                       "prompt": "<evidence>\nev\n</evidence>\n\n<question>\nq1?\n</question>"},
                      {"question": "q2?", "outcome": "abstained", "answer": "y", "judgment": None,
                       "prompt": None}]},
        "abstention": {"cases": 2, "outcomes": {"no_evidence": 1, "abstained": 0, "answered": 1},
                       "abstention_rate": 0.5, "by_kind": {"missing_detail": {"cases": 2, "abstention_rate": 0.5}},
                       "per_case": [
                           {"question": "q1?", "kind": "missing_detail", "outcome": "no_evidence",
                            "top_distance": 1.3, "cited": [], "answer": ""},
                           {"question": "q2?", "kind": "missing_detail", "outcome": "answered",
                            "top_distance": 0.8, "cited": ["a"], "answer": "It is 42."}]},
    }
    REPORT_FUNCTIONS = {"cost": "cost_report", "latency": "latency_report",
                        "stages": "stage_report", "judge": "judge_runs", "abstention": "abstention_report"}

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

    def test_abstention_runs_unanswerable_cases(self):
        output, runs, _ = self.run_command("abstention", "-v")
        self.assertFalse(runs.call_args.kwargs["answerable"])
        self.assertIn("no_evidence=1, abstained=0, answered=1", output)
        self.assertIn("abstention rate: 0.500", output)
        self.assertIn("ANSWERED", output)
        self.assertIn("It is 42.", output)
        self.assertIn("cited: a", output)

    def test_judge(self):
        output, _, fake = self.run_command("judge", "-v")
        self.assertIn("faithfulness: 4.50/5  relevance: 3.00/5", output)
        self.assertIn("faithfulness 5/5  relevance 4/5", output)
        self.assertIn("2. q2?  [abstained]", output)
        self.assertNotIn("cost", output)
        self.assertEqual(len(fake.call_args.args), 2)


    def test_judge_verbose_shows_prompt_response_and_judgment_in_order(self):
        output, _, _ = self.run_command("judge", "-v")
        first_case = output.split("2. q2?")[0]
        positions = [first_case.index(text) for text in
                     ("── Prompt", "<evidence>", "── Model response", "\n       x\n", "── Judge", "Supported.")]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("not judged (abstained)", output)

    def test_score_colors(self):
        from ia_agents.rag.diagnose_cli import score_style
        self.assertEqual([score_style(n) for n in (5, 4, 3, 2, 1)], ["green", "green", "orange", "red", "red"])

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
