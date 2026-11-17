"""Exercise retrieval diagnosis, metrics and the evaluation dataset, without external API calls."""

import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from ia_agents.rag.config import load_config
from ia_agents.rag.diagnose import (
    SYSTEM_JUDGE, Judgment, diagnose_retrieval, judge, classify_error, cost_report, cost_usd, judge_runs, latency_report, load_eval_dataset, retrieval_metrics,
    run_cases, stage_report,
)

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_REPORT = ROOT / "corpus" / "clean.chunks.txt"
EVAL_PATH = ROOT / load_config()["eval_dataset"]

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


def fake_search(results):
    def search(question, collection, k):
        return [{"chunk_id": chunk_id} for chunk_id in results[question][:k]]
    return patch("ia_agents.rag.diagnose.search", search)


class RetrievalMetricsTests(unittest.TestCase):
    def test_computes_metrics_and_skips_unanswerable_cases(self):
        cases = [
            case("one?", chunks=("a",)),
            case("two?", chunks=("a", "b")),
            case("miss?", chunks=("z",)),
            case("none?", chunks=(), must_answer=False),
        ]
        results = {"one?": ["x", "a", "y"], "two?": ["a", "b", "y"], "miss?": ["x", "y", "w"]}
        with fake_search(results):
            metrics = retrieval_metrics(cases, None, k=3)
        self.assertEqual(metrics["cases"], 3)
        self.assertAlmostEqual(metrics["hit_rate"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertAlmostEqual(metrics["precision"], (1 + 2 + 0) / 3 / 3)
        self.assertAlmostEqual(metrics["mrr"], (0.5 + 1 + 0) / 3)

    def test_smaller_k_lowers_recall_and_raises_precision(self):
        cases = [case("q?", chunks=("a", "b"))]
        with fake_search({"q?": ["a", "x", "b"]}):
            at_1 = retrieval_metrics(cases, None, k=1)
            at_3 = retrieval_metrics(cases, None, k=3)
        self.assertLess(at_1["recall"], at_3["recall"])
        self.assertGreater(at_1["precision"], at_3["precision"])

    def test_rejects_bad_input(self):
        with fake_search({}), self.assertRaises(ValueError):
            retrieval_metrics([case()], None, k=0)
        with fake_search({}), self.assertRaises(ValueError):
            retrieval_metrics([case(chunks=(), must_answer=False)], None, k=3)


class FakeLLM:
    def __init__(self, judgment):
        self.judgment = judgment
        self.calls = []

    def generate(self, prompt, **options):
        self.calls.append({"prompt": prompt, **options})
        return SimpleNamespace(structured=self.judgment)


def make_record(retrieved=("a",), passed=None, in_context=("a",), outcome="answered", k=3,
                input_tokens=100, output_tokens=10):
    return {
        "outcome": outcome, "settings": {"k": k}, "input_tokens": input_tokens, "output_tokens": output_tokens,
        "retrieved": [{"chunk_id": c} for c in retrieved],
        "passed_threshold": list(retrieved if passed is None else passed),
        "in_context": [{"chunk_id": c} for c in in_context],
    }


def make_run(question="q?", latency=1.0, answer="an answer", **record_options):
    return {"case": case(question), "answer": answer, "context": f"context for {question}",
            "record": make_record(**record_options), "latency": latency}


def fake_rag(records):
    def rag(question, collection, **options):
        return {"answer": f"answer to {question}", "context": f"context for {question}",
                "record": records[question]}
    return patch("ia_agents.rag.diagnose.rag", rag)


class RunCasesTests(unittest.TestCase):
    def test_runs_only_answerable_cases_and_times_them(self):
        cases = [case("a?"), case("none?", chunks=(), must_answer=False)]
        with fake_rag({"a?": make_record()}):
            runs = run_cases(cases, None, object())
        self.assertEqual([run["case"]["question"] for run in runs], ["a?"])
        self.assertEqual(runs[0]["answer"], "answer to a?")
        self.assertEqual(runs[0]["context"], "context for a?")
        self.assertGreaterEqual(runs[0]["latency"], 0)

    def test_passes_settings_and_does_not_log(self):
        with patch("ia_agents.rag.diagnose.rag", return_value={"answer": "", "context": "", "record": make_record()}) as fake:
            run_cases([case()], "collection", object(), k=2, max_distance=0.5, max_prompt_chars=900)
        self.assertEqual(fake.call_args.kwargs["k"], 2)
        self.assertEqual(fake.call_args.kwargs["max_distance"], 0.5)
        self.assertEqual(fake.call_args.kwargs["max_prompt_chars"], 900)
        self.assertIsNone(fake.call_args.kwargs["log_path"])

    def test_rejects_a_dataset_without_answerable_cases(self):
        with self.assertRaises(ValueError):
            run_cases([case(chunks=(), must_answer=False)], None, object())


class CostTests(unittest.TestCase):
    PRICING = {"input_per_million": 1.0, "output_per_million": 2.0}

    def test_prices_tokens_per_million(self):
        self.assertAlmostEqual(cost_usd(2_000_000, 100_000, {"input_per_million": 3.0, "output_per_million": 15.0}), 7.5)

    def test_unpriced_is_none(self):
        self.assertIsNone(cost_usd(10, 10, None))
        self.assertIsNone(cost_usd(10, 10, {"input_per_million": None, "output_per_million": None}))
        self.assertIsNone(cost_usd(10, 10, {"input_per_million": 3.0, "output_per_million": None}))

    def test_report_totals_tokens_and_cost(self):
        runs = [make_run("a?", input_tokens=1000, output_tokens=100),
                make_run("b?", input_tokens=2000, output_tokens=200),
                make_run("c?", outcome="no_evidence", input_tokens=None, output_tokens=None)]
        report = cost_report(runs, self.PRICING)
        self.assertEqual((report["cases"], report["input_tokens"], report["output_tokens"]), (3, 3000, 300))
        self.assertAlmostEqual(report["cost"], (3000 + 600) / 1_000_000)
        self.assertAlmostEqual(report["per_case"][0]["cost"], (1000 + 200) / 1_000_000)
        self.assertEqual(report["per_case"][2]["input_tokens"], 0)

    def test_report_without_pricing_has_no_cost(self):
        report = cost_report([make_run()])
        self.assertIsNone(report["cost"])
        self.assertEqual(report["input_tokens"], 100)


class LatencyTests(unittest.TestCase):
    def test_report_statistics(self):
        report = latency_report([make_run("a?", latency=1.0), make_run("b?", latency=2.0), make_run("c?", latency=6.0)])
        self.assertEqual(report["cases"], 3)
        self.assertAlmostEqual(report["mean"], 3.0)
        self.assertEqual((report["median"], report["max"]), (2.0, 6.0))
        self.assertEqual(report["per_case"][2], {"question": "c?", "latency": 6.0})


class StageTests(unittest.TestCase):
    def error(self, **record_options):
        return classify_error(case(), make_record(**record_options))

    def test_no_error(self):
        self.assertIsNone(self.error())

    def test_unanswerable_case_has_no_error(self):
        self.assertIsNone(classify_error(case(chunks=(), must_answer=False), make_record(retrieved=("x",))))

    def test_retrieval(self):
        error = self.error(retrieved=("x", "y"), in_context=("x",))
        self.assertEqual(error["stage"], "retrieval")
        self.assertIn("top 3", error["detail"])

    def test_context_beyond_threshold_and_dropped_by_budget(self):
        error = self.error(retrieved=("x", "a"), passed=("x",), in_context=("x",))
        self.assertEqual(error["stage"], "context")
        self.assertIn("max_distance", error["detail"])
        error = self.error(retrieved=("x", "a"), passed=("x", "a"), in_context=("x",))
        self.assertEqual(error["stage"], "context")
        self.assertIn("budget", error["detail"])

    def test_generation_when_the_model_abstains_despite_the_evidence(self):
        self.assertEqual(self.error(outcome="abstained")["stage"], "generation")

    def test_earliest_stage_wins(self):
        self.assertEqual(self.error(retrieved=("x",), in_context=(), outcome="no_evidence")["stage"], "retrieval")

    def test_report_counts_stages(self):
        runs = [make_run("a?"), make_run("b?", retrieved=("x",), in_context=()),
                make_run("c?", outcome="abstained"), make_run("d?", outcome="abstained")]
        report = stage_report(runs)
        self.assertEqual(report["errors"], {"retrieval": 1, "context": 0, "generation": 2})
        self.assertEqual((report["cases"], report["no_error"]), (4, 1))
        self.assertIsNone(report["per_case"][0]["error"])


class JudgeTests(unittest.TestCase):
    def test_returns_the_structured_judgment(self):
        expected = Judgment(faithfulness=4, relevance=5, justification="Supported.")
        llm = FakeLLM(expected)
        self.assertEqual(judge("What is x?", "x is y.", "x is y [1].", llm), expected)

    def test_sends_delimited_inputs_with_the_judge_prompt(self):
        llm = FakeLLM(Judgment(faithfulness=1, relevance=1, justification="No."))
        judge("the question", "the evidence", "the answer", llm)
        call = llm.calls[0]
        for tag, text in (("question", "the question"), ("evidence", "the evidence"), ("answer", "the answer")):
            self.assertIn(f"<{tag}>\n{text}\n</{tag}>", call["prompt"])
        self.assertEqual(call["system"], SYSTEM_JUDGE)
        self.assertEqual(call["temperature"], 0.0)
        self.assertIs(call["schema_model"], Judgment)

    def test_scores_are_limited_to_one_through_five(self):
        for score in (0, 6):
            with self.assertRaises(ValidationError):
                Judgment(faithfulness=score, relevance=3, justification="x")
            with self.assertRaises(ValidationError):
                Judgment(faithfulness=3, relevance=score, justification="x")

    def test_runs_are_averaged_and_unanswered_ones_are_not_judged(self):
        scores = iter([Judgment(faithfulness=5, relevance=4, justification="a"),
                       Judgment(faithfulness=3, relevance=2, justification="b")])
        llm = FakeLLM(None)
        llm.generate = lambda prompt, **options: SimpleNamespace(structured=next(scores))
        runs = [make_run("a?"), make_run("b?"), make_run("c?", outcome="abstained")]
        report = judge_runs(runs, llm)
        self.assertEqual((report["cases"], report["answered"]), (3, 2))
        self.assertAlmostEqual(report["faithfulness"], 4.0)
        self.assertAlmostEqual(report["relevance"], 3.0)
        self.assertIsNone(report["per_case"][2]["judgment"])

    def test_judges_the_context_the_model_saw(self):
        llm = FakeLLM(Judgment(faithfulness=5, relevance=5, justification="ok"))
        judge_runs([make_run("a?", answer="the answer")], llm)
        self.assertIn("<evidence>\ncontext for a?\n</evidence>", llm.calls[0]["prompt"])
        self.assertIn("<answer>\nthe answer\n</answer>", llm.calls[0]["prompt"])

    def test_nothing_answered_gives_no_scores(self):
        report = judge_runs([make_run(outcome="no_evidence")], FakeLLM(None))
        self.assertEqual(report["answered"], 0)
        self.assertIsNone(report["faithfulness"])


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
