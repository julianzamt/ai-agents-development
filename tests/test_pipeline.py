"""Exercise the RAG pipeline and the ask command without external API calls."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ia_agents.rag import ask_cli
from ia_agents.rag.pipeline import (
    NO_EVIDENCE_MESSAGE, SYSTEM_PROMPT, UNKNOWN_ANSWER,
    build_context, build_prompt, extract_citations, log_query, rag,
)


def make_chunk(number, distance=0.5, length=900):
    return {
        "chunk_id": f"Web/JavaScript/Guide/Closures-{number:03d}-abc",
        "text": "x" * length,
        "distance": distance,
        "metadata": {
            "document_id": "Web/JavaScript/Guide/Closures",
            "section_id": f"section_{number:02d}",
            "section_title": "Closures > Closure",
        },
    }


class FakeLLM:
    """Stands in for LLMClient: records prompts and returns a fixed answer in Ollama's shape."""

    provider = "ollama"

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def generate(self, prompt, system=None, temperature=None, **options):
        self.calls.append({"prompt": prompt, "system": system, "temperature": temperature})
        return SimpleNamespace(response={"message": {"content": self.answer}})


def run_rag(chunks, answer="A closure ... [1].", **options):
    llm = FakeLLM(answer)
    with patch("ia_agents.rag.pipeline.search", return_value=chunks) as search:
        result = rag("What is a closure?", collection=None, llm=llm, log_path=None, **options)
    return result, llm, search


class BuildContextTests(unittest.TestCase):
    def test_numbers_blocks_in_ranking_order(self):
        context, sources = build_context([make_chunk(1), make_chunk(2)], max_chars=10_000)
        self.assertEqual([source["ref"] for source in sources], [1, 2])
        self.assertLess(context.index("[1] "), context.index("[2] "))
        self.assertIn("(source: Web/JavaScript/Guide/Closures / section_01)", context)

    def test_drops_lower_ranked_chunks_that_exceed_the_budget(self):
        chunks = [make_chunk(1, length=500), make_chunk(2, length=500), make_chunk(3, length=500)]
        # Each block is a 57-char header plus 500 chars, so two blocks and a separator take 1116.
        context, sources = build_context(chunks, max_chars=1200)
        self.assertLessEqual(len(context), 1200)
        self.assertEqual([source["ref"] for source in sources], [1, 2])
        self.assertFalse(any(source["truncated"] for source in sources))

    def test_truncates_a_first_chunk_larger_than_the_budget(self):
        context, sources = build_context([make_chunk(1, length=3000)], max_chars=500)
        self.assertEqual(len(context), 500)
        self.assertEqual(len(sources), 1)
        self.assertTrue(sources[0]["truncated"])

    def test_returns_nothing_when_not_even_the_header_fits(self):
        self.assertEqual(build_context([make_chunk(1)], max_chars=10), ("", []))


class BuildPromptTests(unittest.TestCase):
    def test_delimits_evidence_and_question(self):
        prompt = build_prompt("What is a closure?", "[1] evidence")
        self.assertEqual(
            prompt,
            "<evidence>\n[1] evidence\n</evidence>\n\n<question>\nWhat is a closure?\n</question>",
        )


class ExtractCitationsTests(unittest.TestCase):
    SOURCES = [{"ref": 1}, {"ref": 2}, {"ref": 3}]

    def test_splits_valid_and_invalid_refs(self):
        self.assertEqual(extract_citations("A [1]. B [2][1]. C [7].", self.SOURCES), ([1, 2], [7]))

    def test_reads_comma_separated_lists(self):
        self.assertEqual(extract_citations("Both [1, 3].", self.SOURCES), ([1, 3], []))

    def test_ignores_brackets_inside_code(self):
        answer = "Read `arr[0]` [2].\n```js\nconst x = list[5];\n```\nSee [3]."
        self.assertEqual(extract_citations(answer, self.SOURCES), ([2, 3], []))

    def test_answer_without_citations(self):
        self.assertEqual(extract_citations("No citations.", self.SOURCES), ([], []))


class LogQueryTests(unittest.TestCase):
    def test_appends_one_json_line_per_record_and_creates_the_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "log.jsonl"
            log_query({"query": "¿qué es un closure?"}, path)
            log_query({"query": "second"}, path)
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(line)["query"] for line in lines], ["¿qué es un closure?", "second"])
        self.assertIn("¿", lines[0])


class RagTests(unittest.TestCase):
    def test_no_chunk_within_threshold_skips_the_llm(self):
        result, llm, _ = run_rag([make_chunk(1, distance=1.5), make_chunk(2, distance=1.4)])
        self.assertEqual(llm.calls, [])
        self.assertEqual(result["answer"], NO_EVIDENCE_MESSAGE)
        self.assertEqual(result["record"]["outcome"], "no_evidence")
        self.assertEqual(result["record"]["passed_threshold"], [])
        self.assertEqual(len(result["record"]["retrieved"]), 2)

    def test_model_abstention_becomes_the_no_evidence_message(self):
        for reply in (UNKNOWN_ANSWER, "i don't know the answer"):
            with self.subTest(reply=reply):
                result, llm, _ = run_rag([make_chunk(1)], answer=reply)
                self.assertEqual(len(llm.calls), 1)
                self.assertEqual(result["answer"], NO_EVIDENCE_MESSAGE)
                self.assertEqual(result["record"]["outcome"], "abstained")
                self.assertEqual(result["cited"], [])

    def test_answer_maps_citations_to_chunks(self):
        chunks = [make_chunk(1, 0.6), make_chunk(2, 0.7), make_chunk(3, 1.3)]
        result, llm, _ = run_rag(chunks, answer="Closures [1]. Also [2] and [5].")
        record = result["record"]
        self.assertEqual(record["outcome"], "answered")
        self.assertEqual(record["passed_threshold"], [chunks[0]["chunk_id"], chunks[1]["chunk_id"]])
        self.assertEqual(record["cited"], [chunks[0]["chunk_id"], chunks[1]["chunk_id"]])
        self.assertEqual(record["invalid_citations"], [5])
        self.assertEqual(llm.calls[0]["system"], SYSTEM_PROMPT)
        self.assertEqual(llm.calls[0]["temperature"], 0.0)

    def test_whole_prompt_stays_within_the_budget(self):
        # The first chunk alone exceeds the ~5000 chars left for evidence, so it is truncated.
        chunks = [make_chunk(number, length=8000) for number in (1, 2, 3)]
        result, llm, _ = run_rag(chunks, max_prompt_chars=6000)
        self.assertLessEqual(len(SYSTEM_PROMPT) + len(llm.calls[0]["prompt"]), 6000)
        self.assertEqual(len(result["sources"]), 1)
        self.assertTrue(result["sources"][0]["truncated"])

    def test_question_larger_than_the_budget_is_rejected_before_searching(self):
        with patch("ia_agents.rag.pipeline.search") as search, self.assertRaises(ValueError):
            rag("x" * 7000, collection=None, llm=FakeLLM("unused"), log_path=None)
        search.assert_not_called()

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            rag("   ", collection=None, log_path=None)

    def test_filters_reach_search_and_the_log(self):
        where = {"document_id": "Web/JavaScript/Guide/Closures"}
        where_document = {"$contains": "closure"}
        result, _, search = run_rag([], where=where, where_document=where_document)
        self.assertEqual(search.call_args.kwargs["where"], where)
        self.assertEqual(search.call_args.kwargs["where_document"], where_document)
        self.assertEqual(result["record"]["settings"]["where"], where)
        self.assertEqual(result["record"]["settings"]["where_document"], where_document)

    def test_writes_the_record_to_the_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rag_log.jsonl"
            with patch("ia_agents.rag.pipeline.search", return_value=[make_chunk(1)]):
                result = rag("What is a closure?", collection=None, llm=FakeLLM("A [1]."), log_path=path)
            logged = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(logged, result["record"])
        for field in ("timestamp", "query", "settings", "retrieved", "passed_threshold",
                      "in_context", "context_chars", "provider", "outcome", "cited",
                      "invalid_citations", "answer"):
            self.assertIn(field, logged)


class AskCliTests(unittest.TestCase):
    def run_ask(self, argv, result):
        stdout = io.StringIO()
        with patch("ia_agents.rag.ask_cli.chromadb.PersistentClient"), \
                patch("ia_agents.rag.ask_cli.rag", return_value=result) as rag_call, \
                redirect_stdout(stdout):
            ask_cli.main(argv)
        return stdout.getvalue(), rag_call

    def test_prints_answer_cited_sources_and_invalid_citations(self):
        chunks = [make_chunk(1), make_chunk(2)]
        result, _, _ = run_rag(chunks, answer="Closures [1] [9].")
        output, _ = self.run_ask(["What is a closure?", "--no-log"], result)
        self.assertIn("Answer: Closures [1] [9].", output)
        self.assertIn(f"[1] {chunks[0]['chunk_id']} (Closures > Closure)", output)
        self.assertNotIn(chunks[1]["chunk_id"], output)
        self.assertIn("cited evidence that was not provided: [9]", output)
        self.assertIn("outcome=answered", output)

    def test_passes_options_and_filters_to_rag(self):
        result, _, _ = run_rag([])
        _, rag_call = self.run_ask([
            "q", "--no-log", "-k", "5", "--max-distance", "1.0", "--max-prompt-chars", "8000",
            "--filter", "section_id=section_04", "--document-id", "Web/JavaScript/Guide/Closures",
            "--contains", "closure",
        ], result)
        options = rag_call.call_args.kwargs
        self.assertEqual((options["k"], options["max_distance"], options["max_prompt_chars"]), (5, 1.0, 8000))
        self.assertIsNone(options["log_path"])
        self.assertEqual(options["where"], {"$and": [
            {"section_id": "section_04"}, {"document_id": "Web/JavaScript/Guide/Closures"},
        ]})
        self.assertEqual(options["where_document"], {"$contains": "closure"})


if __name__ == "__main__":
    unittest.main()
