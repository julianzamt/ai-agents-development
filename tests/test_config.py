"""Exercise the corpus settings file and how the commands take their defaults from it."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ia_agents.rag import ask_cli, diagnose_cli
from ia_agents.rag.config import config_from_argv, load_config

# A complete config; tests change single keys with with_keys().
FULL = {
    "collection": "docs",
    "persist_directory": "db",
    "retrieval": {"k": 5, "max_distance": 1.0, "max_prompt_chars": 6000},
    "chunking": {"min_chars": 300, "target_chars": 1200, "max_chars": 2000},
}


def with_keys(**sections):
    """FULL with some top-level keys or nested sections replaced (sections are merged)."""
    config = json.loads(json.dumps(FULL))
    for key, value in sections.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    return config


def write_config(directory, data):
    path = Path(directory) / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class LoadConfigTests(unittest.TestCase):
    def test_loads_a_complete_file_and_fills_only_the_optional_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(write_config(directory, FULL))
        self.assertEqual(config["retrieval"], FULL["retrieval"])
        self.assertIsNone(config["eval_dataset"])
        self.assertEqual(config["pricing"], {"input_per_million": None, "output_per_million": None})

    def test_a_missing_default_file_is_an_error(self):
        with patch("ia_agents.rag.config.DEFAULT_CONFIG_PATH", Path("/nonexistent/rag_config.json")):
            with self.assertRaisesRegex(ValueError, "not found"):
                load_config()

    def test_explicit_missing_path_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            load_config("/nonexistent/other.json")

    def test_max_distance_may_be_null(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(write_config(directory, with_keys(retrieval={"max_distance": None})))
        self.assertIsNone(config["retrieval"]["max_distance"])

    def test_rejects_invalid_files(self):
        missing_k = with_keys()
        del missing_k["retrieval"]["k"]
        missing_chunking = with_keys()
        del missing_chunking["chunking"]
        invalid = {
            "unknown key": with_keys(colection="x"),
            "unknown nested key": with_keys(retrieval={"top_k": 3}),
            "wrong type": with_keys(retrieval={"k": "5"}),
            "bool is not a number": with_keys(retrieval={"max_distance": True}),
            "section not an object": with_keys(chunking=5),
            "missing nested key": missing_k,
            "missing section": missing_chunking,
            "null required key": with_keys(retrieval={"k": None}),
        }
        for name, data in invalid.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    load_config(write_config(directory, data))

    def test_committed_config_is_valid(self):
        config = load_config()
        self.assertEqual(config["retrieval"]["max_distance"], 1.2)

    def test_config_from_argv_reads_only_the_config_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_config(directory, with_keys(collection="other"))
            self.assertEqual(config_from_argv(["question", "--config", str(path), "-k", "2"])["collection"], "other")


class CliDefaultsTests(unittest.TestCase):
    def run_ask(self, argv, config):
        with tempfile.TemporaryDirectory() as directory:
            path = write_config(directory, config)
            stdout = io.StringIO()
            result = {"answer": "a", "sources": [], "cited": [],
                      "record": {"variants": [], "invalid_citations": [], "outcome": "answered", "retrieved": [],
                                 "passed_threshold": [], "in_context": [], "cited": [], "context_chars": 0}}
            with patch("ia_agents.rag.ask_cli.chromadb.PersistentClient"), \
                    patch("ia_agents.rag.ask_cli.rag", return_value=result) as rag_call, redirect_stdout(stdout):
                ask_cli.main(["q", "--no-log", "--config", str(path), *argv])
        return rag_call.call_args

    def test_ask_takes_its_defaults_from_the_config(self):
        call = self.run_ask([], with_keys(retrieval={"k": 7, "max_distance": 0.8, "max_prompt_chars": 5000}))
        self.assertEqual((call.kwargs["k"], call.kwargs["max_distance"], call.kwargs["max_prompt_chars"]),
                         (7, 0.8, 5000))

    def test_a_flag_beats_the_config(self):
        call = self.run_ask(["-k", "2", "--max-distance", "0.5"], with_keys(retrieval={"k": 7, "max_distance": 0.8}))
        self.assertEqual((call.kwargs["k"], call.kwargs["max_distance"]), (2, 0.5))

    def test_a_bad_config_file_stops_the_command(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_config(directory, with_keys(nope=1))
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exit_info:
                ask_cli.main(["q", "--config", str(path)])
        self.assertEqual(exit_info.exception.code, 2)

    def run_answering(self, command, config):
        with tempfile.TemporaryDirectory() as directory:
            path = write_config(directory, config)
            with patch("ia_agents.rag.diagnose_cli.chromadb.PersistentClient"), \
                    patch("ia_agents.rag.diagnose_cli.LLMClient"), \
                    patch("ia_agents.rag.diagnose_cli.load_eval_dataset", return_value=[]) as load, \
                    patch("ia_agents.rag.diagnose_cli.run_cases", return_value=[]), \
                    patch("ia_agents.rag.diagnose_cli.judge_runs"), patch("ia_agents.rag.diagnose_cli.stage_report"), \
                    patch("ia_agents.rag.diagnose_cli.print_judgments"), patch("ia_agents.rag.diagnose_cli.print_stages"):
                diagnose_cli.main([command, "--config", str(path)])
        return load.call_args.args[0]

    def test_judge_defaults_to_the_judge_dataset(self):
        config = with_keys(eval_dataset="full.json", judge_dataset="small.json")
        self.assertEqual(self.run_answering("judge", config).name, "small.json")
        self.assertEqual(self.run_answering("stages", config).name, "full.json")

    def test_judge_falls_back_to_the_eval_dataset(self):
        self.assertEqual(self.run_answering("judge", with_keys(eval_dataset="full.json")).name, "full.json")

    def test_diagnose_takes_k_from_the_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_config(directory, with_keys(retrieval={"k": 8}))
            result = {"ok": True, "k": 8, "rank": 1, "distance": 0.1, "retrieved": [], "chunks": []}
            with patch("ia_agents.rag.diagnose_cli.chromadb.PersistentClient"), \
                    patch("ia_agents.rag.diagnose_cli.diagnose_retrieval", return_value=result) as fake, \
                    redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                diagnose_cli.main(["check", "q", "chunk", "--config", str(path)])
        self.assertEqual(fake.call_args.kwargs["k"], 8)


if __name__ == "__main__":
    unittest.main()
