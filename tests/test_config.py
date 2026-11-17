"""Exercise the corpus settings file and how the commands take their defaults from it."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ia_agents.rag import ask_cli, diagnose_cli
from ia_agents.rag.config import DEFAULTS, config_from_argv, load_config


def write_config(directory, data):
    path = Path(directory) / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class LoadConfigTests(unittest.TestCase):
    def test_defaults_when_the_default_file_is_missing(self):
        with patch("ia_agents.rag.config.DEFAULT_CONFIG_PATH", Path("/nonexistent/rag_config.json")):
            self.assertEqual(load_config(), DEFAULTS)

    def test_file_overrides_only_the_keys_it_sets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_config(directory, {"collection": "docs", "retrieval": {"k": 9}})
            config = load_config(path)
        self.assertEqual(config["collection"], "docs")
        self.assertEqual(config["retrieval"], {**DEFAULTS["retrieval"], "k": 9})
        self.assertEqual(config["chunking"], DEFAULTS["chunking"])

    def test_loading_does_not_mutate_the_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            load_config(write_config(directory, {"retrieval": {"k": 9}}))
        self.assertEqual(DEFAULTS["retrieval"]["k"], 5)

    def test_rejects_invalid_files(self):
        invalid = {
            "unknown key": {"colection": "x"},
            "unknown nested key": {"retrieval": {"top_k": 3}},
            "wrong type": {"retrieval": {"k": "5"}},
            "bool is not a number": {"retrieval": {"max_distance": True}},
            "section not an object": {"chunking": 5},
        }
        for name, data in invalid.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    load_config(write_config(directory, data))

    def test_explicit_missing_path_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            load_config("/nonexistent/other.json")

    def test_committed_config_is_valid(self):
        config = load_config()
        self.assertEqual(config["retrieval"]["max_distance"], 1.2)

    def test_config_from_argv_reads_only_the_config_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_config(directory, {"collection": "docs"})
            self.assertEqual(config_from_argv(["question", "--config", str(path), "-k", "2"])["collection"], "docs")


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
        call = self.run_ask([], {"retrieval": {"k": 7, "max_distance": 0.8, "max_prompt_chars": 5000}})
        self.assertEqual((call.kwargs["k"], call.kwargs["max_distance"], call.kwargs["max_prompt_chars"]),
                         (7, 0.8, 5000))

    def test_a_flag_beats_the_config(self):
        call = self.run_ask(["-k", "2", "--max-distance", "0.5"], {"retrieval": {"k": 7, "max_distance": 0.8}})
        self.assertEqual((call.kwargs["k"], call.kwargs["max_distance"]), (2, 0.5))

    def test_a_bad_config_file_stops_the_command(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_config(directory, {"nope": 1})
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exit_info:
                ask_cli.main(["q", "--config", str(path)])
        self.assertEqual(exit_info.exception.code, 2)

    def test_diagnose_takes_k_from_the_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_config(directory, {"retrieval": {"k": 8}})
            result = {"ok": True, "k": 8, "rank": 1, "distance": 0.1, "retrieved": [], "chunks": []}
            with patch("ia_agents.rag.diagnose_cli.chromadb.PersistentClient"), \
                    patch("ia_agents.rag.diagnose_cli.diagnose_retrieval", return_value=result) as fake, \
                    redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                diagnose_cli.main(["check", "q", "chunk", "--config", str(path)])
        self.assertEqual(fake.call_args.kwargs["k"], 8)


if __name__ == "__main__":
    unittest.main()
