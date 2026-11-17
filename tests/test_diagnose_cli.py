"""Exercise the diagnose command without external API calls."""

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import chromadb

from ia_agents.rag.diagnose_cli import main


def make_collection(directory):
    client = chromadb.PersistentClient(path=directory)
    client.create_collection("documents")


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
                main(["question", "a", "--persist-directory", directory, *extra])
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
            main(["question", "a", "-k", "0"])
        self.assertEqual(exit_info.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
