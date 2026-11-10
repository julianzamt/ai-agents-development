"""Exercise the retrieval command without external API calls."""

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import chromadb

from ia_agents.rag.retrieve_cli import build_where, main, parse_filter


def make_collection(directory, metadata=None):
    client = chromadb.PersistentClient(path=str(directory))
    collection = client.create_collection("documents", metadata=metadata)
    collection.add(ids=["a"], embeddings=[[1.0, 0.0]], documents=["Alpha text."],
                   metadatas=[{"section_id": "section_01", "section_title": "One",
                               "source": "fixture", "document_id": "doc"}])


class RetrieveCliTests(unittest.TestCase):
    def test_prints_the_embedding_used_for_stored_vectors_and_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            make_collection(directory, {"embedding_provider": "voyage", "embedding_model": "voyage-4"})
            stdout = io.StringIO()
            with patch("ia_agents.rag.retrieve_cli.search", return_value=[[{
                "chunk_id": "a", "text": "Alpha text.", "distance": 0.1,
                "metadata": {"section_id": "section_01", "section_title": "One"},
            }]]) as search, redirect_stdout(stdout):
                main(["question", "--persist-directory", directory])
            self.assertIn("Embedding (stored vectors and queries): voyage / voyage-4",
                          stdout.getvalue())
            self.assertLess(stdout.getvalue().index("Embedding"), stdout.getvalue().index("Query:"))
            search.assert_called_once()

    def test_collection_without_recorded_embedding_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            make_collection(directory)
            stderr = io.StringIO()
            with patch("ia_agents.rag.retrieve_cli.search") as search, \
                    redirect_stderr(stderr), self.assertRaises(SystemExit):
                main(["question", "--persist-directory", directory])
            self.assertIn("does not record its embedding", stderr.getvalue())
            search.assert_not_called()

    def test_filters_are_combined_into_one_where_clause(self):
        with tempfile.TemporaryDirectory() as directory:
            make_collection(directory, {"embedding_provider": "voyage", "embedding_model": "voyage-4"})
            with patch("ia_agents.rag.retrieve_cli.search", return_value=[[]]) as search, \
                    redirect_stdout(io.StringIO()):
                main(["question", "--persist-directory", directory,
                      "--filter", "source=guia-apis", "--filter", "tema=apis",
                      "--section", "section_01"])
            self.assertEqual(search.call_args.kwargs["where"], {"$and": [
                {"source": "guia-apis"}, {"tema": "apis"}, {"section_id": "section_01"},
            ]})

    def test_single_filter_is_not_wrapped_in_and(self):
        with tempfile.TemporaryDirectory() as directory:
            make_collection(directory, {"embedding_provider": "voyage", "embedding_model": "voyage-4"})
            with patch("ia_agents.rag.retrieve_cli.search", return_value=[[]]) as search, \
                    redirect_stdout(io.StringIO()):
                main(["question", "--persist-directory", directory, "--filter", "tema=apis"])
            self.assertEqual(search.call_args.kwargs["where"], {"tema": "apis"})

    def test_no_filters_passes_no_where_clause(self):
        with tempfile.TemporaryDirectory() as directory:
            make_collection(directory, {"embedding_provider": "voyage", "embedding_model": "voyage-4"})
            with patch("ia_agents.rag.retrieve_cli.search", return_value=[[]]) as search, \
                    redirect_stdout(io.StringIO()):
                main(["question", "--persist-directory", directory])
            self.assertIsNone(search.call_args.kwargs["where"])

    def test_filter_value_may_contain_equals_signs(self):
        self.assertEqual(parse_filter("title=a=b"), ("title", "a=b"))

    def test_malformed_filter_is_an_error(self):
        for text in ["tema", "=apis", ""]:
            with self.subTest(text=text), self.assertRaises(Exception):
                parse_filter(text)
        stderr = io.StringIO()
        with patch("ia_agents.rag.retrieve_cli.search") as search, \
                redirect_stderr(stderr), self.assertRaises(SystemExit):
            main(["question", "--filter", "tema"])
        self.assertIn("expected KEY=VALUE", stderr.getvalue())
        search.assert_not_called()

    def test_build_where_is_none_without_filters(self):
        self.assertIsNone(build_where([]))


if __name__ == "__main__":
    unittest.main()
