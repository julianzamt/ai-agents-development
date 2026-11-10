"""Exercise the retrieval command without external API calls."""

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import chromadb

import argparse

from ia_agents.rag.embed import get_chunks
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
        self.assertEqual(parse_filter("title=a=b"), {"title": "a=b"})

    def test_filter_operators_map_to_chroma_clauses(self):
        self.assertEqual(parse_filter("tema:ne=apis"), {"tema": {"$ne": "apis"}})
        self.assertEqual(parse_filter("tema:in=apis, rag"), {"tema": {"$in": ["apis", "rag"]}})
        self.assertEqual(parse_filter("tema:nin=apis"), {"tema": {"$nin": ["apis"]}})
        self.assertEqual(parse_filter("year:gte=2020"), {"year": {"$gte": 2020.0}})
        self.assertEqual(parse_filter("year:lt=1.5"), {"year": {"$lt": 1.5}})

    def test_malformed_filter_is_an_error(self):
        for text in ["tema", "=apis", "", "tema:bogus=x", "year:gt=abc"]:
            with self.subTest(text=text), self.assertRaises(argparse.ArgumentTypeError):
                parse_filter(text)
        stderr = io.StringIO()
        with patch("ia_agents.rag.retrieve_cli.search") as search, \
                redirect_stderr(stderr), self.assertRaises(SystemExit):
            main(["question", "--filter", "tema"])
        self.assertIn("expected KEY=VALUE", stderr.getvalue())
        search.assert_not_called()

    def test_build_where_is_none_without_filters(self):
        self.assertIsNone(build_where([]))

    def test_contains_and_regex_are_passed_as_where_document(self):
        with tempfile.TemporaryDirectory() as directory:
            make_collection(directory, {"embedding_provider": "voyage", "embedding_model": "voyage-4"})
            with patch("ia_agents.rag.retrieve_cli.search", return_value=[[]]) as search, \
                    redirect_stdout(io.StringIO()):
                main(["question", "--persist-directory", directory, "--contains", "Alpha"])
            self.assertEqual(search.call_args.kwargs["where_document"], {"$contains": "Alpha"})
            with patch("ia_agents.rag.retrieve_cli.search", return_value=[[]]) as search, \
                    redirect_stdout(io.StringIO()):
                main(["question", "--persist-directory", directory,
                      "--contains", "Alpha", "--regex", "^A"])
            self.assertEqual(search.call_args.kwargs["where_document"], {"$and": [
                {"$contains": "Alpha"}, {"$regex": "^A"},
            ]})

    def test_chunk_id_lookup_skips_search(self):
        with tempfile.TemporaryDirectory() as directory:
            make_collection(directory, {"embedding_provider": "voyage", "embedding_model": "voyage-4"})
            stdout = io.StringIO()
            with patch("ia_agents.rag.retrieve_cli.search") as search, \
                    redirect_stdout(stdout):
                main(["--persist-directory", directory, "--chunk-id", "a"])
            search.assert_not_called()
            self.assertIn("id=a", stdout.getvalue())
            self.assertIn("Alpha text.", stdout.getvalue())
            self.assertNotIn("distance=", stdout.getvalue())
            self.assertNotIn("Embedding", stdout.getvalue())

    def test_chunk_id_lookup_reports_no_match(self):
        with tempfile.TemporaryDirectory() as directory:
            make_collection(directory)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                main(["--persist-directory", directory, "--chunk-id", "missing"])
            self.assertIn("No chunks matched.", stdout.getvalue())

    def test_invalid_regex_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as directory:
            make_collection(directory)
            stderr = io.StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit):
                main(["--persist-directory", directory, "--chunk-id", "a", "--regex", "("])
            self.assertIn("error", stderr.getvalue().lower())

    def test_queries_and_chunk_id_together_are_an_error(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit):
            main(["question", "--chunk-id", "a"])
        self.assertIn("not both", stderr.getvalue())

    def test_no_queries_and_no_chunk_id_is_an_error(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit):
            main([])
        self.assertIn("Give one or more search queries", stderr.getvalue())


class GetChunksTests(unittest.TestCase):
    def test_fetches_by_id_and_filters_without_embedding(self):
        with tempfile.TemporaryDirectory() as directory:
            make_collection(directory)
            collection = chromadb.PersistentClient(path=directory).get_collection("documents")
            chunks = get_chunks(collection, ids=["a"], where={"section_id": "section_01"},
                                where_document={"$contains": "Alpha"})
            self.assertEqual(chunks, [{"chunk_id": "a", "text": "Alpha text.",
                                       "metadata": {"section_id": "section_01", "section_title": "One",
                                                    "source": "fixture", "document_id": "doc"}}])
            self.assertEqual(get_chunks(collection, ids=["a"], where={"section_id": "other"}), [])


if __name__ == "__main__":
    unittest.main()
