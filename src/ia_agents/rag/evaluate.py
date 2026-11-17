"""Load and validate an evaluation dataset: questions with the chunks that should answer them."""

import json
from pathlib import Path

def load_eval_dataset(path: Path | str) -> list[dict]:
    """Read a JSON list of cases and check each one.

    A case has ``question`` (text), ``relevant_chunks`` (chunk ids; empty when the corpus cannot
    answer), ``must_answer`` (True exactly when there are relevant chunks) and an optional ``kind``.
    """
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path}: expected a non-empty JSON list of cases.")
    seen: set[str] = set()
    for number, case in enumerate(cases, start=1):
        where = f"{path}: case {number}"
        if not isinstance(case, dict):
            raise ValueError(f"{where} is not an object.")
        question = case.get("question")
        chunks = case.get("relevant_chunks")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{where} needs a non-empty question.")
        if question in seen:
            raise ValueError(f"{where} repeats the question {question!r}.")
        seen.add(question)
        if not isinstance(chunks, list) or not all(isinstance(chunk, str) and chunk for chunk in chunks):
            raise ValueError(f"{where} needs relevant_chunks as a list of chunk ids.")
        if len(chunks) != len(set(chunks)):
            raise ValueError(f"{where} repeats a chunk id.")
        if case.get("must_answer") is not bool(chunks):
            raise ValueError(f"{where}: must_answer must be true exactly when relevant_chunks is not empty.")
    return cases
