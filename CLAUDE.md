# Project rules

## Purpose
An agent that answers strictly from a corpus. The shared code is agnostic to the corpus; everything specific to one corpus (its content, cleaning, tuned values, evaluation set) is clearly decoupled from it and lives outside `src/`.

## Language
- The database, corpus, chunk metadata, logs and all stored data are in **English**, even though the course classes (`clases/`) are in Spanish.
- Code identifiers, comments, docstrings, commit messages and user-facing strings are also in English. Translate names from the class notebooks (e.g. `diagnosticar` → `diagnose`, `recuperar` → `search`).

## Comments
- Never mention the course, its classes or notebooks in code comments or docstrings.
- Keep comments and docstrings short and concise.

## Corpus independence
- Code in `src/` must not mention the specific corpus (its name, topic, paths or chunk ids). Corpus-specific material lives in `corpus/`, `scripts/`, the README and `corpus/rag_config.json`.
- Values tuned to a corpus or embedding model (thresholds, budgets, chunk sizes, collection name) belong in `corpus/rag_config.json`, never as code defaults. Code keeps neutral fallbacks.

## Layout
- Shared code lives in `src/ia_agents/`; RAG code in `src/ia_agents/rag/`. Notebooks in `clases/` are course material, not the implementation.
- Tests live in `tests/` and run without external API calls (fake the LLM and embedding providers).

## Commands
- Run tests: `python -m unittest discover tests`
