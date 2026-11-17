# Project rules

## Language
- The database, corpus, chunk metadata, logs and all stored data are in **English**, even though the course classes (`clases/`) are in Spanish.
- Code identifiers, comments, docstrings, commit messages and user-facing strings are also in English. Translate names from the class notebooks (e.g. `diagnosticar` → `diagnose`, `recuperar` → `search`).

## Comments
- Never mention the course, its classes or notebooks in code comments or docstrings.
- Keep comments and docstrings short and concise.

## Layout
- Shared code lives in `src/ia_agents/`; RAG code in `src/ia_agents/rag/`. Notebooks in `clases/` are course material, not the implementation.
- Tests live in `tests/` and run without external API calls (fake the LLM and embedding providers).

## Commands
- Run tests: `python -m unittest discover tests`
