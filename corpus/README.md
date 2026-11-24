# Corpus

Everything specific to the corpus (the MDN JavaScript Guide) lives here, apart from the shared code in `src/`.

- `raw/`: the original MDN Markdown files (and their images), from the MDN content repository, licensed CC-BY-SA 2.5. Never edited by hand.
- `clean/`: generated copy of `raw/` with front matter, MDN macros, HTML tables and links cleaned. This is what gets chunked and embedded. Regenerate it with `python scripts/clean_mdn_corpus.py`.
- `clean.chunks.txt`: generated report of the current chunking of `clean/` (settings, size statistics, every chunk id). Regenerate it with `chunk corpus/clean`; a test fails if it is stale.
- `rag_config.json`: values tuned to this corpus and embedding model: collection name, persist directory, retrieval limits, chunk sizes, default evaluation dataset.
- `tests/`: evaluation datasets (question, expected chunk ids, kind).
  - `clean.eval.json`: the full set, used by default (`eval_dataset` in the config). Kinds:
    - `direct`, `colloquial`: answerable, asked in documentation terms or in a user's own words.
    - `multi_chunk`: the answer needs several chunks combined.
    - `partial`: only part is answerable; the answer should cover it and say the rest is not in the corpus. Its relevant chunks cover the answerable part.
    - `false_premise`: the question assumes something the corpus contradicts; the answer should correct it.
    - `ambiguous`: the question has several readings the corpus covers; the answer should address them or ask which is meant. Its relevant chunks cover each reading.
    - `missing_detail`, `out_of_domain`: must not be answered (a JavaScript detail the corpus lacks, or another topic).
  - `eval5.json`: a 5-question subset of it (one and several relevant chunks, one question that must not be answered), for quick runs, e.g. `d-retrieval judge --dataset corpus/tests/eval5.json`.

Chunk ids in the datasets must exist in `clean.chunks.txt`. After changing a cleaning or chunking rule, regenerate `clean/` and the report, re-ingest, and re-check the datasets.
