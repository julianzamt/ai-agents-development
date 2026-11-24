# AI agents project

A RAG agent that answers strictly from a corpus (currently the MDN JavaScript Guide), citing
the chunks it used, plus a general LLM chat client. Shared code is in `src/ia_agents/`;
everything specific to the corpus is in `corpus/` (see [corpus/README.md](corpus/README.md)).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e .        # re-run after adding a command
cp .env.example .env              # set LLM_PROVIDER, EMBEDDING_PROVIDER and credentials
```

- `LLM_PROVIDER`: `google`, `bedrock`, `anthropic`, `openai` or `ollama`.
- `EMBEDDING_PROVIDER`: `google` (default), `openai`, `ollama` or `voyage` (embeddings only).

## Ask (RAG)

```bash
ask "What is a closure in JavaScript?"
ask --query-variants 3 "How do I loop over my own object?"   # also search N LLM rephrasings
```

For each question: retrieve the top `-k` chunks → drop those beyond `--max-distance` (none
left → "no sufficient evidence", no LLM call) → fit them into `--max-prompt-chars` → answer
only from the evidence with `[n]` citations, or abstain → check the citations.

Each question is appended to `data/rag_log.jsonl` (`--log FILE`, `--no-log`) with retrieved,
in-context and cited chunks and an `outcome`: `answered`, `abstained` or `no_evidence`.
Filters (`--filter`, `--document-id`, `--contains`) work as in `retrieve`.

## Build and inspect the index

```bash
chunk corpus/clean                        # chunking report → corpus/clean.chunks.txt, no API calls
ingest corpus/clean                       # chunk, embed and upsert into ChromaDB
retrieve "closures in loops" -k 5         # show matches with distance, id, metadata, text
retrieve --chunk-id Web/JavaScript/Guide/Closures-000-3f89c664
python scripts/clean_mdn_corpus.py        # regenerate corpus/clean/ from corpus/raw/
```

- After changing a cleaning or chunking rule: clean → `chunk` (commit the report) → `ingest`
  → re-check the eval datasets.
- A collection records its embedding provider and model; ingesting with another one fails,
  so use a new `--collection`.
- `retrieve` filters: `--filter KEY[:OP]=VALUE` (`eq`, `ne`, `in`, `nin`, `gt`, `gte`, `lt`,
  `lte`; repeatable), `--section`, `--document-id`, `--contains`, `--regex`, `--max-distance`.
- Plain-text input: `--strategy paragraphs` or `--strategy characters --chunk-size N --overlap N`.

## Evaluate

Questions come from `eval_dataset` in the config (`--dataset FILE` to change it;
`corpus/tests/eval5.json` for quick runs).

```bash
d-retrieval check "What is a closure?" Web/JavaScript/Guide/Closures-000-3f89c664
d-retrieval metrics -k 1 3 5     # hit rate, recall, precision, MRR, overall and by kind (no LLM)
d-retrieval metrics --only-failures   # only cases missing a relevant chunk, and must-not-answer
                                      # cases within --max-distance; -v lists all, with distances
d-retrieval stages -v             # first failing stage: retrieval, context or generation
d-retrieval judge                 # faithfulness and relevance 1–5, scored by an LLM
d-retrieval abstention -v         # must-not-answer questions: declined or (wrongly) answered
d-retrieval cost                  # tokens and USD (needs `pricing` in the config)
d-retrieval latency
```

`stages`, `judge`, `cost` and `latency` run `rag()` on every answerable question, and
`abstention` on every question that must not be answered, so each call costs LLM requests. In
Python, run once with `run_cases()` (`answerable=False` for abstention) and pass the runs to
`stage_report`, `judge_runs`, `cost_report`, `latency_report` or `abstention_report`
(`ia_agents.rag.diagnose`).

## Configuration

`corpus/rag_config.json` holds the values tuned to the corpus and embedding model:
collection, persist directory, eval dataset, `retrieval` (`k`, `max_distance`,
`max_prompt_chars`), `chunking` sizes and `pricing`. It is the only source of these values:
the code has no numeric defaults, so the file is required and must set every key except
`eval_dataset` and `pricing` (`max_distance` may be `null`: no threshold). A CLI flag overrides
it for one run; `--config FILE` selects another file. In Python, pass `k`, `max_distance` and
`max_prompt_chars` to `rag()` explicitly, e.g. from `load_config()["retrieval"]`. Re-measure
`max_distance` and `max_prompt_chars` when the corpus, embedding model or chunking changes.

## Chat client

```bash
client "My name is Julian"                         # keeps history in data/chat_history.json
client --system "Be brief" --temp 0.3              # no prompt: saves the settings
client --batch "Question 1" "Question 2"           # 1–3 independent queries, no history
client --structured support "My account is locked" # validated Pydantic output
client --tools tickets "What is the status of ticket T-001?"
client --reset-chat | --reset-config | --reset-all
```

- Settings go to `data/global_config.json`; with a prompt, flags apply only to that call.
- `--structured-mode prompt` asks for JSON in the instructions instead of the provider's
  native schema; `--structured-attempts N` (default 3) retries invalid output.
- `tickets` tools are local mocks. Tools and structured output can be combined.
- Transport calls retry up to 3 times with exponential backoff and jitter (`llm/config.py`).

From Python:

```python
from ia_agents.llm import LLMClient
from ia_agents.schemas import SupportQuery

result = LLMClient().generate("My account is locked", schema_model=SupportQuery, tool_groups=["tickets"])
result.structured, result.report
```

## Layout

- `src/ia_agents/llm/`: LLM client, providers, retries, persistence. `cli.py`: `client`.
- `src/ia_agents/rag/`: `preprocess` (chunking), `embed` (embeddings, search),
  `pipeline` (`rag()`), `query_variants`, `diagnose`, and one `*_cli.py` per command.
- `src/ia_agents/schemas/`, `src/ia_agents/tools/`: Pydantic models and tool groups.
- `corpus/`: raw and clean corpus, chunk report, `rag_config.json`, eval datasets.
- `tests/`: offline tests (`python -m unittest discover tests`).
- `clases/`: course notebooks (reference only). `data/`: local state, not in Git.

## Design notes

- Retrieved chunks are not deduplicated: on the MDN guide only one real duplicate pair was
  found (a table repeated in the regex cheat sheet), and any useful similarity threshold also
  dropped neighbouring sections with different content. Fix duplicates in the corpus instead.
