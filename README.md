# AI agents project

An evolving capstone project for the AI agent development course. Shared code
lives in `src/ia_agents`; original course notebooks are in `clases/`.

## Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Editable installation registers `client` and applies source changes without
reinstallation. Dependencies are declared in `requirements.txt` and included
in the package through `pyproject.toml`.

Copy `.env.example` to `.env` and configure your provider and credentials.
The client reads `.env` from the project root.

## Usage

```bash
client --system "Be brief" --temp 0.3 --max_tokens 100
client "My name is Julian"
client "What is my name?"
client --batch "What is the capital of France?" "What is the capital of Argentina?"
client --help
```

You can also run `python -m ia_agents`. Parameters without a prompt are saved;
with a prompt or batch, they override saved values for that invocation only.
Batch queries use the current configuration but do not read or update history.

TODO: Persist local history for batch calls, including structured batches. Currently,
batch prompts and responses are not saved locally.

Saved configuration lives in `data/global_config.json`; conversation history lives
in `data/chat_history.json`. Both are excluded from Git.
The configuration path is fixed. Use `--history file.json` to select another
history file; relative paths resolve from the current working directory.
Configuration is a JSON object with `system`, `temperature`, and `max_tokens` fields. Missing files or
fields fall back to `DEFAULT_PARAMS`; explicit CLI arguments override saved values.
Arguments alongside a prompt apply only to that request. Without a prompt,
configuration arguments are saved for future calls.

Existing `data/client_state.json` files can be split by copying the `params`
object into the configuration file and the `history` object into the history file.

- `--reset-chat`: clear history.
- `--reset-config`: restore default configuration.
- `--reset-all`: reset both.

Local preparation runs before API calls: configuration is loaded and merged with
CLI overrides, and requested resets are saved immediately. Resets remain in effect
even if the subsequent request fails. Overrides alongside a query apply only to
that query; configuration-only commands save them for future calls.

## Organization

- `src/ia_agents/cli.py`: argument parsing and command workflow.
- `src/ia_agents/llm/`: client, providers, responses and persistence.
- `src/ia_agents/schemas/`: Pydantic data models.
- `tests/`: offline tests without real API calls.
- `clases/clase2/` and `clases/clase3/`: original course material.
- `data/`: local runtime state.
- `scripts/install-client.sh`: reinstall the command in `.venv`.

Import the client from Python:

```python
from ia_agents.llm import LLMClient

result = LLMClient().generate("Explain Python briefly")
print(result.report)
```

`generate()` returns a `GenerationResult` with `response` (the provider SDK response),
`history` (the updated conversation), `structured` (the validated model, or `None`),
`latency`, and `report`. Select tools with `tool_groups=["tickets"]` and structured
output with `schema_model=SupportQuery`; both can be used together.
`generate_batch(prompts, **options)` returns a list of the same results in input
order. Each item uses the single-query generation flow. The CLI saves history for
individual requests only.

## Document ingestion

Configure `GOOGLE_API_KEY` and optionally `GOOGLE_EMBEDDING_MODEL_ID` in `.env`.
Refresh the editable installation to register the separate `ingest` command:

```bash
python3 -m pip install -e .
ingest tests/test_corpus.txt --strategy paragraphs --max-chars 1000 --output data/charly_embeddings.json
```

Alternatively, run `python3 -m ia_agents.ingest_cli` with the same arguments.
Character chunking uses `--strategy characters --chunk-size 1000 --overlap 150`.
Add `--print-chunks` to inspect the chunks. `--document-id` and `--source` override
the metadata defaults (input stem and input path). Paragraphs larger than
`--max-chars` remain intact. The command makes real Google API requests and writes
a JSON list with text, chunk ID, document ID, position, source, UTC ingestion time,
and vector. An existing output file is replaced after embedding succeeds.

## Tests

```bash
python -m unittest discover -s tests
```

## Class 3 — section 1: JSON Schema and Pydantic

`ia_agents.schemas.SupportQuery` defines `category`, `urgency`, `summary`
(1 to 200 characters), and the required boolean `requires_escalation`.
It rejects extra fields and implicit type conversions. This is the notebook's
teaching example; the lab will later require a domain-specific model.

Run the demo without credentials or LLM calls:

```bash
python -m ia_agents.schemas
```

Generate the JSON contract with `SupportQuery.model_json_schema()`.
`model_validate(data)` validates a dictionary; `model_validate_json(text)`
validates JSON. Both return a `SupportQuery` instance or raise
`pydantic.ValidationError`. `model_dump()` and `model_dump_json()` serialize
validated objects. The contract checks structure, not whether a query's
meaning was classified correctly.

This stage defines and validates data. Provider structured outputs belong
to section 2 and have not been implemented yet.

## Class 3 — section 2: native structured outputs

```bash
client --structured support --prompt "My account has been locked for two days. Please escalate." --temp 0
```

The client sends `SupportQuery`'s JSON Schema through the provider's native
structured-output parameter, validates the returned JSON with Pydantic, and
prints the validated object alongside latency and token counts. Saved parameters
and conversation history still apply. Invalid output raises a clear error and
is not added to history. `--structured` accepts an individual prompt or a batch:

```bash
client --batch "My bill was charged twice" "I cannot log in" --structured support
```

Structured batches run independently with up to three workers and print validated
results in input order. They use saved configuration without reading or updating
conversation history. `--structured-mode` and `--structured-attempts` apply to each
prompt separately. If any query exhausts its attempts, the command fails without
printing partial batch results.

From Python:

```python
from ia_agents.llm import LLMClient
from ia_agents.schemas import SupportQuery

result = LLMClient().generate(
    "Classify this issue: my account is locked", schema_model=SupportQuery,
)
query = result.structured
```

Native mappings: Bedrock `outputConfig.textFormat`, Anthropic
`output_config.format`, OpenAI `response_format`, Google
`response_mime_type` and `response_json_schema`, Ollama `format`.
Support depends on the selected model and provider's JSON Schema subset.
Unsupported requests fail explicitly; the universal prompt-only mode must be selected explicitly. Schema validation checks structure,
not factual correctness. Existing system prompts can conflict with this task;
use suitable instructions or override `--system` for the request.


## Class 3 — section 3: structured validation retries

Structured generation allows up to three validation attempts by default. Invalid
JSON or schema violations trigger another request containing the original task,
previous output, validation errors, and schema. Exhaustion raises a clear error
with the last validation exception as its cause. Only the successful final turn
is saved in conversation history.

```bash
client --structured support --structured-attempts 3 --prompt "My account is locked"
client --structured support --structured-mode prompt --temp 0 --prompt "My account is locked"
```

`--structured-attempts` (alias `--max-attempts-for-structured-response`) defaults
to 3. Without a prompt or batch, it saves a preference; with a query, it overrides
the saved value for that query. Existing configuration files use the default
when `max_attempts_for_structured_response` is absent.

`--structured-mode prompt` requests JSON through instructions instead of native
provider schema enforcement, supporting the universal pattern from the notebook.
Native mode remains the default; unsupported native requests are not silently
retried in prompt mode. Python callers can pass `max_attempts_for_structured_response=3` and `native=False`
to `generate(..., schema_model=SupportQuery)`.

Validation attempts are separate from transport retries: each validation attempt
can use the existing bounded network retry policy. Report latency includes all
attempts; displayed token counts belong to the final response, not the total
consumed by failed attempts. Validation repair does not add backoff delays.

## Class 3 — section 4: explicit tool calling

```bash
client --tools tickets --system "Use the ticket tools and report their results accurately." --prompt "What is the status of ticket T-999?"
client --tools tickets --prompt "Escalate ticket T-001 because the service has been down for two hours."
```

The registered tools are `get_ticket_status(ticket_id)` and
`escalate_ticket(ticket_id, reason)`. Both are local mocks: escalation does not
change a real ticket or persist a new status. Unknown IDs return `Not found`.
Arguments are validated with strict Pydantic models before execution.

The model requests functions; application code executes them and sends results
back through provider-native messages. Google AFC stays disabled. The loop
supports up to three execution rounds plus a final response, rejects unknown
tools, and saves the complete successful exchange, including signatures and
tool results for individual queries. Tool mode accepts an individual prompt or
`--batch`, and can be combined with `--structured`. Saved parameters apply as usual.

```bash
client --batch "Check ticket T-001" "Check ticket T-002" --tools tickets
client --batch "Classify ticket T-001 using its status" "Classify ticket T-002 using its status" --tools tickets --structured support
```

Each batch item runs its own tool loop with up to three workers, and reports appear
in input order. Batch conversations are not saved locally. If a query fails, the
batch command fails without printing partial results; other queries may already
have executed their tools.

When tools and structured output are combined, the tool conversation completes
first. A separate request converts its transcript into the selected schema, using
`--structured-mode` and `--structured-attempts`. Validation retries do not rerun
tools. Individual queries save both stages in chat history.

In Python, `LLMClient().generate(prompt, tool_groups=["tickets"])` returns the same
`GenerationResult` as other modes, including the final SDK response and updated
history. Add `schema_model=SupportQuery` for validated structured output. Report latency covers the whole loop; token
counts currently describe only the final response. No real API calls are made
by the automated tests.

## Class 3 — section 5: timeouts and exponential backoff with jitter

Transport calls allow at most three attempts. Retryable failures (timeouts,
rate limits and recognized temporary service errors) wait `2 ** attempt` plus
a random value between 0 and 1 second: 1–2 seconds before the second attempt,
2–3 before the third. There is no delay after success or final failure.
Validation and permission errors are not retried.

The shared policy lives in `src/ia_agents/llm/config.py`: 60-second read/request
inactivity timeouts, a 10-second Bedrock connection timeout and three transport
attempts. Google receives its timeout in milliseconds. SDK retries are disabled
where configurable so they do not multiply the application retries.

These are transport timeouts, not a strict deadline for the complete command:
network phases, structured repair attempts and tool rounds can increase total
elapsed time. Structured validation retries remain distinct from transport
backoff. Offline tests cover timeout retries, jitter, bounded exhaustion and
provider timeout/retry settings.

`--tools` accepts one or more registered tool group names, for example
`client "Check T-001" --tools tickets`. Currently only `tickets` is registered.
Additional groups can be registered in `TOOL_GROUPS`; selected groups are combined
without duplicate tools.
