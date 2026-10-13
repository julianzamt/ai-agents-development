# AI agents project

An evolving capstone project for the AI agent development course. Shared code
lives in `src/agentes_ia`; original course notebooks are in `clases/`.

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

You can also run `python -m agentes_ia`. Parameters without a prompt are saved;
with a prompt or batch, they override saved values for that invocation only.
Batch queries use the current configuration but do not read or update history.

Local state is stored in `data/client_state.json`, excluded from Git.
`--history file.json` selects another file; relative paths resolve from the
current working directory.

- `--reset`: clear history.
- `--reset-config`: restore default configuration.
- `--reset-all`: reset both.

## Organization

- `src/agentes_ia/cli.py`: argument parsing and command workflow.
- `src/agentes_ia/llm/`: client, providers, responses and persistence.
- `src/agentes_ia/schemas/`: Pydantic data models.
- `tests/`: offline tests without real API calls.
- `clases/clase2/` and `clases/clase3/`: original course material.
- `data/`: local runtime state.
- `scripts/install-client.sh`: reinstall the command in `.venv`.

Import the client from Python:

```python
from agentes_ia.llm import LLMClient
```

## Tests

```bash
python -m unittest discover -s tests
```

## Class 3 — section 1: JSON Schema and Pydantic

`agentes_ia.schemas.SupportQuery` defines `category`, `urgency`, `summary`
(1 to 200 characters), and the required boolean `requires_escalation`.
It rejects extra fields and implicit type conversions. This is the notebook's
teaching example; the lab will later require a domain-specific model.

Run the demo without credentials or LLM calls:

```bash
python -m agentes_ia.schemas
```

Generate the JSON contract with `SupportQuery.model_json_schema()`.
`model_validate(data)` validates a dictionary; `model_validate_json(text)`
validates JSON. Both return a `SupportQuery` instance or raise
`pydantic.ValidationError`. `model_dump()` and `model_dump_json()` serialize
validated objects. The contract checks structure, not whether a query's
meaning was classified correctly.

This stage defines and validates data. Provider structured outputs belong
to section 2 and have not been implemented yet.
