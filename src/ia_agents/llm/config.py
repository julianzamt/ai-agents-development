"""Shared defaults and project paths."""

from pathlib import Path

DEFAULT_PARAMS = {"system": None, "temperature": 0.7, "max_tokens": 500}
PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATE_PATH = PROJECT_ROOT / "data" / "client_state.json"

# Transport policy shared across providers; Google expects milliseconds.
CONNECT_TIMEOUT_SECONDS = 10
READ_TIMEOUT_SECONDS = 60
TRANSPORT_MAX_ATTEMPTS = 3
