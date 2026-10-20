"""Shared defaults and project paths."""

from pathlib import Path

DEFAULT_PARAMS = {"system": None, "temperature": 0.7, "max_tokens": 500} # Used if global config is not set or reset
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "data" / "global_config.json"
HISTORY_PATH = PROJECT_ROOT / "data" / "chat_history.json"

# Transport policy shared across providers; Google expects milliseconds.
CONNECT_TIMEOUT_SECONDS = 10
READ_TIMEOUT_SECONDS = 60
TRANSPORT_MAX_ATTEMPTS = 3
