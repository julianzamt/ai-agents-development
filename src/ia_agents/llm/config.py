"""Providers shared defaults and project paths."""

from pathlib import Path

DEFAULT_PROVIDER="google"

PROVIDERS_ENV_VAR_NAMES: dict[str, str | None] = {
    "bedrock": "AWS_BEARER_TOKEN_BEDROCK",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "ollama": None,
}

# Transport policy shared across providers; Google expects milliseconds.
CONNECT_TIMEOUT_SECONDS = 10
READ_TIMEOUT_SECONDS = 60 # Max wait for receiving a response. Used also for time in between chunks if streaming.
TRANSPORT_MAX_ATTEMPTS = 3

DEFAULT_SYSTEM = None
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 500

DEFAULT_PARAMS = {
    "system": DEFAULT_SYSTEM,
    "temperature": DEFAULT_TEMPERATURE,
    "max_tokens": DEFAULT_MAX_TOKENS,
} # Used if global config is not set or reset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "data" / "global_config.json"
HISTORY_PATH = PROJECT_ROOT / "data" / "chat_history.json"
