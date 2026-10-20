"""JSON persistence with atomic writes and lossless byte serialization."""

import base64
import json
import math
import tempfile
from pathlib import Path

from .config import DEFAULT_PARAMS


def json_default(value):
    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode("ascii")}
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def json_object(value):
    if set(value) == {"__bytes__"}:
        return base64.b64decode(value["__bytes__"], validate=True)
    return value


def load_config(path):
    """Load saved preferences, filling missing fields from built-in defaults."""
    if not path.exists():
        return dict(DEFAULT_PARAMS)
    try:
        saved = json.loads(path.read_text())
        if not isinstance(saved, dict) or saved.keys() - DEFAULT_PARAMS.keys():
            raise ValueError("Invalid configuration structure")
        params = DEFAULT_PARAMS | saved
        if params["system"] is not None and not isinstance(params["system"], str):
            raise ValueError("Invalid system")
        temp = params["temperature"]
        if isinstance(temp, bool) or not isinstance(temp, (int, float)) or not math.isfinite(temp) or temp < 0:
            raise ValueError("Invalid temperature")
        if type(params["max_tokens"]) is not int or params["max_tokens"] <= 0:
            raise ValueError("Invalid max_tokens")
        return params
    except (ValueError, TypeError) as error:
        raise ValueError(f"Invalid configuration file {path}: {error}. Use --reset-config to restore defaults.") from error


def load_history(path):
    if not path.exists():
        return {}
    try:
        history = json.loads(path.read_text(), object_hook=json_object)
        if not isinstance(history, dict) or any(not isinstance(messages, list) for messages in history.values()):
            raise ValueError("Invalid history structure")
        return history
    except (ValueError, TypeError) as error:
        raise ValueError(f"Invalid history file {path}: {error}. Use --reset to clear history.") from error


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace atomically so an interrupted write does not truncate the history.
    data = json.dumps(value, default=json_default, ensure_ascii=False, indent=2)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            file.write(data + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
