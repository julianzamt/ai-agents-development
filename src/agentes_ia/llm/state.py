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


def new_state():
    return {"params": dict(DEFAULT_PARAMS), "history": {}}


def load_state(path):
    if not path.exists():
        return new_state()
    try:
        state = json.loads(path.read_text(), object_hook=json_object)
        params = state["params"]
        if not isinstance(state["history"], dict) or set(params) != set(DEFAULT_PARAMS):
            raise ValueError("Invalid state structure")
        if params["system"] is not None and not isinstance(params["system"], str):
            raise ValueError("Invalid system")
        temp = params["temperature"]
        if isinstance(temp, bool) or not isinstance(temp, (int, float)) or not math.isfinite(temp) or temp < 0:
            raise ValueError("Invalid temperature")
        if type(params["max_tokens"]) is not int or params["max_tokens"] <= 0:
            raise ValueError("Invalid max_tokens")
        if any(not isinstance(messages, list) for messages in state["history"].values()):
            raise ValueError("Invalid history")
        return state
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Invalid state file {path}: {error}. Use --reset-all to restore defaults.") from error


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace atomically so an interrupted write does not truncate the history.
    data = json.dumps(state, default=json_default, ensure_ascii=False, indent=2)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            file.write(data + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
