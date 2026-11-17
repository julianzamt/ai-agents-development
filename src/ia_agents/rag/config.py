"""Corpus-tuned settings: neutral built-in defaults, overridden by a JSON file, then by CLI flags."""

import argparse
import copy
import json
from pathlib import Path

from ..llm.config import PROJECT_ROOT

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "corpus" / "rag_config.json"

DEFAULTS: dict = {
    "collection": "documents",
    "persist_directory": "chroma_db",
    "eval_dataset": None,
    "retrieval": {"k": 5, "max_distance": None, "max_prompt_chars": 6000},
    "chunking": {"min_chars": 300, "target_chars": 1200, "max_chars": 2000},
    "pricing": {"input_per_million": None, "output_per_million": None},  # USD; unset means no cost report
}

# Accepted types per key; None is allowed where the built-in default is None.
_TYPES: dict = {
    "collection": str,
    "persist_directory": str,
    "eval_dataset": str,
    "retrieval": {"k": int, "max_distance": (int, float), "max_prompt_chars": int},
    "chunking": {"min_chars": int, "target_chars": int, "max_chars": int},
    "pricing": {"input_per_million": (int, float), "output_per_million": (int, float)},
}


def _check(value, expected, where: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(value, dict):
            raise ValueError(f"{where} must be an object.")
        for key, item in value.items():
            if key not in expected:
                raise ValueError(f"Unknown config key {where}.{key}." if where else f"Unknown config key {key}.")
            _check(item, expected[key], f"{where}.{key}" if where else key)
    elif value is not None and (isinstance(value, bool) or not isinstance(value, expected)):
        raise ValueError(f"Config key {where} has the wrong type: {value!r}.")


def load_config(path: Path | str | None = None) -> dict:
    """Return the defaults overridden by the JSON file.

    Without ``path``, ``corpus/rag_config.json`` is used if it exists. An explicit
    ``path`` must exist. Unknown keys and wrong types raise ValueError.
    """
    config = copy.deepcopy(DEFAULTS)
    explicit = path is not None
    path = Path(path) if explicit else DEFAULT_CONFIG_PATH
    if not path.is_file():
        if explicit:
            raise ValueError(f"Config file not found: {path}")
        return config
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    try:
        _check(loaded, _TYPES, "")
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    for key, value in loaded.items():
        if isinstance(value, dict):
            config[key].update(value)
        else:
            config[key] = value
    return config


def config_from_argv(argv: list[str] | None) -> dict:
    """Read only ``--config`` from argv and load that file, so a CLI can take its defaults from it."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config")
    known, _ = pre_parser.parse_known_args(argv)
    return load_config(known.config)


def config_for(parser: argparse.ArgumentParser, argv: list[str] | None) -> dict:
    """Like ``config_from_argv``, but reports a bad config file through ``parser.error``."""
    try:
        return config_from_argv(argv)
    except ValueError as exc:
        parser.error(str(exc))


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", type=Path,
        help=f"JSON file with corpus-tuned settings (default: corpus/{DEFAULT_CONFIG_PATH.name}, if present)",
    )
