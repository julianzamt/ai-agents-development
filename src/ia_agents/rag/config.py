"""Corpus-tuned settings: a required JSON file next to the corpus, overridden by CLI flags."""

import argparse
import copy
import json
from pathlib import Path

from ..llm.config import PROJECT_ROOT

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "corpus" / "rag_config.json"

# Accepted types per key. Every key is required except those in _OPTIONAL.
_TYPES: dict = {
    "collection": str,
    "persist_directory": str,
    "eval_dataset": str,
    "judge_dataset": str,
    "retrieval": {"k": int, "max_distance": (int, float), "max_prompt_chars": int},
    "chunking": {"min_chars": int, "target_chars": int, "max_chars": int},
    "pricing": {"input_per_million": (int, float), "output_per_million": (int, float)},
}

# Keys a config may leave out, and the value that means "not set".
_OPTIONAL: dict = {
    "eval_dataset": None,
    "judge_dataset": None,  # default dataset for `d-retrieval judge`; falls back to eval_dataset
    "pricing": {"input_per_million": None, "output_per_million": None},  # USD; unset means no cost report
}

# Required keys that may be null: no relevance threshold.
_NULLABLE = {"retrieval.max_distance"}


def _check(value, expected, where: str, optional=None) -> None:
    if isinstance(expected, dict):
        if not isinstance(value, dict):
            raise ValueError(f"{where} must be an object.")
        for key, item in value.items():
            if key not in expected:
                raise ValueError(f"Unknown config key {where}.{key}." if where else f"Unknown config key {key}.")
        optional = optional or {}
        for key, item_type in expected.items():
            path = f"{where}.{key}" if where else key
            if key in value:
                _check(value[key], item_type, path, optional.get(key) if isinstance(optional.get(key), dict) else None)
            elif key not in optional:
                raise ValueError(f"Missing config key {path}.")
    elif value is None:
        if where not in _NULLABLE and where.split(".")[0] not in _OPTIONAL:
            raise ValueError(f"Config key {where} must not be null.")
    elif isinstance(value, bool) or not isinstance(value, expected):
        raise ValueError(f"Config key {where} has the wrong type: {value!r}.")


def load_config(path: Path | str | None = None) -> dict:
    """Load and validate the corpus settings file (``corpus/rag_config.json`` without ``path``).

    The file is required and must set every key except ``eval_dataset``, ``judge_dataset`` and ``pricing``.
    A missing file, unknown or missing keys and wrong types raise ValueError.
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise ValueError(f"Config file not found: {path}. Create it or pass --config FILE.")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    try:
        _check(loaded, _TYPES, "", _OPTIONAL)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    config = copy.deepcopy(_OPTIONAL)
    for key, value in loaded.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
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
        help=f"JSON file with corpus-tuned settings (default: corpus/{DEFAULT_CONFIG_PATH.name})",
    )
