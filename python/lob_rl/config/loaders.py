"""Configuration loading utilities."""

from __future__ import annotations

import json
from pathlib import Path


def load_json_config(config_path: str | None) -> dict:
    """Load a JSON run config file.

    Parameters
    ----------
    config_path
        Path to a JSON file. If ``None``, returns an empty dict.

    Returns
    -------
    dict
        Parsed config dictionary.
    """
    if not config_path:
        return {}

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError("Top-level config must be a JSON object")
    return config
