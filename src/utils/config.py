"""Small helpers to load YAML configuration files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "configs/train.yaml") -> dict[str, Any]:
    """Load a YAML config and apply environment overrides.

    ``MLFLOW_TRACKING_URI`` takes precedence over the value stored in the
    config, which makes the same config usable locally and on the VM.
    """
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        config.setdefault("mlflow", {})["tracking_uri"] = env_uri

    return config


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p
