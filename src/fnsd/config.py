from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_CONFIG = Path("configs/default.yaml")


def load_confg(path: str | Path | None = None) -> dict:
    """Load YAML config. Paths inside are relative to the working directory (repo root)."""
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
