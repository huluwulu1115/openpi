"""OpenPI package initialization.

Local development convenience:
If the `HF_LEROBOT_HOME` environment variable is not set, we default it to a
`lerobot_datasets/` folder at the OpenPI repo root. This matches the default
output location used by `examples/droid/convert_eurekaworld_sim_dataset_to_lerobot.py`
and avoids having to manually export `HF_LEROBOT_HOME` for training scripts.

To override this behavior, set `HF_LEROBOT_HOME` in your shell before running
OpenPI scripts.
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_openpi_repo_root(start: Path) -> Path | None:
    """Best-effort search for the OpenPI repo root (directory containing pyproject.toml)."""
    for p in (start, *start.parents):
        pyproject = p / "pyproject.toml"
        if not pyproject.exists():
            continue
        try:
            txt = pyproject.read_text()
        except OSError:
            continue
        # Lightweight check: avoid toml parsing at import time.
        if 'name = "openpi"' in txt or 'name="openpi"' in txt:
            return p
    return None


if "HF_LEROBOT_HOME" not in os.environ:
    repo_root = _find_openpi_repo_root(Path(__file__).resolve())
    if repo_root is not None:
        os.environ["HF_LEROBOT_HOME"] = str(repo_root / "lerobot_datasets")

