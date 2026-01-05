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


if "HF_LEROBOT_HOME" not in os.environ:
    # This file lives at: <openpi repo>/src/openpi/__init__.py
    # So the repo root is always two parents up from the package dir.
    repo_root = Path(__file__).resolve().parents[2]
    os.environ["HF_LEROBOT_HOME"] = str(repo_root / "lerobot_datasets")

