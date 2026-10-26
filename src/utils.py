"""Shared helpers: configuration, seeding, paths and JSON output."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def p(*parts: str) -> Path:
    """Path relative to the repository root; parent folders are created."""
    out = ROOT.joinpath(*parts)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def set_seed(seed: int, threads: int | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if threads:
            torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(False)
    except ImportError:
        pass


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=default)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
