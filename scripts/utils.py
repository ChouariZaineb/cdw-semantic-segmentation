from __future__ import annotations
"""Shared utility helpers for config loading, seeding, and artifact serialization."""

from datetime import datetime
from pathlib import Path
import json
import random
from typing import Any

import numpy as np
import torch
import yaml


def load_yaml_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping: {p}")
    return raw


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_experiment_dir(out_root: str | Path, exp_name: str | None = None) -> Path:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if exp_name is None or not str(exp_name).strip():
        exp_name = datetime.now().strftime("exp_%Y%m%d_%H%M%S")
    out_dir = out_root / str(exp_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    p = Path(path)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def dump_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
