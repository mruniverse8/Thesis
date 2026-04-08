from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_value: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    anchor = base_dir or PROJECT_ROOT
    return (anchor / path).resolve()


def ensure_dir(path_value: str | Path) -> Path:
    path = Path(path_value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path_value: str | Path) -> dict[str, Any]:
    path = Path(path_value)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in YAML config: {path}")
    return data


def dump_yaml(path_value: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path_value)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def read_jsonl(path_value: str | Path) -> list[dict[str, Any]]:
    path = Path(path_value)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected a JSON object on line {line_number} of {path}")
            records.append(payload)
    return records


def write_json(path_value: str | Path, payload: Any) -> None:
    path = Path(path_value)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=False)
        handle.write("\n")


def write_jsonl(path_value: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path_value)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
