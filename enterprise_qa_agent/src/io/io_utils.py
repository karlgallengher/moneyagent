from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from enterprise_qa_agent.src.core.text_utils import maybe_fix_mojibake


def load_json(path: str) -> Any:
    return json.loads(maybe_fix_mojibake(Path(path).read_text(encoding="utf-8-sig")))


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(maybe_fix_mojibake(line)))
    return rows


def load_records(path: str) -> list[dict]:
    if str(path).lower().endswith(".jsonl"):
        return load_jsonl(path)
    data = load_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("questions", "data", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        return [data]
    return []

