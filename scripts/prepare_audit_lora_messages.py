from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "processed/lora_datasets/audit_weak.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "processed/lora_datasets"

SYSTEM_PROMPT = """你是证据审查 Agent。你的任务是根据题目、选项/任务、槽位和已检索证据，判断当前证据是否足够支持判断。
只输出 JSON，不要输出解释性闲聊。JSON 字段包括：
status: satisfied 或 need_more
condition_satisfied: true、false 或 null
filled: 已经从证据中填充的槽位列表
missing: 仍缺少的槽位列表
evidence_marks: 对证据块的 useful/partial/irrelevant 标注
next_action: stop、search_more、change_query、change_section 等
reason: 简短说明"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_to_messages(sample: dict[str, Any]) -> dict[str, Any]:
    model_input = {
        "kind": sample.get("input", {}).get("kind"),
        "option": sample.get("input", {}).get("option"),
        "claim": sample.get("input", {}).get("claim"),
        "option_subject": sample.get("input", {}).get("option_subject"),
        "question_condition": sample.get("input", {}).get("question_condition"),
        "task": sample.get("input", {}).get("task"),
        "target_doc_ids": sample.get("input", {}).get("target_doc_ids") or [],
        "slots": sample.get("input", {}).get("slots") or [],
        "evidence": sample.get("input", {}).get("evidence") or [],
        "notes": sample.get("input", {}).get("notes") or [],
    }
    return {
        "id": sample.get("id"),
        "source": sample.get("source"),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(model_input, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps(sample.get("output") or {}, ensure_ascii=False)},
        ],
    }


def split_by_qid(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    qids = sorted({str((row.get("source") or {}).get("qid") or "") for row in rows})
    rng = random.Random(seed)
    rng.shuffle(qids)
    val_count = max(1, round(len(qids) * val_ratio)) if qids else 0
    val_qids = set(qids[:val_count])
    train = [row for row in rows if str((row.get("source") or {}).get("qid") or "") not in val_qids]
    val = [row for row in rows if str((row.get("source") or {}).get("qid") or "") in val_qids]
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert weak audit JSONL into chat messages JSONL for LoRA.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    samples = load_jsonl(args.input)
    messages = [sample_to_messages(sample) for sample in samples]
    train, val = split_by_qid(messages, args.val_ratio, args.seed)

    train_path = args.output_dir / "audit_messages_train.jsonl"
    val_path = args.output_dir / "audit_messages_val.jsonl"
    dump_jsonl(train_path, train)
    dump_jsonl(val_path, val)

    print(json.dumps({
        "input": str(args.input),
        "train": str(train_path),
        "val": str(val_path),
        "samples": len(messages),
        "train_samples": len(train),
        "val_samples": len(val),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
