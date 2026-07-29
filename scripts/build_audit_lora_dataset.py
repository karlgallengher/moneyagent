from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIRS = [
    PROJECT_ROOT / "processed/agent_debug_clean",
    PROJECT_ROOT / "processed/agent_debug_window",
]
DEFAULT_OUTPUT = PROJECT_ROOT / "processed/lora_datasets/audit_weak.jsonl"

MOJIBAKE_MARKERS = ("绗", "锛", "鍙", "鐨", "涓", "浜", "惧", "鏄", "妗", "�")


def compact_text(text: Any, limit: int) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    return value[:limit]


def mojibake_score(text: str) -> int:
    return sum(text.count(marker) for marker in MOJIBAKE_MARKERS)


def chinese_score(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def repair_mojibake(text: str) -> str:
    if not text:
        return text
    before_bad = mojibake_score(text)
    if before_bad <= 0:
        return text
    candidates: list[str] = []
    for encoding in ("gb18030", "gbk"):
        try:
            candidates.append(text.encode(encoding, errors="ignore").decode("utf-8", errors="ignore"))
        except UnicodeError:
            continue
    best = text
    best_bad = before_bad
    best_cn = chinese_score(text)
    for candidate in candidates:
        candidate_bad = mojibake_score(candidate)
        candidate_cn = chinese_score(candidate)
        if candidate and (candidate_bad < best_bad or (candidate_bad == best_bad and candidate_cn > best_cn)):
            best = candidate
            best_bad = candidate_bad
            best_cn = candidate_cn
    return best


def clean_obj(obj: Any) -> Any:
    if isinstance(obj, str):
        return repair_mojibake(obj)
    if isinstance(obj, list):
        return [clean_obj(item) for item in obj]
    if isinstance(obj, dict):
        return {clean_obj(key): clean_obj(value) for key, value in obj.items()}
    return obj


def extract_backtick_json(text: str, label: str) -> dict:
    match = re.search(rf"{re.escape(label)}:\s*`([^`]*)`", text)
    if not match:
        return {}
    raw = match.group(1).strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def extract_backtick_value(text: str, label: str) -> str:
    match = re.search(rf"{re.escape(label)}:\s*`([^`]*)`", text)
    return match.group(1).strip() if match else ""


def extract_json_block(text: str, heading: str) -> Any:
    pattern = rf"{re.escape(heading)}\s*```json\s*(.*?)\s*```"
    match = re.search(pattern, text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def infer_qid(path: Path, text: str) -> str:
    match = re.search(r"# .*?Debug:\s*([A-Za-z0-9_]+)", text)
    if match:
        return match.group(1)
    stem = path.stem
    for prefix in ("clean_", "window_", "langgraph_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def evidence_items(unit: dict, max_items: int, max_chars: int) -> list[dict]:
    items = unit.get("evidence") or []
    if not isinstance(items, list):
        return []
    marks = unit.get("evidence_marks") or {}
    selected: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("evidence_id") or "")
        mark = (marks.get(evidence_id) or {}).get("mark") if isinstance(marks, dict) else None
        selected.append({
            "evidence_id": evidence_id,
            "doc_id": str(item.get("doc_id") or ""),
            "heading": " > ".join(item.get("heading_path") or []),
            "source": item.get("source"),
            "mark": mark,
            "text": compact_text(item.get("text") or "", max_chars),
        })
        if len(selected) >= max_items:
            break
    return selected


def normalize_audit(audit: Any) -> dict | None:
    if not isinstance(audit, dict):
        return None
    output = {
        "status": audit.get("status"),
        "condition_satisfied": audit.get("condition_satisfied"),
        "filled": audit.get("filled") if isinstance(audit.get("filled"), list) else [],
        "missing": audit.get("missing") if isinstance(audit.get("missing"), list) else [],
        "evidence_marks": audit.get("evidence_marks") if isinstance(audit.get("evidence_marks"), list) else [],
        "next_action": audit.get("next_action"),
        "reason": audit.get("reason"),
    }
    return output


def unit_to_sample(path: Path, qid: str, answer: str, usage: dict, unit: dict, max_evidence: int, evidence_chars: int) -> dict | None:
    audit = normalize_audit(unit.get("audit"))
    if not audit:
        return None
    sample_id = f"{qid}:{unit.get('unit_id') or len(str(unit))}"
    return {
        "id": sample_id,
        "task_type": "audit",
        "source": {
            "log_path": str(path),
            "qid": qid,
            "final_answer": answer,
            "token_usage": usage,
        },
        "input": {
            "kind": unit.get("kind"),
            "option": unit.get("option"),
            "claim": unit.get("claim"),
            "option_subject": unit.get("option_subject"),
            "question_condition": unit.get("question_condition"),
            "task": unit.get("task"),
            "target_doc_ids": unit.get("target_doc_ids") or [],
            "slots": unit.get("slots") or [],
            "evidence": evidence_items(unit, max_evidence, evidence_chars),
            "notes": unit.get("notes") or [],
        },
        "output": audit,
        "quality": {
            "label": "weak_from_agent_log",
            "needs_manual_review": True,
            "warning": "This sample is generated from agent traces and may contain wrong judgments.",
        },
    }


def samples_from_log(path: Path, max_evidence: int, evidence_chars: int) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    qid = infer_qid(path, text)
    answer = extract_backtick_value(text, "answer") or extract_backtick_value(text, "final_answer")
    usage = extract_backtick_json(text, "token_usage")
    units = extract_json_block(text, "## Detailed Units")
    if units is None:
        units = extract_json_block(text, "## Unit Results")
    if not isinstance(units, list):
        return []
    samples: list[dict] = []
    for unit in units:
        if not isinstance(unit, dict):
            continue
        sample = unit_to_sample(path, qid, answer, usage, unit, max_evidence, evidence_chars)
        if sample:
            samples.append(sample)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Build weak audit LoRA JSONL from agent debug markdown logs.")
    parser.add_argument("--log-dir", action="append", type=Path, default=None, help="Directory containing debug .md logs. Can be repeated.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-evidence", type=int, default=8)
    parser.add_argument("--evidence-chars", type=int, default=900)
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of samples.")
    args = parser.parse_args()

    log_dirs = args.log_dir or DEFAULT_LOG_DIRS
    paths: list[Path] = []
    for log_dir in log_dirs:
        if log_dir.exists():
            paths.extend(sorted(log_dir.glob("*.md")))

    samples: list[dict] = []
    for path in paths:
        samples.extend(samples_from_log(path, args.max_evidence, args.evidence_chars))
        if args.limit and len(samples) >= args.limit:
            samples = samples[:args.limit]
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        for sample in samples:
            f.write(json.dumps(clean_obj(sample), ensure_ascii=False) + "\n")

    status_counts: dict[str, int] = {}
    for sample in samples:
        status = str((sample.get("output") or {}).get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    print(json.dumps({
        "output": str(args.output),
        "logs": len(paths),
        "samples": len(samples),
        "status_counts": status_counts,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
