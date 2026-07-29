from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from enterprise_qa_agent.src.core.text_utils import compact_text, fix_mojibake_deep, maybe_fix_mojibake


def normalize_answer(answer: str, answer_format: str) -> str:
    letters = [char for char in answer.upper() if char in "ABCD"]
    if answer_format == "multi":
        return "".join(sorted(set(letters), key="ABCD".index))
    if answer_format in {"mcq", "tf"}:
        return letters[0] if letters else ""
    return "".join(sorted(set(letters), key="ABCD".index))


def upsert_answer_csv(state: dict, answer_csv: str) -> None:
    path = Path(answer_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    answer = normalize_answer(state.get("final_answer", ""), state["question"].get("answer_format", ""))
    usage = state.get("token_usage", {})
    row = {
        "qid": state["qid"],
        "answer": answer,
        "prompt_tokens": str(usage.get("prompt_tokens", 0)),
        "completion_tokens": str(usage.get("completion_tokens", 0)),
        "total_tokens": str(usage.get("total_tokens", 0)),
    }

    rows: list[dict[str, str]] = []
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    rows = [existing for existing in rows if existing.get("qid") not in {state["qid"], "summary"}]
    rows.append(row)

    total_prompt = sum(int(item.get("prompt_tokens") or 0) for item in rows)
    total_completion = sum(int(item.get("completion_tokens") or 0) for item in rows)
    summary = {
        "qid": "summary",
        "answer": "",
        "prompt_tokens": str(total_prompt),
        "completion_tokens": str(total_completion),
        "total_tokens": str(total_prompt + total_completion),
    }
    rows = [summary] + sorted(rows, key=lambda item: item["qid"])

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"answer_csv={path.resolve()}")
    print(f"saved_answer={state['qid']} -> {answer}")


def _append_evidence_lines(lines: list[str], evidence: list[dict], max_text_chars: int = 400, indent: str = "  ") -> None:
    for ev in evidence:
        heading = " > ".join(ev.get("heading_path") or [])
        text = re.sub(r"\s+", " ", ev.get("text", "")).strip()
        lines.append(f"{indent}- `{ev['doc_id']}` `{ev['page_id']}` score={ev['score']:.3f} source={ev['source']}")
        lines.append(f"{indent}  - heading: {heading}")
        if ev.get("selected_sections"):
            lines.append(f"{indent}  - selected_sections: {json.dumps(ev.get('selected_sections'), ensure_ascii=False)}")
        lines.append(f"{indent}  - text: {text[:max_text_chars]}")


def write_outputs(state: dict, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    state = fix_mojibake_deep(state)
    output_json = output_path / f"langgraph_{state['qid']}_state.json"
    output_md = output_path / f"langgraph_{state['qid']}.md"
    output_json.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# LangGraph Agent Debug: {state['qid']}", ""]
    lines.append(f"status: `{state.get('status')}`")
    lines.append(f"final_answer: `{state.get('final_answer', '')}`")
    lines.append(f"token_usage: `{json.dumps(state.get('token_usage', {}), ensure_ascii=False)}`")
    lines.append("")
    lines.append(f"question: {maybe_fix_mojibake(state['question']['question'])}")
    lines.append("")
    lines.append("## Trace")
    for item in state.get("trace", []):
        lines.append(f"- {item}")
    lines.append("")
    if state.get("task_reasoning_debug"):
        debug = state["task_reasoning_debug"]
        lines.append("## Task Reasoning Debug")
        lines.append("")
        lines.append("plan:")
        lines.append("```json")
        lines.append(json.dumps(debug.get("plan", {}), ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
        for task_log in debug.get("tasks", []):
            lines.append(f"### Task {task_log.get('task_index')}")
            lines.append("")
            lines.append("task:")
            lines.append("```json")
            lines.append(json.dumps(task_log.get("task", {}), ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")
            if task_log.get("section_plans"):
                lines.append("section_plans:")
                lines.append("```json")
                lines.append(json.dumps(task_log.get("section_plans"), ensure_ascii=False, indent=2))
                lines.append("```")
                lines.append("")
            for round_log in task_log.get("rounds", []):
                lines.append(f"round {round_log.get('round')}:")
                lines.append("```json")
                lines.append(json.dumps(
                    {
                        "audit": round_log.get("audit"),
                        "filtered_filled_slots": round_log.get("filtered_filled_slots"),
                        "dropped_filled_slots": round_log.get("dropped_filled_slots"),
                        "memory_slots": round_log.get("memory_slots"),
                        "evidence_notes": round_log.get("evidence_notes"),
                        "evidence_notes_after_followup": round_log.get("evidence_notes_after_followup"),
                        "audit_evidence_ids": [
                            item.get("evidence_id") for item in round_log.get("audit_evidence", [])
                        ],
                        "route_steps": round_log.get("route_steps"),
                        "missing_slots_used": round_log.get("missing_slots_used"),
                        "tried_candidates": round_log.get("tried_candidates"),
                        "followup_route_steps": round_log.get("followup_route_steps"),
                        "anchor_expanded_ids": [
                            item.get("evidence_id") for item in round_log.get("anchor_expanded", [])
                        ],
                        "followup_evidence_ids": [
                            item.get("evidence_id") for item in round_log.get("followup_evidence", [])
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ))
                lines.append("```")
                for section_name in (
                    "audit",
                    "filtered_filled_slots",
                    "dropped_filled_slots",
                    "memory_slots",
                    "evidence_notes",
                    "evidence_notes_after_followup",
                    "audit_evidence",
                    "route_steps",
                    "missing_slots_used",
                    "tried_candidates",
                    "followup_route_steps",
                ):
                    value = round_log.get(section_name)
                    if value:
                        lines.append(f"- {section_name}:")
                        lines.append("```json")
                        lines.append(json.dumps(value, ensure_ascii=False, indent=2))
                        lines.append("```")
                if round_log.get("evidence"):
                    lines.append("- evidence:")
                    _append_evidence_lines(lines, round_log.get("evidence", []), max_text_chars=400)
                if round_log.get("followup_evidence"):
                    lines.append("- followup_evidence:")
                    _append_evidence_lines(lines, round_log.get("followup_evidence", []), max_text_chars=400)
                if round_log.get("anchor_expanded"):
                    lines.append("- anchor_expanded:")
                    _append_evidence_lines(lines, round_log.get("anchor_expanded", []), max_text_chars=400)
                lines.append("")
            lines.append("result:")
            lines.append("```json")
            lines.append(json.dumps(task_log.get("result", {}), ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")
        lines.append("final judgment:")
        lines.append("```json")
        lines.append(json.dumps(state.get("reasoning_judgment", {}), ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    for option, option_state in state["option_states"].items():
        lines.append(f"## Option {option}")
        lines.append("")
        lines.append(f"claim: {option_state['claim']}")
        lines.append(f"target_doc_ids: {option_state.get('target_doc_ids')}")
        lines.append(f"status: `{option_state.get('status')}`")
        lines.append("")
        for key in ("router", "initial_rerank", "followup_rerank", "doc_reroute", "evidence_facts", "audit", "inner_focus", "toc_repair", "judgment"):
            lines.append(f"{key}:")
            lines.append("```json")
            lines.append(json.dumps(option_state.get(key, {}), ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")
        lines.append("evidence:")
        for ev in option_state.get("evidence", []):
            heading = " > ".join(ev.get("heading_path") or [])
            text = re.sub(r"\s+", " ", ev.get("text", "")).strip()
            lines.append(f"- `{ev['doc_id']}` `{ev['page_id']}` score={ev['score']:.3f} source={ev['source']}")
            lines.append(f"  - heading: {heading}")
            lines.append(f"  - text: {text[:1200]}")
        lines.append("")
    output_md.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def enterprise_answer_from_state(state: dict) -> str:
    question = state.get("question", {})
    task_type = str(question.get("enterprise_task_type") or "")
    raw = str(state.get("final_answer") or "").strip()
    if task_type == "claim_verification":
        if raw in {"T", "A", "TRUE", "姝ｇ‘"}:
            return "supported"
        if raw in {"F", "B", "FALSE", "閿欒"}:
            return "contradicted_or_insufficient"
    if task_type == "rule_applicability":
        if raw in {"T", "A", "TRUE", "姝ｇ‘"}:
            return "applicable"
        if raw in {"F", "B", "FALSE", "閿欒"}:
            return "not_applicable_or_insufficient"
    return raw


def evidence_summary_from_state(state: dict, max_items: int = 8) -> list[dict]:
    evidence: list[dict] = []
    for option_state in (state.get("option_states") or {}).values():
        for item in option_state.get("evidence") or []:
            evidence.append(item)
    if not evidence:
        for item in state.get("reasoning_memory_slots") or []:
            evidence.append({"evidence_id": "", "doc_id": "", "text": str(item)})
    seen: set[str] = set()
    rows: list[dict] = []
    for item in evidence:
        evidence_id = str(item.get("evidence_id") or item.get("page_id") or "")
        key = evidence_id or str(item)[:120]
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "doc_id": item.get("doc_id", ""),
                "evidence_id": evidence_id,
                "heading": " > ".join(item.get("heading_path") or []),
                "quote": compact_text(str(item.get("text") or ""))[:600],
                "source": item.get("source", ""),
            }
        )
        if len(rows) >= max_items:
            break
    return rows


def enterprise_reason_from_state(state: dict) -> str:
    judgments = []
    for option_state in (state.get("option_states") or {}).values():
        judgment = option_state.get("judgment") or {}
        if judgment.get("reason"):
            judgments.append(str(judgment.get("reason")))
    if judgments:
        return "\n".join(judgments[:2])
    judgment = state.get("reasoning_judgment") or {}
    if judgment.get("reason"):
        return str(judgment.get("reason"))
    return ""


def write_enterprise_result(state: dict, result_jsonl: str) -> dict:
    state = fix_mojibake_deep(state)
    atomic = state.get("question", {}).get("enterprise_atomic") or {}
    result = {
        "atomic_id": state.get("qid"),
        "source_qid": atomic.get("source_qid"),
        "source_split": atomic.get("source_split"),
        "domain": atomic.get("domain"),
        "task_type": atomic.get("task_type"),
        "question": atomic.get("question") or state.get("question", {}).get("question"),
        "status": state.get("status"),
        "answer": enterprise_answer_from_state(state),
        "raw_final_answer": state.get("final_answer", ""),
        "evidence": evidence_summary_from_state(state),
        "reason": enterprise_reason_from_state(state),
        "token_usage": state.get("token_usage", {}),
    }
    result = fix_mojibake_deep(result)
    path = Path(result_jsonl)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"enterprise_result_jsonl={path.resolve()}")
    return result

