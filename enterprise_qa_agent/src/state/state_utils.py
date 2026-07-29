from __future__ import annotations

import re

from enterprise_qa_agent.src.retrieval.retrieval_tools import evidence_by_ids_strict
from enterprise_qa_agent.src.core.text_utils import maybe_fix_mojibake


def merge_memory_slots(existing: list[dict], filled_slots: list[dict]) -> list[dict]:
    merged = list(existing)
    index = {item.get("slot"): idx for idx, item in enumerate(merged)}
    for item in filled_slots or []:
        if not isinstance(item, dict):
            continue
        slot = str(item.get("slot") or "").strip()
        value = str(item.get("value") or "").strip()
        if not slot or not value:
            continue
        evidence_ids = item.get("evidence_ids") or []
        normalized = {
            "slot": slot,
            "value": value,
            "evidence_ids": [str(evidence_id) for evidence_id in evidence_ids if evidence_id],
        }
        if slot in index:
            merged[index[slot]] = normalized
        else:
            index[slot] = len(merged)
            merged.append(normalized)
    return merged


def slot_evidence_ids(slots: list[dict]) -> set[str]:
    ids: set[str] = set()
    for item in slots or []:
        if not isinstance(item, dict):
            continue
        for evidence_id in item.get("evidence_ids") or []:
            evidence_id = str(evidence_id or "").strip()
            if evidence_id:
                ids.add(evidence_id)
    return ids


def normalize_evidence_id_list(values: list | None) -> list[str]:
    normalized: list[str] = []
    for value in values or []:
        evidence_id = str(value or "").strip()
        if evidence_id and evidence_id not in normalized:
            normalized.append(evidence_id)
    return normalized


def remembered_evidence_ids(option_state: dict) -> set[str]:
    remembered = slot_evidence_ids(option_state.get("memory_slots") or [])
    remembered.update(slot_evidence_ids(option_state.get("evidence_facts") or []))
    audit = option_state.get("audit") or {}
    remembered.update(slot_evidence_ids(audit.get("filled_slots") or []))
    return remembered


def slot_key(text: str) -> str:
    fixed = maybe_fix_mojibake(str(text or "")).lower()
    return re.sub(r"[\s\W_]+", "", fixed)


def prioritize_evidence_for_review(option_state: dict) -> list[dict]:
    evidence = option_state.get("evidence", []) or []
    evidence_by_id = {item.get("evidence_id"): item for item in evidence}
    ordered_ids: list[str] = []

    def add_id(evidence_id: str) -> None:
        if evidence_id and evidence_id in evidence_by_id and evidence_id not in ordered_ids:
            ordered_ids.append(evidence_id)

    for evidence_id in option_state.get("new_evidence_ids") or option_state.get("recent_evidence_ids") or []:
        add_id(evidence_id)
    for evidence_id in remembered_evidence_ids(option_state):
        add_id(evidence_id)
    for slot in option_state.get("memory_slots") or []:
        for evidence_id in slot.get("evidence_ids") or []:
            add_id(evidence_id)
    audit = option_state.get("audit") or {}
    for slot in audit.get("filled_slots") or []:
        for evidence_id in slot.get("evidence_ids") or []:
            add_id(evidence_id)
    for item in evidence:
        add_id(item.get("evidence_id"))
    return [evidence_by_id[evidence_id] for evidence_id in ordered_ids]


def current_evidence_for_read(option_state: dict, max_items: int = 6) -> list[dict]:
    batch_ids = normalize_evidence_id_list(option_state.get("evidence_read_current_ids") or [])
    if not batch_ids:
        batch_ids = normalize_evidence_id_list(
            option_state.get("new_evidence_ids") or option_state.get("recent_evidence_ids") or []
        )
    evidence = option_state.get("evidence", []) or []
    selected = evidence_by_ids_strict(evidence, batch_ids)
    if not selected:
        selected = prioritize_evidence_for_review(option_state)[:max_items]
    source_priority = {
        "evidence_extract": 0,
        "evidence_local_subsearch": 1,
        "evidence_inner_focus": 1,
    }
    selected.sort(key=lambda item: (source_priority.get(str(item.get("source") or ""), 5), -float(item.get("score") or 0.0)))
    return selected[:max_items]


def select_judge_evidence(option_state: dict) -> list[dict]:
    audit = option_state.get("audit") or {}
    selected_ids: list[str] = []
    for evidence_id in remembered_evidence_ids(option_state):
        if evidence_id not in selected_ids:
            selected_ids.append(evidence_id)
    for evidence_id in option_state.get("new_evidence_ids") or option_state.get("recent_evidence_ids") or []:
        if evidence_id not in selected_ids:
            selected_ids.append(evidence_id)
    for slot in option_state.get("memory_slots") or []:
        for evidence_id in slot.get("evidence_ids") or []:
            if evidence_id not in selected_ids:
                selected_ids.append(evidence_id)
    for slot in audit.get("filled_slots") or []:
        for evidence_id in slot.get("evidence_ids") or []:
            if evidence_id not in selected_ids:
                selected_ids.append(evidence_id)
    if not selected_ids:
        return prioritize_evidence_for_review(option_state)

    evidence_by_id = {item.get("evidence_id"): item for item in option_state.get("evidence", [])}
    if (not audit.get("can_judge")) or (audit.get("missing_slots") or []):
        for evidence_id in option_state.get("new_evidence_ids") or option_state.get("recent_evidence_ids") or []:
            if evidence_id not in selected_ids:
                selected_ids.append(evidence_id)
        for item in prioritize_evidence_for_review(option_state)[:8]:
            evidence_id = item.get("evidence_id")
            if evidence_id and evidence_id not in selected_ids:
                selected_ids.append(evidence_id)
    selected = [evidence_by_id[evidence_id] for evidence_id in selected_ids if evidence_id in evidence_by_id]
    return selected or prioritize_evidence_for_review(option_state)

