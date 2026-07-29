"""
Archived helpers from legacy_agent.py.

These functions had no active call sites when archived. They are kept for
reference while the agent is being decomposed. Do not import from this module
in production flow unless restored deliberately.
"""

from __future__ import annotations

import re


def retrieve_multi_doc_by_sections(
    query: str,
    doc_ids: list[str],
    top_k_per_doc: int,
    source: str,
    slot: str = "",
    use_llm_sections: bool = False,
    state: AgentState | None = None,
) -> list[dict]:
    evidence: list[dict] = []
    for doc_id in doc_ids:
        evidence.extend(
            retrieve_one_doc_by_sections(
                query,
                doc_id,
                top_k_per_doc,
                source,
                slot=slot,
                use_llm_sections=use_llm_sections,
                state=state,
            )
        )
    return evidence


def expand_neighbor_evidence(evidence: list[dict], window: int = 2, source_suffix: str = "_neighbor") -> list[dict]:
    expanded: list[dict] = []
    seen = {item["evidence_id"] for item in evidence}
    for item in evidence:
        pages = PAGES_BY_DOC.get(item["doc_id"], [])
        page_indexes = [idx for idx, page in enumerate(pages) if page.get("page_id") == item["page_id"]]
        if not page_indexes:
            continue
        idx = page_indexes[0]
        for neighbor_idx in range(max(0, idx - window), min(len(pages), idx + window + 1)):
            page = pages[neighbor_idx]
            page_id = page.get("page_id")
            if not page_id or page_id in seen:
                continue
            expanded.append(page_to_evidence(item.get("score", 0.0), page, item.get("source", "") + source_suffix))
            seen.add(page_id)
    return expanded


def normalize_required_slots(task: dict) -> list[str]:
    raw_slots = task.get("required_slots") or task.get("slots") or []
    slots: list[str] = []
    for item in raw_slots:
        slot = str(item.get("slot") if isinstance(item, dict) else item).strip()
        if is_speculative_required_slot(slot):
            continue
        if slot and slot not in slots:
            slots.append(slot[:80])
    if not slots:
        slots = [str(task.get("task") or task.get("query") or "子任务结论")[:80]]
    return slots


def is_speculative_required_slot(slot: str) -> bool:
    compact = re.sub(r"\s+", "", maybe_fix_mojibake(str(slot or "")))
    patterns = (
        "是否有其他",
        "是否还有",
        "有无其他",
        "有没有其他",
        "其他退保扣费",
        "其他扣费",
        "额外扣费",
        "额外费用",
        "另有扣费",
        "另有费用",
        "是否存在其他",
    )
    return any(pattern in compact for pattern in patterns)


def filled_required_slots(required_slots: list[str], memory_slots: list[dict]) -> tuple[list[str], list[str]]:
    filled: list[str] = []
    missing: list[str] = []
    memory_keys = [
        (
            slot_key(item.get("slot", "")),
            slot_key(item.get("value", "")),
            set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9%]+", maybe_fix_mojibake(str(item.get("slot", ""))))),
        )
        for item in memory_slots
    ]
    for required in required_slots:
        req_key = slot_key(required)
        req_terms = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9%]+", maybe_fix_mojibake(str(required))))
        is_filled = False
        for mem_key, value_key, mem_terms in memory_keys:
            if not value_key:
                continue
            if req_key and (req_key in mem_key or mem_key in req_key):
                is_filled = True
                break
            overlap = req_terms & mem_terms
            if req_terms and len(overlap) >= max(1, min(2, len(req_terms))):
                is_filled = True
                break
        if is_filled:
            filled.append(required)
        else:
            missing.append(required)
    return filled, missing


def evidence_by_ids(evidence: list[dict], evidence_ids: list[str]) -> list[dict]:
    if not evidence_ids:
        return evidence
    wanted = set(evidence_ids)
    selected = [item for item in evidence if item.get("evidence_id") in wanted]
    return selected or evidence


def evidence_has_toc_like_content(option_state: dict) -> bool:
    for item in option_state.get("evidence", []) or []:
        heading = compact_text(" > ".join(item.get("heading_path") or []))
        text = compact_text(item.get("text", ""))[:1000]
        if any(term in heading + text for term in ("目录", "图表目录", "图 ", "图:", "图：", "表 ", "表:", "表：")):
            return True
    return False


def missing_slots_metric_like(missing_slots: list) -> bool:
    terms = (
        "年",
        "%",
        "同比",
        "增速",
        "增长",
        "比例",
        "占比",
        "贡献",
        "金额",
        "规模",
        "排名",
        "市场份额",
        "剪刀差",
        "收入",
        "利润",
        "表",
        "图",
    )
    for item in missing_slots or []:
        if isinstance(item, dict):
            text = " ".join(str(item.get(key) or "") for key in ("slot", "followup_query", "query"))
        else:
            text = str(item)
        text = maybe_fix_mojibake(text)
        if any(term in text for term in terms):
            return True
    return False
