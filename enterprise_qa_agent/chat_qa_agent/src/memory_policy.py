from __future__ import annotations

import re

from .config import (
    COLD_EVIDENCE_TURN_WINDOW,
    HOT_EVIDENCE_TURN_WINDOW,
    MAX_EVIDENCE_MEMORY,
    MAX_EVIDENCE_SUMMARY_CHARS,
)
from .utils import safe_text


MEMORY_TERM_STOPWORDS = {
    "什么", "哪些", "哪个", "怎么", "如何", "是否", "有没有", "是不是", "区别", "不同", "相同", "比较",
    "判断", "说明", "依据", "它们", "他们", "这些", "那些", "这个", "那个", "上文", "当前", "问题",
    "研报", "报告", "文档", "条款", "规定", "合同", "主体", "对象", "相关", "信息", "情况",
    "the", "and", "for", "with", "this", "that", "what", "which", "how", "why",
}


def evidence_note_is_negative(text: str) -> bool:
    fixed = safe_text(text)
    negative_markers = (
        "未涉及", "无关", "不能支持", "无法支持", "证据不足", "未提供", "不属于", "不能用于", "不匹配",
        "未找到", "无法确认", "无法判断", "需重新检索", "需要重新检索", "待检索确认", "证据缺失",
        "irrelevant", "not relevant", "unsupported",
    )
    return any(marker in fixed for marker in negative_markers)


def infer_evidence_subject(task_result: dict, note: dict) -> str:
    note_text = safe_text(note.get("note"), 420)
    match = re.search(r"主体=([^；。\n]{1,80})", note_text)
    if match:
        return safe_text(match.group(1), 80)
    return safe_text(task_result.get("task") or task_result.get("slot"), 100)


def evidence_memory_matches_subject(row: dict, subjects: list[str]) -> bool:
    if not subjects:
        return True
    row_subject = safe_text(row.get("subject"), 300)
    if row_subject:
        return any(subject and (subject in row_subject or row_subject in subject) for subject in subjects)
    haystack = " ".join(
        safe_text(row.get(key), 300)
        for key in ("subject", "slot", "evidence_summary", "doc_id", "evidence_id")
    )
    return any(subject and subject in haystack for subject in subjects)


def evidence_memory_terms(text: str) -> set[str]:
    fixed = safe_text(text, 1200).lower()
    fixed = re.sub(r"\bpack\d+_text\d+(?:_[a-z]{1,4}_\d+)?\b", " ", fixed)
    fixed = re.sub(r"\b[a-z]+_[ab]_\d{3}(?:_[a-d])?\b", " ", fixed)
    terms: set[str] = set()
    for token in re.findall(r"[a-z0-9]{2,}", fixed):
        if token not in MEMORY_TERM_STOPWORDS and not token.startswith(("pack", "text")):
            terms.add(token)
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", fixed):
        if chunk not in MEMORY_TERM_STOPWORDS and len(chunk) <= 8:
            terms.add(chunk)
        max_n = min(6, len(chunk))
        for size in range(2, max_n + 1):
            for start in range(0, len(chunk) - size + 1):
                term = chunk[start : start + size]
                if term not in MEMORY_TERM_STOPWORDS:
                    terms.add(term)
    return terms


def evidence_memory_matches_current_slot(row: dict, route: dict) -> bool:
    query_text = " ".join(
        safe_text(part, 600)
        for part in (route.get("standalone_query"), route.get("current_query"))
        if part
    )
    query_terms = evidence_memory_terms(query_text)
    if not query_terms:
        return True
    evidence_text = " ".join(safe_text(row.get(key), 500) for key in ("slot", "evidence_summary"))
    return bool(query_terms & evidence_memory_terms(evidence_text))


def evidence_memory_turn_age(row: dict, current_turn_count: int) -> int:
    try:
        source_turn = int(row.get("source_turn") or 0)
    except (TypeError, ValueError):
        source_turn = 0
    return max(0, current_turn_count - source_turn)


def classify_evidence_memory_temperature(row: dict, route: dict, state: dict) -> str:
    status = safe_text(row.get("evidence_status"), 20)
    if status not in {"useful", "partial"}:
        return "cold"
    subjects = [safe_text(item, 100) for item in route.get("memory_subjects") or [] if safe_text(item)]
    subject_match = evidence_memory_matches_subject(row, subjects)
    slot_match = evidence_memory_matches_current_slot(row, route)
    age = evidence_memory_turn_age(row, int(state.get("turn_count") or 0))
    try:
        use_count = int(row.get("use_count") or 0)
    except (TypeError, ValueError):
        use_count = 0
    if subject_match and slot_match and (age <= HOT_EVIDENCE_TURN_WINDOW or use_count >= 2):
        return "hot"
    if subject_match and age <= COLD_EVIDENCE_TURN_WINDOW:
        return "warm"
    return "cold"


def select_evidence_memory_for_route(state: dict, route: dict) -> list[dict]:
    query_type = str(route.get("query_type") or "standalone")
    retrieval_scope = str(route.get("retrieval_scope") or "fresh_search")
    if query_type not in {"followup_same_scope", "followup_expand_scope"}:
        return []
    if retrieval_scope not in {"active_docs_only", "active_docs_plus_new_search"}:
        return []

    active_doc_ids = {str(item) for item in state.get("active_doc_ids") or [] if item}
    selected: list[dict] = []
    seen: set[str] = set()
    for row in reversed(state.get("evidence_memory") or []):
        evidence_id = safe_text(row.get("evidence_id"), 120)
        status = safe_text(row.get("evidence_status"), 20)
        if not evidence_id or status not in {"useful", "partial"}:
            continue
        if active_doc_ids and row.get("doc_id") and str(row.get("doc_id")) not in active_doc_ids:
            continue
        if classify_evidence_memory_temperature(row, route, state) != "hot":
            continue
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        selected.append(row)
        if len(selected) >= MAX_EVIDENCE_MEMORY:
            break
    return selected


def format_evidence_memory_for_prompt(rows: list[dict]) -> str:
    if not rows:
        return ""
    lines = ["可复用证据记忆（仅限当前槽位相关的压缩摘要；不要把不同槽位的旧证据当作本轮依据）："]
    for row in rows[:MAX_EVIDENCE_MEMORY]:
        lines.append(
            "- "
            f"[{safe_text(row.get('evidence_status'), 20)}] "
            f"主体={safe_text(row.get('subject'), 80)} "
            f"槽位={safe_text(row.get('slot'), 90)}："
            f"{safe_text(row.get('evidence_summary'), MAX_EVIDENCE_SUMMARY_CHARS)}"
        )
    return "\n".join(lines)
