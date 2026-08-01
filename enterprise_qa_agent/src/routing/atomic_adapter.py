from __future__ import annotations

import os
import re
from collections.abc import Callable

from enterprise_qa_agent.src.core.retrieval_core import tokenize
from enterprise_qa_agent.src.core.text_utils import compact_text, maybe_fix_mojibake


def select_atomic_record(records: list[dict], atomic_id: str) -> dict:
    for item in records:
        if str(item.get("atomic_id") or item.get("qid") or "") == atomic_id:
            return item
    raise StopIteration(f"Atomic question not found: {atomic_id}")


def enterprise_doc_query_text(atomic: dict) -> str:
    parts = [
        str(atomic.get("question") or ""),
        " ".join(str(item) for item in atomic.get("entities") or []),
        " ".join(str(item) for item in atomic.get("metrics") or []),
        " ".join(str(item) for item in atomic.get("constraints") or []),
    ]
    source_trace = atomic.get("source_trace") or {}
    parts.extend(
        [
            str(source_trace.get("original_question") or ""),
            str(source_trace.get("option_text") or ""),
        ]
    )
    return compact_text(" ".join(part for part in parts if part))


def is_doc_hint_token(token: str) -> bool:
    token = str(token or "").strip()
    if not token:
        return False
    if re.fullmatch(r"(?:19|20)\d{2}(?:年)?", token):
        return False
    if re.fullmatch(r"[\d.%-]+", token):
        return False
    if len(token) <= 1:
        return False
    if re.fullmatch(r"[a-z]+", token) and len(token) < 4:
        return False
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z]", token))


def enterprise_doc_route_score(
    doc_id: str,
    query: str,
    atomic: dict,
    doc_profile_fn: Callable[[str], str],
    tree_doc_by_id: dict[str, dict],
) -> tuple[float, list[str]]:
    profile = doc_profile_fn(doc_id)
    profile_tokens = set(tokenize(profile))
    query_tokens = [
        token
        for token in dict.fromkeys(tokenize(query))
        if is_doc_hint_token(token)
    ]
    overlap = [token for token in query_tokens if token in profile_tokens]
    score = float(len(overlap))
    for entity in atomic.get("entities") or []:
        entity = compact_text(str(entity))
        if entity and entity in profile:
            score += 8.0
            overlap.append(entity)
    for metric in atomic.get("metrics") or []:
        metric = compact_text(str(metric))
        if metric and metric in profile:
            score += 3.0
            overlap.append(metric)
    tree_doc = tree_doc_by_id.get(doc_id) or {}
    title_surface = compact_text(" ".join(str(tree_doc.get(key) or "") for key in ("title", "doc_title", "source_path")))
    for entity in atomic.get("entities") or []:
        entity = compact_text(str(entity))
        if entity and entity in title_surface:
            score += 12.0
    if any(term in title_surface for term in query_tokens[:16]):
        score += 2.0
    return score, list(dict.fromkeys(overlap))[:20]


def enterprise_prefilter_doc_ids(
    atomic: dict,
    all_doc_ids: list[str],
    doc_profile_fn: Callable[[str], str],
    tree_doc_by_id: dict[str, dict],
    max_docs: int | None = None,
) -> list[str]:
    source_trace = atomic.get("source_trace") or {}
    explicit = [doc_id for doc_id in source_trace.get("source_doc_ids") or [] if doc_id in all_doc_ids]
    if explicit:
        return explicit
    if os.environ.get("DISABLE_ENTERPRISE_DOC_ROUTER", "0").lower() in {"1", "true", "yes"}:
        return all_doc_ids
    max_docs = max_docs or int(os.environ.get("ENTERPRISE_DOC_TOP_K", "6"))
    query = enterprise_doc_query_text(atomic)
    scored = []
    for doc_id in all_doc_ids:
        score, overlap = enterprise_doc_route_score(doc_id, query, atomic, doc_profile_fn, tree_doc_by_id)
        if score > 0:
            scored.append((score, doc_id, overlap))
    scored.sort(key=lambda row: row[0], reverse=True)
    selected = [doc_id for _score, doc_id, _overlap in scored[:max_docs]]
    return selected or all_doc_ids


def is_atomic_verification_task(atomic: dict, question_text: str) -> bool:
    task_type = str(atomic.get("task_type") or "").strip()
    if task_type in {"claim_verification", "rule_applicability"}:
        return True
    text = maybe_fix_mojibake(question_text)
    verification_markers = (
        "请核验",
        "核验",
        "是否成立",
        "是否正确",
        "是否准确",
        "是否支持",
        "是否适用",
        "是否符合",
        "陈述是否",
        "说法是否",
    )
    return any(marker in text for marker in verification_markers)


def atomic_to_legacy_question(
    atomic: dict,
    target_qid: str,
    all_doc_ids: list[str],
    doc_profile_fn: Callable[[str], str],
    tree_doc_by_id: dict[str, dict],
) -> dict:
    task_type = str(atomic.get("task_type") or "claim_verification")
    doc_ids = enterprise_prefilter_doc_ids(atomic, all_doc_ids, doc_profile_fn, tree_doc_by_id)
    question_text = maybe_fix_mojibake(str(atomic.get("question") or ""))

    base = {
        "qid": str(atomic.get("atomic_id") or atomic.get("source_qid") or target_qid),
        "question": question_text,
        "domain": atomic.get("domain"),
        "split": atomic.get("source_split"),
        "doc_ids": doc_ids,
        "enterprise_atomic": atomic,
        "enterprise_task_type": task_type,
        "enterprise_domain": atomic.get("domain"),
        "enterprise_prefiltered_doc_ids": doc_ids,
    }
    if is_atomic_verification_task(atomic, question_text):
        base.update(
            {
                "type": "判断题",
                "answer_format": "tf",
                "options": {"A": "支持/适用/成立", "B": "不支持/不适用/不成立或证据不足"},
            }
        )
    else:
        base.update(
            {
                "type": "计算题",
                "answer_format": "free",
                "options": {},
            }
        )
    return base


def chat_to_legacy_question(
    query: str,
    domain: str,
    target_qid: str,
    all_doc_ids: list[str],
    doc_profile_fn: Callable[[str], str],
    tree_doc_by_id: dict[str, dict],
) -> dict:
    query_text = maybe_fix_mojibake(str(query or "")).strip()
    tokens = [token for token in dict.fromkeys(tokenize(query_text)) if is_doc_hint_token(token)]
    atomic = {
        "atomic_id": target_qid,
        "question": query_text,
        "domain": domain,
        "task_type": "chat_qa",
        "entities": tokens[:12],
        "metrics": [],
        "constraints": [],
        "source_trace": {"original_question": query_text},
    }
    doc_ids = enterprise_prefilter_doc_ids(atomic, all_doc_ids, doc_profile_fn, tree_doc_by_id)
    return {
        "qid": target_qid,
        "question": query_text,
        "domain": domain,
        "split": "chat",
        "type": "计算题",
        "answer_format": "free",
        "options": {},
        "doc_ids": doc_ids,
        "enterprise_atomic": atomic,
        "enterprise_task_type": "chat_qa",
        "enterprise_domain": domain,
        "enterprise_prefiltered_doc_ids": doc_ids,
        "enterprise_chat": True,
    }

