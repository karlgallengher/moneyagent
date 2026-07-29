from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Callable

from enterprise_qa_agent.src.core.retrieval_core import tokenize
from enterprise_qa_agent.src.retrieval.section_retrieval import normalize_heading_key
from enterprise_qa_agent.src.core.text_utils import compact_text, maybe_fix_mojibake


NormalizeKeyTerms = Callable[[list | None, str, int], list[str]]
ClaimGetter = Callable[[dict], str]
SlotQueryGetter = Callable[[dict], str]

_PAGES_BY_DOC: dict[str, list[dict]] = {}
_TREE_DOC_BY_ID: dict[str, dict] = {}
_DOC_TREE_TEXT_BY_ID: dict[str, str] = {}
_NORMALIZE_KEY_TERMS: NormalizeKeyTerms | None = None
_OPTION_CLAIM: ClaimGetter | None = None
_OPTION_SLOT_QUERY_TEXT: SlotQueryGetter | None = None
_DOC_REROUTE_TOP_K = 2
_DOC_REROUTE_MIN_SCORE = 7.0
_DOC_REROUTE_MARGIN = 0.6


def configure_doc_routing(
    *,
    pages_by_doc: dict[str, list[dict]],
    tree_doc_by_id: dict[str, dict],
    doc_tree_text_by_id: dict[str, str],
    normalize_key_terms: NormalizeKeyTerms,
    option_claim: ClaimGetter,
    option_slot_query_text: SlotQueryGetter,
    doc_reroute_top_k: int,
    doc_reroute_min_score: float,
    doc_reroute_margin: float,
) -> None:
    global _PAGES_BY_DOC, _TREE_DOC_BY_ID, _DOC_TREE_TEXT_BY_ID, _NORMALIZE_KEY_TERMS
    global _OPTION_CLAIM, _OPTION_SLOT_QUERY_TEXT
    global _DOC_REROUTE_TOP_K, _DOC_REROUTE_MIN_SCORE, _DOC_REROUTE_MARGIN
    _PAGES_BY_DOC = pages_by_doc
    _TREE_DOC_BY_ID = tree_doc_by_id
    _DOC_TREE_TEXT_BY_ID = doc_tree_text_by_id
    _NORMALIZE_KEY_TERMS = normalize_key_terms
    _OPTION_CLAIM = option_claim
    _OPTION_SLOT_QUERY_TEXT = option_slot_query_text
    _DOC_REROUTE_TOP_K = doc_reroute_top_k
    _DOC_REROUTE_MIN_SCORE = doc_reroute_min_score
    _DOC_REROUTE_MARGIN = doc_reroute_margin


def doc_aliases(doc_ids: list[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    names = ["第一份文档", "第二份文档", "第三份文档", "第四份文档"]
    for idx, doc_id in enumerate(doc_ids):
        aliases[doc_id] = names[idx] if idx < len(names) else f"第{idx + 1}份文档"
    return aliases


def doc_profile_text(doc_id: str, max_chars: int = 420) -> str:
    parts: list[str] = []
    tree_doc = _TREE_DOC_BY_ID.get(doc_id) or {}
    for key in ("title", "doc_title", "source_path"):
        value = tree_doc.get(key)
        if value:
            parts.append(Path(str(value)).stem if key == "source_path" else str(value))
    tree_text = _DOC_TREE_TEXT_BY_ID.get(doc_id, "")
    if tree_text:
        parts.append(tree_text[:1200])
    profile_pages = list(_PAGES_BY_DOC.get(doc_id, [])[:3])
    for page in _PAGES_BY_DOC.get(doc_id, []):
        heading = " > ".join(page.get("heading_path") or [])
        title = str(page.get("title") or "")
        if any(term in f"{title}\n{heading}" for term in ("鐩綍", "鍥捐〃鐩綍", "鍐呭鐩綍")):
            if page not in profile_pages:
                profile_pages.append(page)
        if len(profile_pages) >= 6:
            break
    for page in profile_pages:
        title = str(page.get("title") or "")
        heading = " > ".join(page.get("heading_path") or [])
        text = str(page.get("text") or "")[:520 if "鐩綍" in f"{title}{heading}" else 180]
        parts.extend([title, heading, text])
    compact = compact_text(" ".join(part for part in parts if part))
    return compact[:max_chars]


def doc_profile_lines(doc_ids: list[str]) -> str:
    lines: list[str] = []
    for doc_id in doc_ids:
        profile = doc_profile_text(doc_id, max_chars=260)
        lines.append(f"- {doc_id}: {profile}")
    return "\n".join(lines)


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


def is_doc_hint_phrase(phrase_key: str) -> bool:
    if not phrase_key or len(phrase_key) < 4:
        return False
    if re.fullmatch(r"(?:19|20)\d{2}(?:年)?(?:至|到|-)?(?:19|20)?\d{0,4}(?:年)?", phrase_key):
        return False
    if re.fullmatch(r"[\d.%-]+", phrase_key):
        return False
    return len(re.findall(r"[\u4e00-\u9fff]", phrase_key)) >= 2


def doc_hint_scores(claim: str, doc_ids: list[str]) -> list[tuple[float, str, list[str]]]:
    claim_text = compact_text(claim)
    claim_tokens = {token for token in tokenize(claim_text) if is_doc_hint_token(token)}
    if not claim_tokens:
        return []
    claim_key = normalize_heading_key(claim_text)
    scored: list[tuple[float, str, list[str]]] = []
    for doc_id in doc_ids:
        profile = doc_profile_text(doc_id, max_chars=900)
        profile_tokens = {token for token in tokenize(profile) if is_doc_hint_token(token)}
        overlap = sorted(claim_tokens & profile_tokens, key=lambda item: (-len(item), item))[:12]
        if not overlap:
            continue
        score = 0.0
        for token in overlap:
            score += 2.5 if len(token) >= 4 else 1.0
        normalized_profile = normalize_heading_key(profile)
        for phrase in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{4,}", claim_text):
            phrase_key = normalize_heading_key(phrase)
            if is_doc_hint_phrase(phrase_key) and phrase_key in normalized_profile:
                score += min(10.0, len(phrase_key) / 2)
        phrase_hits: list[str] = []
        for size in range(12, 3, -1):
            for start in range(0, max(0, len(claim_key) - size + 1)):
                phrase_key = claim_key[start : start + size]
                if is_doc_hint_phrase(phrase_key) and phrase_key in normalized_profile and phrase_key not in phrase_hits:
                    phrase_hits.append(phrase_key)
                if len(phrase_hits) >= 3:
                    break
            if len(phrase_hits) >= 3:
                break
        for phrase_key in phrase_hits:
            score += min(8.0, len(phrase_key) / 1.5)
        scored.append((score, doc_id, overlap))
    return sorted(scored, key=lambda item: item[0], reverse=True)


def build_missing_doc_reroute_query(state: dict, missing_item: dict | str, option_state: dict) -> str:
    if _OPTION_CLAIM is None or _OPTION_SLOT_QUERY_TEXT is None:
        return ""
    parts = [
        maybe_fix_mojibake(str(state.get("question", {}).get("question") or "")),
        _OPTION_CLAIM(state),
        _OPTION_SLOT_QUERY_TEXT(option_state),
    ]
    if isinstance(missing_item, dict):
        parts.extend(
            [
                str(missing_item.get("slot") or ""),
                str(missing_item.get("followup_query") or missing_item.get("query") or ""),
                " ".join(str(term) for term in missing_item.get("key_terms") or []),
            ]
        )
    else:
        parts.append(str(missing_item or ""))
    return maybe_fix_mojibake(compact_text(" ".join(part for part in parts if part)))


def doc_reroute_scores(query: str, doc_ids: list[str]) -> list[tuple[float, str, list[str]]]:
    if _NORMALIZE_KEY_TERMS is None:
        return []
    query = maybe_fix_mojibake(query)
    query_terms = [token for token in dict.fromkeys(tokenize(query)) if is_doc_hint_token(token)]
    if not query_terms:
        return []
    scored: list[tuple[float, str, list[str]]] = []
    phrase_terms = [
        compact_text(term)
        for term in _NORMALIZE_KEY_TERMS([], query, 16)
        if is_doc_hint_phrase(compact_text(term))
    ]
    for doc_id in doc_ids:
        profile = doc_profile_text(doc_id, max_chars=5000)
        profile_tokens = set(tokenize(profile))
        overlap = [token for token in query_terms if token in profile_tokens]
        phrase_hits = [term for term in phrase_terms if term and term in compact_text(profile)]
        if not overlap and not phrase_hits:
            continue
        score = 0.0
        for token in overlap:
            score += 3.0 if len(token) >= 4 else 1.0
        for term in phrase_hits:
            score += min(12.0, max(4.0, len(term) / 1.5))
        if any(term in profile for term in ("图表目录", "目录", "表", "图")) and any(
            term in query for term in ("图", "表", "出口", "环比", "同比", "日期", "金额", "比例", "数量")
        ):
            score += 2.0
        scored.append((score, doc_id, list(dict.fromkeys(overlap + phrase_hits))[:16]))
    return sorted(scored, key=lambda row: row[0], reverse=True)


def reroute_docs_for_missing_slot(state: dict, missing_item: dict | str, option_state: dict) -> list[str]:
    attempted = set(option_state.setdefault("doc_reroute_attempted_slots", []))
    query = build_missing_doc_reroute_query(state, missing_item, option_state)
    query_key = hashlib.md5(query.encode("utf-8", errors="ignore")).hexdigest()[:12]
    if query_key in attempted:
        return []
    attempted.add(query_key)
    option_state["doc_reroute_attempted_slots"] = list(attempted)[-12:]

    current_doc_ids = list(dict.fromkeys(option_state.get("target_doc_ids") or state.get("doc_ids") or []))
    all_doc_ids = sorted(_PAGES_BY_DOC.keys())
    if not query or not all_doc_ids:
        return []
    global_scores = doc_reroute_scores(query, all_doc_ids)
    if not global_scores:
        return []
    current_scores = doc_reroute_scores(query, current_doc_ids)
    current_best = current_scores[0][0] if current_scores else 0.0
    selected: list[str] = []
    for score, doc_id, _overlap in global_scores:
        if doc_id in current_doc_ids:
            continue
        if score < _DOC_REROUTE_MIN_SCORE:
            continue
        if current_best and score < current_best * _DOC_REROUTE_MARGIN:
            continue
        selected.append(doc_id)
        if len(selected) >= _DOC_REROUTE_TOP_K:
            break
    option_state.setdefault("doc_reroute", []).append(
        {
            "query": query[:300],
            "current_doc_ids": current_doc_ids,
            "current_best": round(current_best, 3),
            "top_global": [
                {"doc_id": doc_id, "score": round(score, 3), "overlap": overlap}
                for score, doc_id, overlap in global_scores[:6]
            ],
            "added_doc_ids": selected,
        }
    )
    if not selected:
        return []

    state["doc_ids"] = list(dict.fromkeys((state.get("doc_ids") or []) + selected))
    state["doc_aliases"] = doc_aliases(state["doc_ids"])
    option_state["target_doc_ids"] = list(dict.fromkeys(current_doc_ids + selected))
    return selected


def has_doc_reroute_candidate(state: dict) -> bool:
    option_state = state.get("option_states", {}).get(state.get("current_option", ""), {})
    missing = option_state.get("audit", {}).get("missing_slots") or []
    if not missing:
        return False
    if len(state.get("doc_ids") or []) >= len(_PAGES_BY_DOC):
        return False
    attempted = set(option_state.get("doc_reroute_attempted_slots") or [])
    current_doc_ids = list(dict.fromkeys(option_state.get("target_doc_ids") or state.get("doc_ids") or []))
    current_scores_cache: dict[str, float] = {}
    for item in missing:
        query = build_missing_doc_reroute_query(state, item, option_state)
        if not query:
            continue
        query_key = hashlib.md5(query.encode("utf-8", errors="ignore")).hexdigest()[:12]
        if query_key in attempted:
            continue
        global_scores = doc_reroute_scores(query, sorted(_PAGES_BY_DOC.keys()))
        if not global_scores:
            continue
        if query not in current_scores_cache:
            current_scores = doc_reroute_scores(query, current_doc_ids)
            current_scores_cache[query] = current_scores[0][0] if current_scores else 0.0
        current_best = current_scores_cache[query]
        for score, doc_id, _overlap in global_scores:
            if doc_id in current_doc_ids:
                continue
            if score < _DOC_REROUTE_MIN_SCORE:
                continue
            if current_best and score < current_best * _DOC_REROUTE_MARGIN:
                continue
            return True
    return False


def deterministic_doc_hints(state: dict, max_docs: int = 2) -> list[str]:
    if _OPTION_CLAIM is None:
        return []
    claim = _OPTION_CLAIM(state)
    scored = doc_hint_scores(claim, state.get("doc_ids") or [])
    if not scored:
        return []
    best_score = scored[0][0]
    if best_score < 8.0:
        return []
    selected: list[str] = []
    for score, doc_id, _overlap in scored[:max_docs]:
        if score >= 8.0 and score >= best_score * 0.55:
            selected.append(doc_id)
    return selected

