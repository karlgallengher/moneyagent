from __future__ import annotations

import re
from typing import Callable

from enterprise_qa_agent.src.core.retrieval_core import tokenize
from enterprise_qa_agent.src.retrieval.retrieval_tools import page_to_evidence
from enterprise_qa_agent.src.core.text_utils import compact_text, maybe_fix_mojibake


EvidenceList = list[dict]
PagePredicate = Callable[[dict], bool]
TextScorer = Callable[[dict, str], float]
TermsNormalizer = Callable[[list | None, str, int], list[str]]

_PAGES: list[dict] = []
_PAGES_BY_DOC: dict[str, list[dict]] = {}
_IS_HEADING_ONLY_PAGE: PagePredicate = lambda _page: False
_IS_SUMMARY_LIKE_PAGE: PagePredicate = lambda _page: False
_IS_LOW_VALUE_EVIDENCE: PagePredicate = lambda _item: False
_NORMALIZE_KEY_TERMS: TermsNormalizer | None = None
_SUMMARY_CANDIDATES_PER_DOC = 2
_VERBOSE_CONSOLE = False
_SUMMARY_PAGE_TERMS: tuple[str, ...] = ()


def configure_retrieval_aux_tools(
    *,
    pages: list[dict],
    pages_by_doc: dict[str, list[dict]],
    is_heading_only_page: PagePredicate,
    is_summary_like_page: PagePredicate,
    is_low_value_evidence: PagePredicate,
    normalize_key_terms: TermsNormalizer,
    summary_candidates_per_doc: int,
    verbose_console: bool = False,
    summary_page_terms: tuple[str, ...] = (),
) -> None:
    global _PAGES, _PAGES_BY_DOC, _IS_HEADING_ONLY_PAGE, _IS_SUMMARY_LIKE_PAGE
    global _IS_LOW_VALUE_EVIDENCE, _NORMALIZE_KEY_TERMS, _SUMMARY_CANDIDATES_PER_DOC
    global _VERBOSE_CONSOLE, _SUMMARY_PAGE_TERMS
    _PAGES = pages
    _PAGES_BY_DOC = pages_by_doc
    _IS_HEADING_ONLY_PAGE = is_heading_only_page
    _IS_SUMMARY_LIKE_PAGE = is_summary_like_page
    _IS_LOW_VALUE_EVIDENCE = is_low_value_evidence
    _NORMALIZE_KEY_TERMS = normalize_key_terms
    _SUMMARY_CANDIDATES_PER_DOC = summary_candidates_per_doc
    _VERBOSE_CONSOLE = verbose_console
    _SUMMARY_PAGE_TERMS = summary_page_terms


def evidence_surface_text(item: dict) -> str:
    return "\n".join(
        [
            " > ".join(item.get("heading_path") or []),
            str(item.get("text") or ""),
        ]
    )


def evidence_query_score(item: dict, query: str) -> float:
    surface = evidence_surface_text(item)
    query_tokens = set(tokenize(query))
    surface_tokens = set(tokenize(surface))
    score = float(item.get("score") or 0.0)
    score += len(query_tokens & surface_tokens) * 1.5
    for year in re.findall(r"(?:19|20)\d{2}", query or ""):
        if year in surface:
            score += 2.0
    for number in re.findall(r"\d+(?:\.\d+)?\s*%?", query or ""):
        if number and number.replace(" ", "") in surface.replace(" ", ""):
            score += 1.0
    if _IS_LOW_VALUE_EVIDENCE(item):
        score -= 12.0
    elif any(term in compact_text(surface[:1200]) for term in _SUMMARY_PAGE_TERMS):
        score += 2.0
    return score


def rerank_initial_evidence(evidence: EvidenceList, query: str) -> EvidenceList:
    seen: set[str] = set()
    unique: EvidenceList = []
    for item in evidence:
        evidence_id = str(item.get("evidence_id") or "")
        if not evidence_id or evidence_id in seen:
            continue
        seen.add(evidence_id)
        unique.append(item)
    scored = [(evidence_query_score(item, query), idx, item) for idx, item in enumerate(unique)]
    scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
    return [item for _score, _idx, item in scored]


def evidence_slot_score(item: dict, slot: str, query: str, key_terms: list[str] | None = None) -> float:
    surface = evidence_surface_text(item)
    compact_surface = compact_text(surface)
    normalizer = _NORMALIZE_KEY_TERMS
    terms = normalizer(key_terms or [], " ".join([slot, query]), 12) if normalizer else []
    score = evidence_query_score(item, query)
    covered = 0.0
    for term in terms:
        compact_term = compact_text(term)
        if not compact_term:
            continue
        if compact_term in compact_surface:
            score += 4.0
            covered += 1.0
        else:
            term_tokens = set(tokenize(term))
            surface_tokens = set(tokenize(surface))
            overlap = len(term_tokens & surface_tokens)
            if overlap:
                score += min(2.5, overlap * 0.8)
                covered += 0.5
    if terms:
        score += min(6.0, covered * 1.2)
        if covered == 0:
            score -= 6.0
    return score


def rerank_evidence_for_slot(
    evidence: EvidenceList,
    slot: str,
    query: str,
    key_terms: list[str] | None = None,
    max_items: int | None = None,
) -> EvidenceList:
    seen: set[str] = set()
    unique: EvidenceList = []
    for item in evidence:
        evidence_id = str(item.get("evidence_id") or "")
        if not evidence_id or evidence_id in seen:
            continue
        seen.add(evidence_id)
        unique.append(item)
    scored = [(evidence_slot_score(item, slot, query, key_terms), idx, item) for idx, item in enumerate(unique)]
    scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
    ranked = [item for _score, _idx, item in scored]
    return ranked[:max_items] if max_items is not None else ranked


def summary_candidate_evidence(doc_ids: list[str], query: str, source: str = "initial_summary") -> EvidenceList:
    candidates: EvidenceList = []
    for doc_id in doc_ids:
        scored: list[tuple[float, dict]] = []
        for page in _PAGES_BY_DOC.get(doc_id, [])[:8]:
            if not page or _IS_HEADING_ONLY_PAGE(page) or not _IS_SUMMARY_LIKE_PAGE(page):
                continue
            item = page_to_evidence(0.0, page, source)
            score = evidence_query_score(item, query)
            if score <= 0:
                continue
            scored.append((score, item))
        scored.sort(key=lambda row: row[0], reverse=True)
        candidates.extend(item for _score, item in scored[:_SUMMARY_CANDIDATES_PER_DOC])
    return candidates


def retrieve_by_key_term_coverage(
    doc_ids: list[str],
    key_terms: list[str],
    source: str,
    max_per_doc: int = 3,
) -> EvidenceList:
    normalizer = _NORMALIZE_KEY_TERMS
    terms = normalizer(key_terms or [], "", 12) if normalizer else []
    terms = [term for term in terms if len(compact_text(term)) >= 2]
    if len(terms) < 2:
        return []
    evidence: EvidenceList = []
    allowed = set(doc_ids)
    for doc_id in doc_ids:
        scored: list[tuple[float, dict]] = []
        for page in _PAGES:
            if page.get("doc_id") != doc_id or (allowed and page.get("doc_id") not in allowed):
                continue
            surface = compact_text(
                "\n".join(
                    [
                        str(page.get("title") or ""),
                        " ".join(page.get("heading_path") or []),
                        str(page.get("index_text") or page.get("text") or ""),
                    ]
                )
            )
            if not surface:
                continue
            covered = [term for term in terms if compact_text(term) in surface]
            if len(covered) < 2:
                continue
            score = len(covered) * 20.0 + min(len(surface), 2000) / 2000.0
            scored.append((score, page))
        scored.sort(key=lambda row: row[0], reverse=True)
        evidence.extend(page_to_evidence(score, page, source) for score, page in scored[:max_per_doc])
    return evidence


def is_expandable_anchor_evidence(item: dict) -> bool:
    text = maybe_fix_mojibake(str(item.get("text") or "")).strip()
    compact = re.sub(r"\s+", "", text)
    lower_text = text.lower()
    if not compact:
        return False
    if "<table" in lower_text or "</table" in lower_text or lower_text.count("<tr") >= 2:
        return False
    if len(compact) > 180:
        return False
    anchor_endings = (
        "为：",
        "为:",
        "公式：",
        "公式:",
        "如下：",
        "如下:",
        "其中：",
        "其中:",
        "下表：",
        "下表:",
        "下列：",
        "下列:",
        "如下",
    )
    if compact.endswith(anchor_endings):
        return True
    anchor_phrases = (
        "比例为",
        "数值为",
        "标准为",
        "如下",
        "见下表",
        "下表",
        "下列",
        "以下公式",
        "如下公式",
        "按照以下公式",
        "按以下公式",
        "计算并给付",
        "计算公式",
    )
    tail = compact[-40:]
    return any(phrase in tail for phrase in anchor_phrases) and not re.search(r"\d+%|<td>|</td>", tail)


def expand_forward_evidence_ids(
    evidence: EvidenceList,
    evidence_ids: list[str],
    forward: int = 4,
    source_suffix: str = "_anchor_expand",
) -> EvidenceList:
    expanded: EvidenceList = []
    requested = {str(evidence_id) for evidence_id in evidence_ids or [] if evidence_id}
    if not requested:
        return expanded
    seen = {item["evidence_id"] for item in evidence}
    for item in evidence:
        if item["evidence_id"] not in requested:
            continue
        if not is_expandable_anchor_evidence(item):
            if _VERBOSE_CONSOLE:
                print(f"[anchor expand skipped] id={item['evidence_id']} reason=not_incomplete_anchor", flush=True)
            continue
        pages = _PAGES_BY_DOC.get(item["doc_id"], [])
        page_indexes = [idx for idx, page in enumerate(pages) if page.get("page_id") == item["page_id"]]
        if not page_indexes:
            continue
        idx = page_indexes[0]
        for next_idx in range(idx + 1, min(len(pages), idx + forward + 1)):
            page = pages[next_idx]
            page_id = page.get("page_id")
            if not page_id or page_id in seen:
                continue
            expanded.append(page_to_evidence(item.get("score", 0.0), page, item.get("source", "") + source_suffix))
            seen.add(page_id)
    return expanded


def expand_auto_anchor_evidence(
    evidence: EvidenceList,
    forward: int = 2,
    source_suffix: str = "_auto_anchor_expand",
) -> EvidenceList:
    anchor_ids = [
        item.get("evidence_id")
        for item in evidence
        if item.get("evidence_id") and is_expandable_anchor_evidence(item)
    ]
    return expand_forward_evidence_ids(evidence, anchor_ids, forward=forward, source_suffix=source_suffix)

