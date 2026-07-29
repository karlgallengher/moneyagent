from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from enterprise_qa_agent.src.io.formatters import compact_html_text
from enterprise_qa_agent.src.core.retrieval_core import tokenize
from enterprise_qa_agent.src.core.text_utils import compact_text, maybe_fix_mojibake


EvidenceList = list[dict[str, Any]]
TextFixer = Callable[[str], str]

_BM25_INDEX: Any | None = None
_FIX_TEXT: TextFixer = lambda value: value


def configure_basic_retrieval(*, bm25_index: Any, fix_text: TextFixer | None = None) -> None:
    global _BM25_INDEX, _FIX_TEXT
    _BM25_INDEX = bm25_index
    if fix_text is not None:
        _FIX_TEXT = fix_text


def page_to_evidence(score: float, page: dict, source: str) -> dict:
    return {
        "evidence_id": page["page_id"],
        "doc_id": page["doc_id"],
        "page_id": page["page_id"],
        "heading_path": [_FIX_TEXT(str(item)) for item in page.get("heading_path") or []],
        "text": _FIX_TEXT(page.get("text", "")),
        "score": round(score, 6),
        "source": source,
    }


def retrieve_one_doc(query: str, doc_id: str, top_k: int, source: str) -> EvidenceList:
    if _BM25_INDEX is None:
        raise RuntimeError("Basic retrieval is not configured. Call configure_basic_retrieval first.")
    hits = _BM25_INDEX.score(query, allowed_doc_ids={doc_id})[:top_k]
    return [page_to_evidence(score, page, source) for score, page in hits]


def retrieve_multi_doc(query: str, doc_ids: list[str], top_k_per_doc: int, source: str) -> EvidenceList:
    evidence: EvidenceList = []
    for doc_id in doc_ids:
        evidence.extend(retrieve_one_doc(query, doc_id, top_k_per_doc, source))
    return evidence


def merge_evidence(
    existing: EvidenceList,
    incoming: EvidenceList,
    excluded_ids: set[str] | None = None,
) -> EvidenceList:
    excluded_ids = excluded_ids or set()
    seen = {item["evidence_id"] for item in existing}
    merged = list(existing)
    for item in incoming:
        evidence_id = item["evidence_id"]
        if evidence_id in excluded_ids:
            continue
        if evidence_id in seen:
            continue
        merged.append(item)
        seen.add(evidence_id)
    return merged


def evidence_by_ids_strict(evidence: EvidenceList, evidence_ids: list[str]) -> EvidenceList:
    if not evidence_ids:
        return []
    wanted = set(evidence_ids)
    return [item for item in evidence if item.get("evidence_id") in wanted]


def _query_years(query: str) -> set[str]:
    return set(re.findall(r"(?:19|20)\d{2}", maybe_fix_mojibake(query or "")))


def _trim_snippet_around_query(snippet: str, query: str, max_chars: int) -> str:
    snippet = snippet.strip()
    if len(snippet) <= max_chars:
        return snippet
    query_tokens = [token for token in tokenize(query) if len(token) >= 2]
    anchors = list(_query_years(query)) + query_tokens
    best_pos = -1
    for anchor in anchors:
        pos = snippet.find(anchor)
        if pos >= 0 and (best_pos < 0 or pos < best_pos):
            best_pos = pos
    if best_pos < 0:
        return snippet[:max_chars]
    start = max(0, best_pos - max_chars // 3)
    end = min(len(snippet), start + max_chars)
    start = max(0, end - max_chars)
    return snippet[start:end].strip()


def _focus_prefix_text(prefix_text: str, query: str, max_chars: int = 420) -> str:
    prefix_text = compact_text(prefix_text)
    if not prefix_text:
        return ""
    if len(prefix_text) <= max_chars:
        return prefix_text
    return _trim_snippet_around_query(prefix_text, query, max_chars=max_chars)


def _focus_markdown_table_snippet(snippet: str, query: str, max_chars: int) -> str | None:
    lines = snippet.splitlines()
    if sum(1 for line in lines if "|" in line) < 3:
        return None
    years = _query_years(query)
    query_tokens = {token for token in tokenize(query) if len(token) >= 2}
    table_start = next((idx for idx, line in enumerate(lines) if "|" in line), -1)
    if table_start < 0:
        return None
    prefix_text = _focus_prefix_text("\n".join(lines[:table_start]), query)
    prefix = [prefix_text] if prefix_text else []
    table_lines = [line for line in lines[table_start:] if "|" in line]
    if len(table_lines) < 3:
        return None
    header = table_lines[:2]
    body = table_lines[2:]
    selected: list[str] = []
    for idx, line in enumerate(body):
        line_years = set(re.findall(r"(?:19|20)\d{2}", line))
        token_hit = bool(query_tokens & set(tokenize(line)))
        year_hit = bool(years and line_years & years)
        if year_hit or (not years and token_hit):
            for near in range(max(0, idx - 1), min(len(body), idx + 2)):
                if body[near] not in selected:
                    selected.append(body[near])
    if not selected:
        return None
    focused = "\n".join(prefix + header + selected)
    if len(focused) <= max_chars:
        return focused.strip()
    kept = prefix + header
    for line in selected:
        candidate = "\n".join(kept + [line])
        if len(candidate) > max_chars:
            break
        kept.append(line)
    return "\n".join(kept).strip()


def _focus_inner_snippet(snippet: str, query: str, max_chars: int) -> str:
    table_focus = _focus_markdown_table_snippet(snippet, query, max_chars=max_chars)
    if table_focus:
        return table_focus
    return _trim_snippet_around_query(snippet, query, max_chars=max_chars)


def split_long_evidence_for_subsearch(
    text: str,
    *,
    min_chars: int,
    snippet_chars: int,
) -> list[str]:
    text = compact_html_text(maybe_fix_mojibake(text or ""))
    text = re.sub(r"[ \t]+", " ", text)
    if len(compact_text(text)) <= min_chars:
        return []

    starts = {0}
    anchor_patterns = [
        r"(?:^|\n)\s*(?:图|表)\s*[\d一二三四五六七八九十]+[：:]",
        r"(?:^|\n)\s*(?:\d+[）)]|[（(]\d+[）)]|[①②③④⑤⑥⑦⑧⑨⑩])",
        r"(?:^|\n)\s*(?:首先|其次|再次|最后|一是|二是|三是|四是)[，,：:]",
        r"(?:^|\n)\s*(?:居民收入端|居民资产端|收入端|资产端|更为关键的是)[，,：:]",
        r"(?:^|\n)\s*\|.+\|\s*(?:\n|$)",
        r"<table\b",
    ]
    for pattern in anchor_patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            starts.add(match.start())
    for match in re.finditer(r"\n{2,}", text):
        starts.add(match.end())

    ordered = sorted(start for start in starts if 0 <= start < len(text))
    raw_segments: list[str] = []
    for idx, start in enumerate(ordered):
        end = ordered[idx + 1] if idx + 1 < len(ordered) else len(text)
        segment = text[start:end].strip()
        if len(compact_text(segment)) >= 80:
            raw_segments.append(segment)

    merged: list[str] = []
    buffer = ""
    for segment in raw_segments:
        candidate = (buffer + "\n\n" + segment).strip() if buffer else segment
        if len(candidate) < 180:
            buffer = candidate
            continue
        merged.append(candidate)
        buffer = ""
    if buffer:
        merged.append(buffer)

    chunks: list[str] = []
    for segment in merged or [text]:
        if len(segment) <= snippet_chars:
            chunks.append(segment)
            continue
        table_focus = _focus_markdown_table_snippet(segment, "", max_chars=snippet_chars)
        if table_focus and len(table_focus) >= 80:
            chunks.append(table_focus)
            continue
        step = max(360, snippet_chars // 2)
        for start in range(0, len(segment), step):
            window = segment[start : start + snippet_chars]
            if len(compact_text(window)) >= 100:
                chunks.append(window.strip())
            if start + snippet_chars >= len(segment):
                break
    return chunks


def local_subsearch_score(snippet: str, query: str) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    snippet_tokens = set(tokenize(snippet))
    overlap = len(query_tokens & snippet_tokens)
    if overlap <= 0:
        return 0.0
    score = overlap * 3.0
    if re.search(r"\d{4}|第[一二三四五六七八九十\d]+", snippet):
        score += 1.0
    if re.search(r"\d+(?:\.\d+)?\s*%|百分|pct|pcts", snippet, flags=re.I):
        score += 1.0
    if re.search(r"(?:图|表)\s*[\d一二三四五六七八九十]+[：:]", snippet):
        score += 1.0
    query_numbers = set(re.findall(r"\d+(?:\.\d+)?%?|\d{4}", maybe_fix_mojibake(query or "")))
    snippet_numbers = set(re.findall(r"\d+(?:\.\d+)?%?|\d{4}", maybe_fix_mojibake(snippet or "")))
    score += len(query_numbers & snippet_numbers) * 2.0
    if any(term in snippet for term in ("同比", "增速", "增长", "下降", "收入", "利润", "费用", "现金价值", "比例")):
        score += 0.8
    return score


def local_subsearch(
    *,
    evidence: EvidenceList,
    evidence_ids: list[str],
    query: str,
    source: str,
    min_chars: int,
    max_snippets: int,
    snippet_chars: int,
) -> EvidenceList:
    if not query:
        return []
    evidence_by_id = {item.get("evidence_id"): item for item in evidence or []}
    scored: list[tuple[float, dict]] = []
    for evidence_id in evidence_ids or []:
        ev = evidence_by_id.get(evidence_id)
        if not ev:
            continue
        ev_source = str(ev.get("source") or "")
        if ev_source in {"evidence_extract", "evidence_inner_focus", "evidence_local_subsearch"}:
            continue
        text = ev.get("text", "")
        if len(compact_text(text)) <= min_chars:
            continue
        for idx, snippet in enumerate(
            split_long_evidence_for_subsearch(text, min_chars=min_chars, snippet_chars=snippet_chars)
        ):
            focused_text = _focus_inner_snippet(snippet, query, max_chars=snippet_chars)
            score = local_subsearch_score(focused_text, query)
            if score <= 0:
                continue
            focused = dict(ev)
            focused["evidence_id"] = f"{evidence_id}_sub_{idx}"
            focused["page_id"] = focused["evidence_id"]
            focused["text"] = focused_text[:snippet_chars]
            focused["score"] = round(score, 6)
            focused["source"] = "evidence_local_subsearch"
            focused["focused_from"] = evidence_id
            focused["subsearch_source"] = source
            scored.append((score, focused))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item for _score, item in scored[:max_snippets]]


@dataclass(frozen=True)
class RetrievalToolResult:
    tool: str
    query: str
    evidence: EvidenceList
    debug: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "query": self.query,
            "evidence": self.evidence,
            "debug": self.debug,
        }


class RetrievalTools:
    """Small adapter layer for retrieval operations used by the graph.

    The basic BM25 retrieval functions live in this module. More coupled
    section/TOC operations can be supplied as callables while they are being
    migrated out of the legacy flow.
    """

    def __init__(
        self,
        *,
        section_search_fn: Callable[..., EvidenceList] | None = None,
        toc_hint_fn: Callable[[str, str, str, int], EvidenceList] | None = None,
        toc_text_fn: Callable[[list[str], int], str] | None = None,
    ) -> None:
        self._section_search_fn = section_search_fn
        self._toc_hint_fn = toc_hint_fn
        self._toc_text_fn = toc_text_fn

    def global_search(
        self,
        *,
        query: str,
        doc_ids: list[str],
        top_k_per_doc: int,
        source: str,
    ) -> dict[str, Any]:
        evidence = retrieve_multi_doc(query, doc_ids, top_k_per_doc, source)
        return RetrievalToolResult(
            tool="global_search",
            query=query,
            evidence=evidence,
            debug={"doc_ids": doc_ids, "top_k_per_doc": top_k_per_doc, "source": source},
        ).as_dict()

    def section_search(
        self,
        *,
        query: str,
        doc_id: str,
        top_k: int,
        source: str,
        section_top_n: int = 3,
        slot: str = "",
        use_llm_sections: bool = False,
        state: Any | None = None,
        section_hint: str = "",
    ) -> dict[str, Any]:
        if section_hint and self._toc_hint_fn is not None:
            evidence = self._toc_hint_fn(doc_id, section_hint, query, top_k)
            return RetrievalToolResult(
                tool="section_search",
                query=query,
                evidence=evidence,
                debug={"doc_id": doc_id, "section_hint": section_hint, "top_k": top_k, "source": source},
            ).as_dict()
        if self._section_search_fn is None:
            return RetrievalToolResult(
                tool="section_search",
                query=query,
                evidence=[],
                debug={"error": "section_search_fn_not_configured"},
            ).as_dict()
        evidence = self._section_search_fn(
            query,
            doc_id,
            top_k,
            source,
            section_top_n=section_top_n,
            slot=slot,
            use_llm_sections=use_llm_sections,
            state=state,
        )
        return RetrievalToolResult(
            tool="section_search",
            query=query,
            evidence=evidence,
            debug={
                "doc_id": doc_id,
                "top_k": top_k,
                "section_top_n": section_top_n,
                "source": source,
            },
        ).as_dict()

    def toc_hint_search(
        self,
        *,
        doc_id: str,
        section_hint: str,
        query: str,
        max_items: int,
    ) -> dict[str, Any]:
        if self._toc_hint_fn is None:
            return RetrievalToolResult(
                tool="toc_hint_search",
                query=query,
                evidence=[],
                debug={"error": "toc_hint_fn_not_configured"},
            ).as_dict()
        evidence = self._toc_hint_fn(doc_id, section_hint, query, max_items)
        return RetrievalToolResult(
            tool="toc_hint_search",
            query=query,
            evidence=evidence,
            debug={"doc_id": doc_id, "section_hint": section_hint, "max_items": max_items},
        ).as_dict()

    def toc_text(self, *, doc_ids: list[str], max_lines_per_doc: int = 120) -> str:
        if self._toc_text_fn is None:
            return ""
        return self._toc_text_fn(doc_ids, max_lines_per_doc)

    def local_subsearch(
        self,
        *,
        evidence: EvidenceList,
        evidence_ids: list[str],
        query: str,
        source: str,
        min_chars: int,
        max_snippets: int,
        snippet_chars: int,
    ) -> dict[str, Any]:
        focused = local_subsearch(
            evidence=evidence,
            evidence_ids=evidence_ids,
            query=query,
            source=source,
            min_chars=min_chars,
            max_snippets=max_snippets,
            snippet_chars=snippet_chars,
        )
        return RetrievalToolResult(
            tool="local_subsearch",
            query=query,
            evidence=focused,
            debug={
                "source_evidence_ids": evidence_ids,
                "source": source,
                "min_chars": min_chars,
                "max_snippets": max_snippets,
                "snippet_chars": snippet_chars,
            },
        ).as_dict()

    def evidence_by_ids(self, *, evidence: EvidenceList, evidence_ids: list[str]) -> EvidenceList:
        return evidence_by_ids_strict(evidence, evidence_ids)

    def merge(self, existing: EvidenceList, incoming: EvidenceList) -> EvidenceList:
        return merge_evidence(existing, incoming)

