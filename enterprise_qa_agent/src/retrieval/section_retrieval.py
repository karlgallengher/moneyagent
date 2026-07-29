from __future__ import annotations

import json
import re
from typing import Any, Callable

from enterprise_qa_agent.src.core.retrieval_core import tokenize
from enterprise_qa_agent.src.retrieval.retrieval_tools import merge_evidence, page_to_evidence, retrieve_one_doc
from enterprise_qa_agent.src.core.text_utils import compact_text, maybe_fix_mojibake


JsonCall = Callable[[str, str], tuple[dict, dict[str, int]]]
UsageCallback = Callable[[Any, dict[str, int]], None]
HeadingOnlyCallback = Callable[[dict], bool]

_BM25_INDEX: Any | None = None
_PAGE_BY_ID: dict[str, dict] = {}
_PAGES_BY_DOC: dict[str, list[dict]] = {}
_SECTIONS_BY_DOC: dict[str, dict[str, list[dict]]] = {}
_DOC_SECTIONS_BY_DOC: dict[str, list[dict]] = {}
_TREE_SECTIONS_BY_DOC: dict[str, list[dict]] = {}
_DOC_TREE_TEXT_BY_ID: dict[str, str] = {}
_CALL_JSON: JsonCall | None = None
_ADD_USAGE: UsageCallback | None = None
_IS_HEADING_ONLY_PAGE: HeadingOnlyCallback = lambda _page: False
_DRY_RUN = False
_SECTION_CONTEXT_PER_SECTION = 3
_SECTION_CONTEXT_MAX_TOTAL = 6


def configure_section_retrieval(
    *,
    bm25_index: Any,
    page_by_id: dict[str, dict],
    pages_by_doc: dict[str, list[dict]],
    sections_by_doc: dict[str, dict[str, list[dict]]],
    doc_sections_by_doc: dict[str, list[dict]],
    tree_sections_by_doc: dict[str, list[dict]],
    doc_tree_text_by_id: dict[str, str],
    is_heading_only_page: HeadingOnlyCallback,
    call_json: JsonCall | None = None,
    add_usage: UsageCallback | None = None,
    dry_run: bool = False,
    section_context_per_section: int = 3,
    section_context_max_total: int = 6,
) -> None:
    global _BM25_INDEX, _PAGE_BY_ID, _PAGES_BY_DOC, _SECTIONS_BY_DOC, _DOC_SECTIONS_BY_DOC
    global _TREE_SECTIONS_BY_DOC, _DOC_TREE_TEXT_BY_ID, _CALL_JSON, _ADD_USAGE, _IS_HEADING_ONLY_PAGE
    global _DRY_RUN, _SECTION_CONTEXT_PER_SECTION, _SECTION_CONTEXT_MAX_TOTAL
    _BM25_INDEX = bm25_index
    _PAGE_BY_ID = page_by_id
    _PAGES_BY_DOC = pages_by_doc
    _SECTIONS_BY_DOC = sections_by_doc
    _DOC_SECTIONS_BY_DOC = doc_sections_by_doc
    _TREE_SECTIONS_BY_DOC = tree_sections_by_doc
    _DOC_TREE_TEXT_BY_ID = doc_tree_text_by_id
    _IS_HEADING_ONLY_PAGE = is_heading_only_page
    _CALL_JSON = call_json
    _ADD_USAGE = add_usage
    _DRY_RUN = dry_run
    _SECTION_CONTEXT_PER_SECTION = section_context_per_section
    _SECTION_CONTEXT_MAX_TOTAL = section_context_max_total


def section_title(section: dict) -> str:
    return maybe_fix_mojibake(str(section.get("title") or " > ".join(section.get("heading_path") or []))).strip()


def normalize_heading_key(text: str) -> str:
    fixed = maybe_fix_mojibake(str(text or ""))
    fixed = re.sub(r"\s+", "", fixed)
    fixed = re.sub(r"[，,。；;：:、.．（）()《》<>【】\[\]\"'“”‘’]", "", fixed)
    return fixed.lower()


def section_context_evidence(
    selected_sections: list[dict],
    source: str,
    selected_section_ids: list[str],
    selected_section_names: list[str],
    max_total: int | None = None,
) -> list[dict]:
    context: list[dict] = []
    seen: set[str] = set()
    max_total = _SECTION_CONTEXT_MAX_TOTAL if max_total is None else max_total
    for section in selected_sections:
        added_for_section = 0
        page_ids = section.get("descendant_page_ids") or section.get("page_ids") or []
        for page_id in page_ids:
            if len(context) >= max_total:
                return context
            if added_for_section >= _SECTION_CONTEXT_PER_SECTION:
                break
            if page_id in seen:
                continue
            page = _PAGE_BY_ID.get(page_id)
            if not page or _IS_HEADING_ONLY_PAGE(page):
                continue
            item = page_to_evidence(0.0, page, source + "_context")
            item["selected_section_ids"] = selected_section_ids
            item["selected_sections"] = selected_section_names
            context.append(item)
            seen.add(page_id)
            added_for_section += 1
    return context


def section_pages_as_evidence(
    section: dict,
    source: str,
    selected_section_ids: list[str],
    selected_section_names: list[str],
    max_pages: int = 2,
) -> list[dict]:
    evidence: list[dict] = []
    for page_id in section.get("descendant_page_ids") or section.get("page_ids") or []:
        if len(evidence) >= max_pages:
            break
        page = _PAGE_BY_ID.get(page_id)
        if not page or _IS_HEADING_ONLY_PAGE(page):
            continue
        item = page_to_evidence(0.0, page, source)
        item["selected_section_ids"] = selected_section_ids
        item["selected_sections"] = selected_section_names
        item["pointer_expanded_from"] = section.get("section_id")
        evidence.append(item)
    return evidence


def section_pages_as_evidence_ranked(
    section: dict,
    query: str,
    source: str,
    selected_section_ids: list[str],
    selected_section_names: list[str],
    max_pages: int = 2,
) -> list[dict]:
    page_ids = section.get("descendant_page_ids") or section.get("page_ids") or []
    pages = [_PAGE_BY_ID.get(page_id) for page_id in page_ids]
    pages = [page for page in pages if page and not _IS_HEADING_ONLY_PAGE(page)]
    if not pages:
        return []
    query_tokens = set(tokenize(query or section_title(section)))
    scored: list[tuple[float, dict]] = []
    for page in pages:
        heading_text = " > ".join(page.get("heading_path") or [])
        if any(term in compact_text(heading_text + " " + page.get("text", "")[:80]) for term in ("内容目录", "图表目录")):
            continue
        text = " ".join([heading_text, page.get("index_text", ""), page.get("text", "")])
        page_tokens = set(tokenize(text))
        score = float(len(query_tokens & page_tokens))
        for year in re.findall(r"(?:19|20)\d{2}", query or ""):
            if year in text:
                score += 2.0
        for number in re.findall(r"\d+(?:\.\d+)?\s*%", query or ""):
            if number.replace(" ", "") in text.replace(" ", ""):
                score += 3.0
        scored.append((score, page))
    scored.sort(key=lambda item: item[0], reverse=True)
    evidence: list[dict] = []
    for score, page in scored[:max_pages]:
        item = page_to_evidence(score, page, source)
        item["selected_section_ids"] = selected_section_ids
        item["selected_sections"] = selected_section_names
        item["pointer_expanded_from"] = section.get("section_id")
        evidence.append(item)
    return evidence


def section_title_as_evidence(
    section: dict,
    source: str,
    selected_section_ids: list[str],
    selected_section_names: list[str],
) -> dict | None:
    section_id = str(section.get("section_id") or "")
    doc_id = str(section.get("doc_id") or "")
    title = section_title(section)
    heading_path = [maybe_fix_mojibake(str(item)) for item in section.get("heading_path") or []]
    preview = compact_text(section.get("preview", ""))
    text_parts = []
    if heading_path:
        text_parts.append(" > ".join(heading_path))
    elif title:
        text_parts.append(title)
    if preview and preview not in text_parts[-1:]:
        text_parts.append(preview)
    text = "\n".join(part for part in text_parts if part).strip()
    if not section_id or not doc_id or not text:
        return None
    return {
        "evidence_id": f"{section_id}_toc_title",
        "doc_id": doc_id,
        "page_id": "",
        "heading_path": heading_path or [title],
        "text": text,
        "score": 0.0,
        "source": source,
        "selected_section_ids": selected_section_ids,
        "selected_sections": selected_section_names,
        "pointer_expanded_from": section_id,
    }


TITLE_POINTER_RE = re.compile(
    r"(?m)(?:^|\s)(?:第[一二三四五六七八九十百千万\d]+[章节条]|[一二三四五六七八九十]+[、.]|\d+(?:\.\d+)+|[（(]\d+[）)]|[①②③④⑤⑥⑦⑧⑨⑩])\s*[^。\n]{1,40}"
)


def find_sections_referred_by_text(doc_id: str, text: str, exclude_section_ids: set[str] | None = None) -> list[dict]:
    compact = compact_text(text)
    if not compact or len(compact) > 260:
        return []
    exclude_section_ids = exclude_section_ids or set()
    candidates = []
    for match in TITLE_POINTER_RE.finditer(compact):
        title = maybe_fix_mojibake(match.group(0)).strip()
        if title:
            candidates.append(title)
    if not candidates:
        candidates = [compact]
    candidate_keys = [normalize_heading_key(item) for item in candidates]
    matched: list[dict] = []
    for section in _DOC_SECTIONS_BY_DOC.get(doc_id, []):
        section_id = str(section.get("section_id") or "")
        if section_id in exclude_section_ids:
            continue
        section_keys = [
            normalize_heading_key(section_title(section)),
            normalize_heading_key(" > ".join(section.get("heading_path") or [])),
        ]
        for candidate_key in candidate_keys:
            if not candidate_key:
                continue
            if any(candidate_key == key or candidate_key in key or key in candidate_key for key in section_keys):
                if section not in matched:
                    matched.append(section)
                break
        if len(matched) >= 3:
            break
    return matched


def expand_title_pointer_evidence(evidence: list[dict], source: str) -> list[dict]:
    expanded: list[dict] = []
    for item in evidence:
        text = compact_text(item.get("text", ""))
        if len(text) > 260:
            continue
        selected_ids = {str(section_id) for section_id in item.get("selected_section_ids") or []}
        referred_sections = find_sections_referred_by_text(item.get("doc_id", ""), text, selected_ids)
        for section in referred_sections:
            section_id = str(section.get("section_id") or "")
            section_name = section_title(section)
            expanded.extend(
                section_pages_as_evidence(
                    section,
                    source + "_pointer",
                    list(dict.fromkeys(list(selected_ids) + [section_id])),
                    list(dict.fromkeys((item.get("selected_sections") or []) + [section_name])),
                )
            )
    return expanded


def section_score(query: str, heading: str, pages: list[dict]) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    heading_tokens = set(tokenize(maybe_fix_mojibake(heading)))
    sample_text = "\n".join(maybe_fix_mojibake(page.get("index_text", ""))[:300] for page in pages[:3])
    sample_tokens = set(tokenize(sample_text))
    return 3.0 * len(query_tokens & heading_tokens) + 1.0 * len(query_tokens & sample_tokens)


def select_relevant_sections(query: str, doc_id: str, top_n: int = 3) -> list[str]:
    sections = _SECTIONS_BY_DOC.get(doc_id, {})
    scored = [(section_score(query, heading, pages), heading) for heading, pages in sections.items()]
    scored = [(score, heading) for score, heading in scored if score > 0]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [heading for _score, heading in scored[:top_n]]


def doc_section_catalog(doc_id: str, max_sections: int = 80) -> list[str]:
    headings = list(_SECTIONS_BY_DOC.get(doc_id, {}).keys())
    headings.sort()
    return headings[:max_sections]


def compact_doc_tree_text(doc_id: str, max_lines: int = 120) -> str:
    text = _DOC_TREE_TEXT_BY_ID.get(doc_id, "")
    if not text:
        return ""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    return "\n".join(lines)


def doc_section_score(query: str, section: dict) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    heading_tokens = set(tokenize(maybe_fix_mojibake(" ".join(section.get("heading_path") or []))))
    return 4.0 * len(query_tokens & heading_tokens)


def select_relevant_doc_sections(query: str, doc_id: str, top_n: int = 3) -> list[dict]:
    sections = _DOC_SECTIONS_BY_DOC.get(doc_id, [])
    scored = [(doc_section_score(query, section), section) for section in sections]
    scored = [(score, section) for score, section in scored if score > 0]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [section for _score, section in scored[:top_n]]


def select_relevant_doc_sections_with_llm(
    query: str,
    slot: str,
    doc_id: str,
    top_n: int = 3,
    state: Any | None = None,
) -> list[dict]:
    tree_text = compact_doc_tree_text(doc_id)
    sections = _DOC_SECTIONS_BY_DOC.get(doc_id, [])
    if not tree_text or not sections:
        return []
    system = (
        "你是目录路由Agent。只根据任务、槽位和目录标题选择最可能包含证据的section_id。"
        "优先选择现金价值、账户价值、费用比例、给付规则、年度区间等计算依据章节，避免只选手续流程章节。"
        "只输出JSON。"
    )
    user = f"""
slot: {maybe_fix_mojibake(slot)}
query: {maybe_fix_mojibake(query)}
doc_id: {doc_id}

title tree:
{tree_text}

Return JSON:
{{
  "section_ids": ["section id from title tree"]
}}
"""
    if _DRY_RUN or _CALL_JSON is None:
        return select_relevant_doc_sections(query, doc_id, top_n)
    try:
        output, usage = _CALL_JSON(system, user)
        if state is not None and _ADD_USAGE is not None:
            _ADD_USAGE(state, usage)
    except Exception:
        return select_relevant_doc_sections(query, doc_id, top_n)
    by_id = {section.get("section_id"): section for section in sections}
    selected: list[dict] = []
    for section_id in output.get("section_ids") or []:
        section_id = str(section_id).strip()
        section = by_id.get(section_id)
        if section and section not in selected:
            selected.append(section)
        if len(selected) >= top_n:
            break
    return selected or select_relevant_doc_sections(query, doc_id, top_n)


def tree_section_catalog(doc_id: str, max_sections: int = 80) -> list[dict]:
    items: list[dict] = []
    for section in _TREE_SECTIONS_BY_DOC.get(doc_id, [])[:max_sections]:
        heading = " > ".join(section.get("heading_path") or [])
        items.append(
            {
                "section_id": section.get("section_id"),
                "level": section.get("level"),
                "heading": maybe_fix_mojibake(heading),
                "preview": maybe_fix_mojibake(section.get("preview") or "")[:160],
            }
        )
    return items


def tree_section_score(query: str, section: dict) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    heading_tokens = set(tokenize(maybe_fix_mojibake(" ".join(section.get("heading_path") or []))))
    preview_tokens = set(tokenize(maybe_fix_mojibake(section.get("preview") or "")))
    return 4.0 * len(query_tokens & heading_tokens) + 1.0 * len(query_tokens & preview_tokens)


def select_relevant_tree_sections(query: str, doc_id: str, top_n: int = 3) -> list[dict]:
    sections = _TREE_SECTIONS_BY_DOC.get(doc_id, [])
    scored = [(tree_section_score(query, section), section) for section in sections]
    scored = [(score, section) for score, section in scored if score > 0]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [section for _score, section in scored[:top_n]]


def select_relevant_tree_sections_with_llm(
    query: str,
    slot: str,
    doc_id: str,
    top_n: int = 3,
    state: Any | None = None,
) -> list[dict]:
    catalog = tree_section_catalog(doc_id)
    if not catalog:
        return []
    system = (
        "You are a table-of-contents router for retrieval. "
        "Choose section_id values that are most likely to contain evidence for the slot. "
        "Only choose from the given catalog. Do not invent ids. "
        "Prefer headings about the requested benefit, amount, cash value, surrender, ratio, age band, definition, or table. "
        "Return JSON only."
    )
    user = f"""
slot: {maybe_fix_mojibake(slot)}
query: {maybe_fix_mojibake(query)}
doc_id: {doc_id}

catalog:
{json.dumps(catalog, ensure_ascii=False, indent=2)}

Return JSON:
{{
  "section_ids": ["section id from catalog"]
}}
"""
    if _DRY_RUN or _CALL_JSON is None:
        return select_relevant_tree_sections(query, doc_id, top_n)
    try:
        output, usage = _CALL_JSON(system, user)
        if state is not None and _ADD_USAGE is not None:
            _ADD_USAGE(state, usage)
    except Exception:
        return select_relevant_tree_sections(query, doc_id, top_n)
    by_id = {section.get("section_id"): section for section in _TREE_SECTIONS_BY_DOC.get(doc_id, [])}
    selected: list[dict] = []
    for section_id in output.get("section_ids") or []:
        section_id = str(section_id).strip()
        section = by_id.get(section_id)
        if section and section not in selected:
            selected.append(section)
        if len(selected) >= top_n:
            break
    return selected or select_relevant_tree_sections(query, doc_id, top_n)


def select_relevant_sections_with_llm(
    query: str,
    slot: str,
    doc_id: str,
    top_n: int = 3,
    state: Any | None = None,
) -> list[str]:
    catalog = doc_section_catalog(doc_id)
    if not catalog:
        return []
    system = (
        "你是目录导航Agent。根据槽位、检索词和章节列表，选择最可能包含证据的章节。"
        "只能从给定sections中选择，最多选择3个。只输出JSON。"
    )
    user = f"""
槽位：{maybe_fix_mojibake(slot)}
检索词：{maybe_fix_mojibake(query)}
doc_id：{doc_id}

sections：
{json.dumps(catalog, ensure_ascii=False, indent=2)}

请输出：
{{
  "sections": ["从sections中原样复制的章节名"]
}}
"""
    if _DRY_RUN or _CALL_JSON is None:
        return select_relevant_sections(query, doc_id, top_n)
    try:
        output, usage = _CALL_JSON(system, user)
        if state is not None and _ADD_USAGE is not None:
            _ADD_USAGE(state, usage)
    except Exception:
        return select_relevant_sections(query, doc_id, top_n)
    allowed = set(catalog)
    selected: list[str] = []
    for heading in output.get("sections") or []:
        heading = str(heading).strip()
        if heading in allowed and heading not in selected:
            selected.append(heading)
        if len(selected) >= top_n:
            break
    return selected or select_relevant_sections(query, doc_id, top_n)


def retrieve_one_doc_by_sections(
    query: str,
    doc_id: str,
    top_k: int,
    source: str,
    section_top_n: int = 3,
    slot: str = "",
    use_llm_sections: bool = False,
    state: Any | None = None,
) -> list[dict]:
    if _BM25_INDEX is None:
        raise RuntimeError("Section retrieval is not configured. Call configure_section_retrieval first.")
    doc_sections = _DOC_SECTIONS_BY_DOC.get(doc_id, [])
    if doc_sections:
        selected_doc_sections = (
            select_relevant_doc_sections_with_llm(query, slot or query, doc_id, section_top_n, state=state)
            if use_llm_sections
            else select_relevant_doc_sections(query, doc_id, section_top_n)
        )
        if selected_doc_sections:
            allowed_page_ids = {
                page_id
                for section in selected_doc_sections
                for page_id in (section.get("descendant_page_ids") or section.get("page_ids") or [])
            }
            hits = [
                (score, page)
                for score, page in _BM25_INDEX.score(query, allowed_doc_ids={doc_id})
                if page.get("page_id") in allowed_page_ids
            ][:top_k]
            selected_sections = [" > ".join(section.get("heading_path") or []) for section in selected_doc_sections]
            selected_section_ids = [section.get("section_id") for section in selected_doc_sections]
            context = section_context_evidence(
                selected_doc_sections,
                source + "_doc_tree_section",
                selected_section_ids,
                selected_sections,
            )
            if hits:
                evidence = [page_to_evidence(score, page, source + "_doc_tree_section") for score, page in hits]
                for item in evidence:
                    item["selected_section_ids"] = selected_section_ids
                    item["selected_sections"] = selected_sections
                return merge_evidence(evidence, context)
            if context:
                return context

    tree_sections = _TREE_SECTIONS_BY_DOC.get(doc_id, [])
    if tree_sections:
        selected_tree_sections = (
            select_relevant_tree_sections_with_llm(query, slot or query, doc_id, section_top_n, state=state)
            if use_llm_sections
            else select_relevant_tree_sections(query, doc_id, section_top_n)
        )
        if selected_tree_sections:
            allowed_page_ids = {
                page_id
                for section in selected_tree_sections
                for page_id in (section.get("descendant_page_ids") or section.get("page_ids") or [])
            }
            hits = [
                (score, page)
                for score, page in _BM25_INDEX.score(query, allowed_doc_ids={doc_id})
                if page.get("page_id") in allowed_page_ids
            ][:top_k]
            selected_sections = [" > ".join(section.get("heading_path") or []) for section in selected_tree_sections]
            selected_section_ids = [section.get("section_id") for section in selected_tree_sections]
            context = section_context_evidence(
                selected_tree_sections,
                source + "_tree_section",
                selected_section_ids,
                selected_sections,
            )
            if hits:
                evidence = [page_to_evidence(score, page, source + "_tree_section") for score, page in hits]
                for item in evidence:
                    item["selected_section_ids"] = selected_section_ids
                    item["selected_sections"] = selected_sections
                return merge_evidence(evidence, context)
            if context:
                return context

    section_headings = (
        select_relevant_sections_with_llm(query, slot or query, doc_id, section_top_n, state=state)
        if use_llm_sections
        else select_relevant_sections(query, doc_id, section_top_n)
    )
    if not section_headings:
        return retrieve_one_doc(query, doc_id, top_k, source + "_doc")
    allowed_page_ids = {
        page.get("page_id")
        for heading in section_headings
        for page in _SECTIONS_BY_DOC.get(doc_id, {}).get(heading, [])
    }
    hits = [
        (score, page)
        for score, page in _BM25_INDEX.score(query, allowed_doc_ids={doc_id})
        if page.get("page_id") in allowed_page_ids
    ][:top_k]
    if not hits:
        return retrieve_one_doc(query, doc_id, top_k, source + "_doc_fallback")
    evidence = [page_to_evidence(score, page, source + "_section") for score, page in hits]
    for item in evidence:
        item["selected_sections"] = section_headings
    return evidence


def compact_toc_for_docs(doc_ids: list[str], max_lines_per_doc: int = 120) -> str:
    chunks: list[str] = []
    for doc_id in doc_ids:
        text = compact_doc_tree_text(doc_id, max_lines=max_lines_per_doc)
        if not text:
            headings = doc_section_catalog(doc_id, max_sections=max_lines_per_doc)
            text = "\n".join(f"- {heading}" for heading in headings)
        if text:
            chunks.append(f"## {doc_id}\n{text}")
    return "\n\n".join(chunks)


def page_matches_hint(page: dict, hint: str) -> bool:
    key = normalize_heading_key(hint)
    if not key:
        return False
    haystacks = [
        " > ".join(page.get("heading_path") or []),
        page.get("title", ""),
        page.get("index_text", ""),
        page.get("text", ""),
    ]
    for text in haystacks:
        text_key = normalize_heading_key(str(text or "")[:5000])
        if key and text_key and (key in text_key or text_key in key):
            return True
    return False


def section_matches_hint(section: dict, hint: str) -> bool:
    key = normalize_heading_key(hint)
    if not key:
        return False
    candidates = [section_title(section), " > ".join(section.get("heading_path") or []), section.get("preview", "")]
    for text in candidates:
        text_key = normalize_heading_key(str(text or "")[:3000])
        if key and text_key and (key in text_key or text_key in key):
            return True
    return False


def retrieve_by_toc_hint(doc_id: str, section_hint: str, query: str, max_items: int = 3) -> list[dict]:
    evidence: list[dict] = []
    selected_sections = [
        section for section in _DOC_SECTIONS_BY_DOC.get(doc_id, []) if section_matches_hint(section, section_hint)
    ][:2]
    for section in selected_sections:
        section_id = str(section.get("section_id") or "")
        section_name = section_title(section)
        if any(term in compact_text(section_name) for term in ("内容目录", "图表目录")):
            continue
        section_evidence = section_pages_as_evidence_ranked(
            section,
            query,
            "toc_repair",
            [section_id],
            [section_name],
            max_pages=max_items,
        )
        if not section_evidence:
            title_evidence = section_title_as_evidence(section, "toc_repair_title", [section_id], [section_name])
            if title_evidence:
                section_evidence.append(title_evidence)
        evidence.extend(section_evidence)
    if len(evidence) >= max_items:
        return evidence[:max_items]

    seen = {item.get("evidence_id") for item in evidence}
    for page in _PAGES_BY_DOC.get(doc_id, []):
        if len(evidence) >= max_items:
            break
        if page.get("page_id") in seen or not page_matches_hint(page, section_hint):
            continue
        item = page_to_evidence(0.0, page, "toc_repair_hint")
        item["selected_sections"] = [section_hint]
        evidence.append(item)
        seen.add(page.get("page_id"))
    if evidence:
        return evidence[:max_items]

    fallback_query = " ".join(part for part in [section_hint, query] if part)
    return retrieve_one_doc(fallback_query, doc_id, max_items, "toc_repair_bm25")

