from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, TypedDict

import requests
from langgraph.graph import END, StateGraph


TARGET_QID = "ins_a_006" 
# QUESTIONS_PATH = "public_dataset_upload/questions/group_a/financial_reports_questions.json"
# PAGE_INDEX_PATH = "processed/page_index_financial_reports/page_index.jsonl"

# QUESTIONS_PATH = "public_dataset_upload/questions/group_a/financial_contracts_questions.json"
# PAGE_INDEX_PATH = "processed/page_index_financial_contracts/page_index.jsonl"

QUESTIONS_PATH = "public_dataset_upload/questions/group_a/insurance_questions.json"
PAGE_INDEX_PATH = "processed/page_index_insurance/page_index.jsonl"



ANSWER_CSV = "processed/submission/insurance_answer.csv"
OUTPUT_DIR = "processed/agent_debug"

INITIAL_TOP_K = 4
FOLLOWUP_TOP_K = 3
MAX_AUDIT_ROUNDS = 2
MAX_REASONING_AUDIT_ROUNDS = 3
REQUEST_TIMEOUT_SECONDS = 180
REQUEST_RETRY_TIMES = 3
VERBOSE_CONSOLE = True
PRINT_EVIDENCE_TEXT = True
EVIDENCE_TEXT_PREVIEW_CHARS = 700
COMPACT_HTML_EVIDENCE = True
SECTION_CONTEXT_PER_SECTION = 3
SECTION_CONTEXT_MAX_TOTAL = 6
SECTION_CONTEXT_MIN_CHARS = 20
MAX_TOC_ROUTE_DEPTH = 4
MAX_TOC_CHILDREN = 24
SAVE_DEBUG_OUTPUTS = True
APPEND_ANSWER_CSV =True
DRY_RUN_WITHOUT_LLM = False
# DASHSCOPE_API_KEY_ENV = "DASHSCOPE_API_KEY"
# DASHSCOPE_API_KEY_FILE = "api"
# DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
# QWEN_MODEL = "qwen3.6-plus"
# QWEN_MODEL = "qwen3.5-plus-2026-04-20"


# 1. 鐩存帴鏄惧紡鍐欏叆鎮ㄧ殑璁鏄熻景 API Key锛堣鏇挎崲涓烘偍鐪熷疄鐨?Key锛?SPARKAI_API_KEY = "a24abac9ef09acd39638685937f0a999:NWM0ZTE1ODBlMmYyNDI4MzkxYmJlMmE3"
# 2. 璁鏄熻景 OpenAI 鍏煎妯″紡鐨?Base URL
SPARKAI_API_KEY_ENV = "SPARKAI_API_KEY"
SPARKAI_API_KEY_FILE = "spark_api"
SPARKAI_BASE_URL = "https://maas-api.cn-huabei-1.xf-yun.com/v2"
SPARKAI_CHAT_URL = f"{SPARKAI_BASE_URL.rstrip('/')}/chat/completions"
# 3. 瑕佷娇鐢ㄧ殑妯″瀷鍚嶇О锛堜互 Qwen3.6 涓轰緥锛岃鏍规嵁骞冲彴瀹為檯鏀寔鐨勫悕绉板～鍐欙級
QWEN_MODEL = "xopqwen36v35b"


class AgentState(TypedDict, total=False):
    qid: str
    question: dict
    doc_ids: list[str]
    doc_aliases: dict[str, str]
    current_option: str
    option_order: list[str]
    option_index: int
    option_states: dict[str, dict]
    final_answer: str
    status: str
    trace: list[str]
    token_usage: dict[str, int]
    task_reasoning_plan: dict
    task_reasoning_tasks: list[dict]
    task_reasoning_results: list[dict]
    task_reasoning_debug: dict
    reasoning_plan: dict
    reasoning_slots: list[dict]
    reasoning_memory_slots: list[dict]
    reasoning_judgment: dict


def maybe_fix_mojibake(text: str) -> str:
    try:
        fixed = text.encode("gb18030").decode("utf-8")
    except UnicodeError:
        return text
    common = ("\u7684", "\u7b2c", "\u516c\u53f8", "\u53d1\u884c", "\u503a\u5238", "\u4fe1\u606f", "\u62a5\u544a")
    mojibake = tuple(chr(code) for code in (0x951B, 0x7ED7, 0x93C9, 0x95B2, 0x9429, 0x7039, 0x5F42, 0x20AC))
    fixed_score = sum(fixed.count(word) for word in common) * 3 - sum(fixed.count(word) for word in mojibake)
    text_score = sum(text.count(word) for word in common) * 3 - sum(text.count(word) for word in mojibake)
    return fixed if fixed_score > text_score else text


def load_json(path: str) -> Any:
    return json.loads(maybe_fix_mojibake(Path(path).read_text(encoding="utf-8")))


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def tokenize(text: str) -> list[str]:
    text = maybe_fix_mojibake(text)
    tokens: list[str] = []
    tokens.extend(re.findall(r"[A-Za-z0-9_.%-]+", text.lower()))
    tokens.extend(re.findall(r"[\u4e00-\u9fff]{2,}", text))
    for word in re.findall(r"[\u4e00-\u9fff]{4,}", text):
        tokens.extend(word[i : i + 2] for i in range(len(word) - 1))
        tokens.extend(word[i : i + 3] for i in range(len(word) - 2))
    return tokens


class BM25:
    def __init__(self, docs: list[dict], text_key: str = "index_text", k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(doc.get(text_key, "")) for doc in docs]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        df: dict[str, int] = defaultdict(int)
        for tf in self.term_freqs:
            for term in tf:
                df[term] += 1
        n = len(docs)
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def score(self, query: str, allowed_doc_ids: set[str]) -> list[tuple[float, dict]]:
        query_terms = Counter(tokenize(query))
        scored: list[tuple[float, dict]] = []
        for idx, doc in enumerate(self.docs):
            if doc["doc_id"] not in allowed_doc_ids:
                continue
            score = 0.0
            tf = self.term_freqs[idx]
            dl = self.doc_lens[idx] or 1
            for term, qf in query_terms.items():
                if term not in tf:
                    continue
                freq = tf[term]
                idf = self.idf.get(term, 0.0)
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * (freq * (self.k1 + 1) / denom) * min(qf, 3)
            if score > 0:
                scored.append((score, doc))
        return sorted(scored, key=lambda item: item[0], reverse=True)


PAGES = load_jsonl(PAGE_INDEX_PATH)
BM25_INDEX = BM25(PAGES)
PAGE_BY_ID = {page["page_id"]: page for page in PAGES}
TREE_DOC_PATH = str(Path(PAGE_INDEX_PATH).with_name("tree_doc.jsonl"))
TREE_DOCS = load_jsonl(TREE_DOC_PATH) if Path(TREE_DOC_PATH).exists() else []
TREE_DOC_BY_ID = {doc["doc_id"]: doc for doc in TREE_DOCS}
TREE_SECTIONS_BY_DOC = {doc["doc_id"]: doc.get("sections") or [] for doc in TREE_DOCS}
DOC_INDEX_DIR = Path(PAGE_INDEX_PATH).with_name("docs")
DOC_TREE_TEXT_BY_ID: dict[str, str] = {}
DOC_SECTIONS_BY_DOC: dict[str, list[dict]] = {}
if DOC_INDEX_DIR.exists():
    for doc_dir in DOC_INDEX_DIR.iterdir():
        if not doc_dir.is_dir():
            continue
        doc_id = doc_dir.name
        tree_path = doc_dir / "tree.md"
        sections_path = doc_dir / "sections.jsonl"
        if tree_path.exists():
            DOC_TREE_TEXT_BY_ID[doc_id] = tree_path.read_text(encoding="utf-8")
        if sections_path.exists():
            DOC_SECTIONS_BY_DOC[doc_id] = load_jsonl(str(sections_path))
SECTION_BY_DOC_ID: dict[str, dict[str, dict]] = {
    doc_id: {str(section.get("section_id")): section for section in sections}
    for doc_id, sections in DOC_SECTIONS_BY_DOC.items()
}
PAGES_BY_DOC: dict[str, list[dict]] = defaultdict(list)
for page in PAGES:
    PAGES_BY_DOC[page["doc_id"]].append(page)
for pages in PAGES_BY_DOC.values():
    pages.sort(key=lambda item: (item.get("page_no", 0), item.get("page_id", "")))

SECTIONS_BY_DOC: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
for page in PAGES:
    heading_key = " > ".join(page.get("heading_path") or [])
    if heading_key:
        SECTIONS_BY_DOC[page["doc_id"]][heading_key].append(page)


def add_trace(state: AgentState, message: str) -> None:
    state.setdefault("trace", []).append(message)
    if VERBOSE_CONSOLE:
        print(f"[trace] {message}", flush=True)


def add_usage(state: AgentState, usage: dict[str, int]) -> None:
    current = state.setdefault("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    current["prompt_tokens"] += usage.get("prompt_tokens", 0)
    current["completion_tokens"] += usage.get("completion_tokens", 0)
    current["total_tokens"] += usage.get("total_tokens", 0)


def page_to_evidence(score: float, page: dict, source: str) -> dict:
    return {
        "evidence_id": page["page_id"],
        "doc_id": page["doc_id"],
        "page_id": page["page_id"],
        "heading_path": page.get("heading_path") or [],
        "text": maybe_fix_mojibake(page.get("text", "")),
        "score": round(score, 6),
        "source": source,
    }


def compact_text(value: str) -> str:
    return re.sub(r"\s+", " ", maybe_fix_mojibake(value or "")).strip()


def is_heading_only_page(page: dict) -> bool:
    text = compact_text(page.get("text", ""))
    if len(text) < SECTION_CONTEXT_MIN_CHARS:
        return True
    heading = compact_text(" > ".join(page.get("heading_path") or []))
    if heading and (text == heading or heading.endswith(text)):
        return True
    return False


def section_context_evidence(
    selected_sections: list[dict],
    source: str,
    selected_section_ids: list[str],
    selected_section_names: list[str],
    max_total: int = SECTION_CONTEXT_MAX_TOTAL,
) -> list[dict]:
    context: list[dict] = []
    seen: set[str] = set()
    for section in selected_sections:
        added_for_section = 0
        page_ids = section.get("descendant_page_ids") or section.get("page_ids") or []
        for page_id in page_ids:
            if len(context) >= max_total:
                return context
            if added_for_section >= SECTION_CONTEXT_PER_SECTION:
                break
            if page_id in seen:
                continue
            page = PAGE_BY_ID.get(page_id)
            if not page or is_heading_only_page(page):
                continue
            item = page_to_evidence(0.0, page, source + "_context")
            item["selected_section_ids"] = selected_section_ids
            item["selected_sections"] = selected_section_names
            context.append(item)
            seen.add(page_id)
            added_for_section += 1
    return context


TITLE_POINTER_RE = re.compile(
    r"(?m)(?:^|\s)(?:第[一二三四五六七八九十百千万\d]+[章节条]|[一二三四五六七八九十]+[、.]|\d+(?:\.\d+)+|[（(]\d+[）)]|[①②③④⑤⑥⑦⑧⑨⑩])\s*[^。\n]{1,40}"
)


def normalize_heading_key(text: str) -> str:
    fixed = maybe_fix_mojibake(str(text or ""))
    fixed = re.sub(r"\s+", "", fixed)
    fixed = re.sub(r"[，,。；;：:、.．（）()《》<>【】\[\]\"'“”‘’]", "", fixed)
    return fixed.lower()


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
        page = PAGE_BY_ID.get(page_id)
        if not page or is_heading_only_page(page):
            continue
        item = page_to_evidence(0.0, page, source)
        item["selected_section_ids"] = selected_section_ids
        item["selected_sections"] = selected_section_names
        item["pointer_expanded_from"] = section.get("section_id")
        evidence.append(item)
    return evidence


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
    for section in DOC_SECTIONS_BY_DOC.get(doc_id, []):
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


def retrieve_one_doc(query: str, doc_id: str, top_k: int, source: str) -> list[dict]:
    hits = BM25_INDEX.score(query, allowed_doc_ids={doc_id})[:top_k]
    return [page_to_evidence(score, page, source) for score, page in hits]


def retrieve_multi_doc(query: str, doc_ids: list[str], top_k_per_doc: int, source: str) -> list[dict]:
    evidence: list[dict] = []
    for doc_id in doc_ids:
        evidence.extend(retrieve_one_doc(query, doc_id, top_k_per_doc, source))
    return evidence


def section_score(query: str, heading: str, pages: list[dict]) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    heading_text = maybe_fix_mojibake(heading)
    heading_tokens = set(tokenize(heading_text))
    sample_text = "\n".join(maybe_fix_mojibake(page.get("index_text", ""))[:300] for page in pages[:3])
    sample_tokens = set(tokenize(sample_text))
    score = 0.0
    score += 3.0 * len(query_tokens & heading_tokens)
    score += 1.0 * len(query_tokens & sample_tokens)
    return score


def select_relevant_sections(query: str, doc_id: str, top_n: int = 3) -> list[str]:
    sections = SECTIONS_BY_DOC.get(doc_id, {})
    scored = [
        (section_score(query, heading, pages), heading)
        for heading, pages in sections.items()
    ]
    scored = [(score, heading) for score, heading in scored if score > 0]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [heading for _score, heading in scored[:top_n]]


def doc_section_catalog(doc_id: str, max_sections: int = 80) -> list[str]:
    headings = list(SECTIONS_BY_DOC.get(doc_id, {}).keys())
    headings.sort()
    return headings[:max_sections]


def compact_doc_tree_text(doc_id: str, max_lines: int = 120) -> str:
    text = DOC_TREE_TEXT_BY_ID.get(doc_id, "")
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
    heading = " ".join(section.get("heading_path") or [])
    heading_text = maybe_fix_mojibake(heading)
    heading_tokens = set(tokenize(heading_text))
    return 4.0 * len(query_tokens & heading_tokens)


def select_relevant_doc_sections(query: str, doc_id: str, top_n: int = 3) -> list[dict]:
    sections = DOC_SECTIONS_BY_DOC.get(doc_id, [])
    scored = [(doc_section_score(query, section), section) for section in sections]
    scored = [(score, section) for score, section in scored if score > 0]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [section for _score, section in scored[:top_n]]


def select_relevant_doc_sections_with_llm(
    query: str,
    slot: str,
    doc_id: str,
    top_n: int = 3,
    state: AgentState | None = None,
) -> list[dict]:
    tree_text = compact_doc_tree_text(doc_id)
    sections = DOC_SECTIONS_BY_DOC.get(doc_id, [])
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
    if DRY_RUN_WITHOUT_LLM:
        return select_relevant_doc_sections(query, doc_id, top_n)
    try:
        output, usage = call_qwen_json(system, user)
        if state is not None:
            add_usage(state, usage)
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
    sections = TREE_SECTIONS_BY_DOC.get(doc_id, [])
    items: list[dict] = []
    for section in sections[:max_sections]:
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
    heading = " ".join(section.get("heading_path") or [])
    preview = section.get("preview") or ""
    heading_tokens = set(tokenize(maybe_fix_mojibake(heading)))
    preview_tokens = set(tokenize(maybe_fix_mojibake(preview)))
    score = 0.0
    score += 4.0 * len(query_tokens & heading_tokens)
    score += 1.0 * len(query_tokens & preview_tokens)
    return score


def select_relevant_tree_sections(query: str, doc_id: str, top_n: int = 3) -> list[dict]:
    sections = TREE_SECTIONS_BY_DOC.get(doc_id, [])
    scored = [
        (tree_section_score(query, section), section)
        for section in sections
    ]
    scored = [(score, section) for score, section in scored if score > 0]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [section for _score, section in scored[:top_n]]


def select_relevant_tree_sections_with_llm(
    query: str,
    slot: str,
    doc_id: str,
    top_n: int = 3,
    state: AgentState | None = None,
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
    if DRY_RUN_WITHOUT_LLM:
        return select_relevant_tree_sections(query, doc_id, top_n)
    try:
        output, usage = call_qwen_json(system, user)
        if state is not None:
            add_usage(state, usage)
    except Exception:
        return select_relevant_tree_sections(query, doc_id, top_n)
    by_id = {section.get("section_id"): section for section in TREE_SECTIONS_BY_DOC.get(doc_id, [])}
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
    state: AgentState | None = None,
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
    if DRY_RUN_WITHOUT_LLM:
        return select_relevant_sections(query, doc_id, top_n)
    try:
        output, usage = call_qwen_json(system, user)
        if state is not None:
            add_usage(state, usage)
    except Exception:
        return select_relevant_sections(query, doc_id, top_n)
    allowed = set(catalog)
    selected = []
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
    state: AgentState | None = None,
) -> list[dict]:
    doc_sections = DOC_SECTIONS_BY_DOC.get(doc_id, [])
    if doc_sections:
        if use_llm_sections:
            selected_doc_sections = select_relevant_doc_sections_with_llm(
                query,
                slot or query,
                doc_id,
                section_top_n,
                state=state,
            )
        else:
            selected_doc_sections = select_relevant_doc_sections(query, doc_id, section_top_n)
        if selected_doc_sections:
            allowed_page_ids = {
                page_id
                for section in selected_doc_sections
                for page_id in (section.get("descendant_page_ids") or section.get("page_ids") or [])
            }
            hits = [
                (score, page)
                for score, page in BM25_INDEX.score(query, allowed_doc_ids={doc_id})
                if page.get("page_id") in allowed_page_ids
            ][:top_k]
            selected_sections = [
                " > ".join(section.get("heading_path") or [])
                for section in selected_doc_sections
            ]
            selected_section_ids = [section.get("section_id") for section in selected_doc_sections]
            if hits:
                evidence = [page_to_evidence(score, page, source + "_doc_tree_section") for score, page in hits]
                for item in evidence:
                    item["selected_section_ids"] = selected_section_ids
                    item["selected_sections"] = selected_sections
                context = section_context_evidence(
                    selected_doc_sections,
                    source + "_doc_tree_section",
                    selected_section_ids,
                    selected_sections,
                )
                evidence = merge_evidence(evidence, context)
                return evidence
            context = section_context_evidence(
                selected_doc_sections,
                source + "_doc_tree_section",
                selected_section_ids,
                selected_sections,
            )
            if context:
                return context

    tree_sections = TREE_SECTIONS_BY_DOC.get(doc_id, [])
    if tree_sections:
        if use_llm_sections:
            selected_tree_sections = select_relevant_tree_sections_with_llm(
                query,
                slot or query,
                doc_id,
                section_top_n,
                state=state,
            )
        else:
            selected_tree_sections = select_relevant_tree_sections(query, doc_id, section_top_n)
        if selected_tree_sections:
            allowed_page_ids = {
                page_id
                for section in selected_tree_sections
                for page_id in (section.get("descendant_page_ids") or section.get("page_ids") or [])
            }
            hits = [
                (score, page)
                for score, page in BM25_INDEX.score(query, allowed_doc_ids={doc_id})
                if page.get("page_id") in allowed_page_ids
            ][:top_k]
            selected_sections = [
                " > ".join(section.get("heading_path") or [])
                for section in selected_tree_sections
            ]
            selected_section_ids = [section.get("section_id") for section in selected_tree_sections]
            if hits:
                evidence = [page_to_evidence(score, page, source + "_tree_section") for score, page in hits]
                for item in evidence:
                    item["selected_section_ids"] = selected_section_ids
                    item["selected_sections"] = selected_sections
                context = section_context_evidence(
                    selected_tree_sections,
                    source + "_tree_section",
                    selected_section_ids,
                    selected_sections,
                )
                evidence = merge_evidence(evidence, context)
                return evidence
            context = section_context_evidence(
                selected_tree_sections,
                source + "_tree_section",
                selected_section_ids,
                selected_sections,
            )
            if context:
                return context

    if use_llm_sections:
        section_headings = select_relevant_sections_with_llm(
            query,
            slot or query,
            doc_id,
            section_top_n,
            state=state,
        )
    else:
        section_headings = select_relevant_sections(query, doc_id, section_top_n)
    if not section_headings:
        return retrieve_one_doc(query, doc_id, top_k, source + "_doc")
    allowed_page_ids = {
        page.get("page_id")
        for heading in section_headings
        for page in SECTIONS_BY_DOC.get(doc_id, {}).get(heading, [])
    }
    hits = [
        (score, page)
        for score, page in BM25_INDEX.score(query, allowed_doc_ids={doc_id})
        if page.get("page_id") in allowed_page_ids
    ][:top_k]
    if not hits:
        return retrieve_one_doc(query, doc_id, top_k, source + "_doc_fallback")
    evidence = [page_to_evidence(score, page, source + "_section") for score, page in hits]
    for item in evidence:
        item["selected_sections"] = section_headings
    return evidence


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


def expand_forward_evidence_ids(
    evidence: list[dict],
    evidence_ids: list[str],
    forward: int = 4,
    source_suffix: str = "_anchor_expand",
) -> list[dict]:
    expanded: list[dict] = []
    requested = {str(evidence_id) for evidence_id in evidence_ids or [] if evidence_id}
    if not requested:
        return expanded
    seen = {item["evidence_id"] for item in evidence}
    for item in evidence:
        if item["evidence_id"] not in requested:
            continue
        if not is_expandable_anchor_evidence(item):
            if VERBOSE_CONSOLE:
                print(f"[anchor expand skipped] id={item['evidence_id']} reason=not_incomplete_anchor", flush=True)
            continue
        pages = PAGES_BY_DOC.get(item["doc_id"], [])
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
    anchor_phrases = ("比例为", "数值为", "标准为", "如下", "见下表", "下表", "下列")
    tail = compact[-40:]
    return any(phrase in tail for phrase in anchor_phrases) and not re.search(r"\d+%|<td>|</td>", tail)


def merge_evidence(existing: list[dict], incoming: list[dict]) -> list[dict]:
    seen = {item["evidence_id"] for item in existing}
    merged = list(existing)
    for item in incoming:
        if item["evidence_id"] in seen:
            continue
        merged.append(item)
        seen.add(item["evidence_id"])
    return merged


class CompactHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"tr", "p", "div", "br", "h1", "h2", "h3", "h4", "li"}:
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"tr", "p", "div", "h1", "h2", "h3", "h4", "li"}:
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append(" | ")

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.parts.append(text)

    def text(self) -> str:
        text = "".join(self.parts)
        text = re.sub(r"[ \t]*\|[ \t]*", " | ", text)
        text = re.sub(r"(?:[ \t]*\|[ \t]*){2,}", " | ", text)
        lines = [re.sub(r"\s+", " ", line).strip(" |") for line in text.splitlines()]
        return "\n".join(line for line in lines if line)


def compact_html_text(text: str) -> str:
    if not COMPACT_HTML_EVIDENCE or "<" not in text or ">" not in text:
        return text
    parser = CompactHTMLParser()
    try:
        parser.feed(text)
        compacted = parser.text()
    except Exception:
        compacted = re.sub(r"<[^>]+>", " ", text)
    compacted = unescape(compacted)
    compacted = re.sub(r"\n{3,}", "\n\n", compacted).strip()
    return compacted or text


def format_evidence(evidence: list[dict], max_chars: int = 7000) -> str:
    chunks: list[str] = []
    used = 0
    for item in evidence:
        text = compact_html_text(item.get("text", ""))
        text = re.sub(r"[ \t]+", " ", text).strip()
        if used >= max_chars:
            break
        remaining = max_chars - used
        if len(text) > remaining:
            text = text[:remaining]
        heading = " > ".join(item.get("heading_path") or [])
        chunks.append(f"[{item['evidence_id']}] doc={item['doc_id']} heading={heading}\n{text}")
        used += len(text)
    return "\n\n".join(chunks)


def print_evidence_preview(evidence: list[dict]) -> None:
    for ev in evidence:
        heading = " > ".join(ev.get("heading_path") or [])
        print(f"- {ev['doc_id']} {ev['page_id']} score={ev['score']:.3f} heading={heading}", flush=True)
        if PRINT_EVIDENCE_TEXT:
            text = re.sub(r"\s+", " ", ev.get("text", "")).strip()
            print(f"  text: {text[:EVIDENCE_TEXT_PREVIEW_CHARS]}", flush=True)


def merge_memory_slots(existing: list[dict], filled_slots: list[dict]) -> list[dict]:
    merged = list(existing)
    index = {item.get("slot"): idx for idx, item in enumerate(merged)}
    for item in filled_slots or []:
        if not isinstance(item, dict):
            continue
        slot = str(item.get("slot") or "").strip()
        if not slot:
            continue
        compact = {
            "slot": slot[:80],
            "value": str(item.get("value") or "")[:160],
            "evidence_ids": list(item.get("evidence_ids") or [])[:5],
        }
        if slot in index:
            merged[index[slot]] = compact
        else:
            index[slot] = len(merged)
            merged.append(compact)
    return merged


def filter_task_audit_slots(filled_slots: list[dict]) -> list[dict]:
    """Keep audit memory to rules and inputs; final task results are computed later."""
    blocked_slot_terms = ("最终", "结果", "答案", "实际退保金额", "退保金额", "所得金额", "计算结果", "排序")
    kept: list[dict] = []
    for item in filled_slots or []:
        if not isinstance(item, dict):
            continue
        slot = maybe_fix_mojibake(str(item.get("slot") or "")).strip()
        value = maybe_fix_mojibake(str(item.get("value") or "")).strip()
        if any(term in slot for term in blocked_slot_terms):
            continue
        if any(term in value for term in ("选项A", "选项B", "选项C", "选项D", "以选项")):
            continue
        kept.append(item)
    return kept


def format_memory_slots(memory_slots: list[dict]) -> str:
    if not memory_slots:
        return "[]"
    lines = []
    for item in memory_slots:
        evidence_ids = ", ".join(item.get("evidence_ids") or [])
        lines.append(f"- {item.get('slot')}: {item.get('value')} [{evidence_ids}]")
    return "\n".join(lines)


def slot_key(text: str) -> str:
    fixed = maybe_fix_mojibake(str(text or "")).lower()
    return re.sub(r"[\s\W_]+", "", fixed)


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


def use_task_reasoning_path(question: dict) -> bool:
    question_type = str(question.get("type") or "").strip()
    answer_format = str(question.get("answer_format") or "").strip().lower()
    if question_type == "计算题":
        return True
    if question_type == "推理判断" and answer_format != "multi":
        return True
    return False


def select_judge_evidence(option_state: dict) -> list[dict]:
    audit = option_state.get("audit") or {}
    selected_ids: list[str] = []
    for slot in option_state.get("memory_slots") or []:
        for evidence_id in slot.get("evidence_ids") or []:
            if evidence_id not in selected_ids:
                selected_ids.append(evidence_id)
    for slot in audit.get("filled_slots") or []:
        for evidence_id in slot.get("evidence_ids") or []:
            if evidence_id not in selected_ids:
                selected_ids.append(evidence_id)
    if not selected_ids:
        return option_state.get("evidence", [])

    evidence_by_id = {item.get("evidence_id"): item for item in option_state.get("evidence", [])}
    selected = [evidence_by_id[evidence_id] for evidence_id in selected_ids if evidence_id in evidence_by_id]
    return selected or option_state.get("evidence", [])


def load_sparkai_api_key() -> str:
    api_key = os.getenv(SPARKAI_API_KEY_ENV)
    if api_key:
        return api_key.strip()
    key_file = Path(SPARKAI_API_KEY_FILE)
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()
    raise RuntimeError(f"Set {SPARKAI_API_KEY_ENV} or create {SPARKAI_API_KEY_FILE}.")


def extract_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"No JSON object found: {text[:300]}")
    return json.loads(match.group(0))


def call_qwen_json(system_prompt: str, user_prompt: str) -> tuple[dict, dict[str, int]]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Missing dependency: openai. Run `pip install openai`.") from exc

    api_key = load_sparkai_api_key()
    if VERBOSE_CONSOLE:
        print(f"[spark request] model={QWEN_MODEL} chars={len(system_prompt) + len(user_prompt)}", flush=True)
    client = OpenAI(api_key=api_key, base_url=SPARKAI_BASE_URL)
    last_error: Exception | None = None
    for attempt in range(1, REQUEST_RETRY_TIMES + 1):
        try:
            response = client.chat.completions.create(
                model=QWEN_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            break
        except Exception as exc:
            last_error = exc
            if VERBOSE_CONSOLE:
                print(f"[spark request error] attempt={attempt}/{REQUEST_RETRY_TIMES} error={exc}", flush=True)
            if attempt < REQUEST_RETRY_TIMES:
                time.sleep(2 * attempt)
    else:
        raise RuntimeError(f"SparkAI request failed after {REQUEST_RETRY_TIMES} attempts: {last_error}") from last_error

    usage = getattr(response, "usage", None) or {}
    if VERBOSE_CONSOLE:
        print(f"[spark response] total_tokens={getattr(usage, 'total_tokens', 0) if not isinstance(usage, dict) else usage.get('total_tokens', 0)}", flush=True)
    content = response.choices[0].message.content or ""
    usage_dict = usage if isinstance(usage, dict) else {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }
    return extract_json_object(content), {
        "prompt_tokens": int(usage_dict.get("prompt_tokens") or 0),
        "completion_tokens": int(usage_dict.get("completion_tokens") or 0),
        "total_tokens": int(usage_dict.get("total_tokens") or 0),
    }


def doc_aliases(doc_ids: list[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    names = ["第一份文档", "第二份文档", "第三份文档", "第四份文档"]
    for idx, doc_id in enumerate(doc_ids):
        aliases[doc_id] = names[idx] if idx < len(names) else f"第{idx + 1}份文档"
    return aliases


def option_claim(state: AgentState) -> str:
    option = state["current_option"]
    option_state = state.get("option_states", {}).get(option, {})
    if "claim" in option_state:
        return maybe_fix_mojibake(option_state["claim"])
    return maybe_fix_mojibake(state["question"]["options"][option])


def build_doc_router_prompt(state: AgentState) -> tuple[str, str]:
    option = state["current_option"]
    doc_lines = "\n".join(f"- {alias}: {doc_id}" for doc_id, alias in state["doc_aliases"].items())
    system = "你是Doc Router。判断当前选项需要检索哪些参考文档。只输出JSON。"
    user = f"""
题干：{maybe_fix_mojibake(state["question"]["question"])}
当前选项：{option}. {option_claim(state)}

可用文档：
{doc_lines}

请输出：
{{
  "target_doc_ids": ["doc_id"],
  "doc_reasons": {{"doc_id": "为什么需要该文档"}},
  "reason": "简短说明"
}}
"""
    return system, user


def build_audit_prompt(state: AgentState) -> tuple[str, str]:
    option = state["current_option"]
    option_state = state["option_states"][option]
    system = "你是证据充分性审查Agent。判断当前证据是否足以验证当前选项；不足则给出缺失槽位和短检索词。只输出JSON。"
    user = f"""
题干：{maybe_fix_mojibake(state["question"]["question"])}
当前选项：{option}. {option_claim(state)}
目标文档：{json.dumps(option_state.get("target_doc_ids", []), ensure_ascii=False)}

已确认事实：
{format_memory_slots(option_state.get("memory_slots", []))}

当前证据：
{format_evidence(option_state.get("evidence", []))}

请输出：
{{
  "can_judge": true,
  "filled_slots": [{{"slot": "事实槽位", "value": "事实值", "evidence_ids": ["证据ID"]}}],
  "missing_slots": [{{"slot": "缺失槽位", "target_doc_ids": ["doc_id"], "followup_query": "短关键词"}}]
}}
"""
    return system, user


def build_judge_prompt(state: AgentState) -> tuple[str, str]:
    option = state["current_option"]
    option_state = state["option_states"][option]
    system = "你是做题Agent。只能基于给定证据判断当前选项是否正确。只输出JSON。"
    user = f"""
题干：{maybe_fix_mojibake(state["question"]["question"])}
题型：{state["question"].get("answer_format")}
当前选项：{option}. {option_claim(state)}

证据：
{format_evidence(select_judge_evidence(option_state), max_chars=9000)}

请输出：
{{
  "verdict": true,
  "confidence": 0.0,
  "reason": "简短依据",
  "answer": "A/B/C/D/T/F或空"
}}
"""
    return system, user


def make_option_state(claim: str) -> dict:
    return {
        "claim": maybe_fix_mojibake(claim),
        "status": "pending",
        "round": 0,
        "target_doc_ids": [],
        "searches": [],
        "router": {},
        "evidence": [],
        "new_evidence_ids": [],
        "memory_slots": [],
        "audit": {},
        "judgment": {},
    }


def init_state(state: AgentState) -> AgentState:
    question = state["question"]
    answer_format = question.get("answer_format")
    if answer_format == "tf":
        options = {"T": make_option_state(question["question"])}
        order = ["T"]
    else:
        options = {key: make_option_state(value) for key, value in question["options"].items()}
        order = list(question["options"].keys())
    state.update(
        {
            "qid": question["qid"],
            "doc_ids": question.get("doc_ids") or [],
            "doc_aliases": doc_aliases(question.get("doc_ids") or []),
            "option_order": order,
            "option_index": 0,
            "current_option": order[0],
            "option_states": options,
            "status": "running",
            "trace": [],
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    )
    add_trace(state, "init")
    return state


def route_docs_node(state: AgentState) -> AgentState:
    option = state["current_option"]
    option_state = state["option_states"][option]
    if DRY_RUN_WITHOUT_LLM:
        system, user = build_doc_router_prompt(state)
        router = dry_json(system, user, "router")
    else:
        system, user = build_doc_router_prompt(state)
        router, usage = call_qwen_json(system, user)
        add_usage(state, usage)
    targets = [doc_id for doc_id in router.get("target_doc_ids", []) if doc_id in state["doc_ids"]]
    if not targets:
        targets = list(state["doc_ids"])
    searches = []
    for item in router.get("searches") or []:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query") or item.get("slot") or "").strip()
        search_doc_ids = [doc_id for doc_id in item.get("target_doc_ids", []) if doc_id in state["doc_ids"]]
        if not search_doc_ids:
            search_doc_ids = targets
        if query:
            searches.append(
                {
                    "slot": str(item.get("slot") or "")[:80],
                    "target_doc_ids": search_doc_ids,
                    "query": query,
                }
            )
    option_state["router"] = router
    option_state["target_doc_ids"] = targets
    option_state["searches"] = searches
    option_state["status"] = "routed"
    add_trace(state, f"{option}: route_docs -> {targets}")
    if VERBOSE_CONSOLE:
        print(f"\n[option {option}] {option_state['claim']}", flush=True)
        print("[doc router]", json.dumps(router, ensure_ascii=False, indent=2), flush=True)
    return state


def retrieve_initial_node(state: AgentState) -> AgentState:
    option = state["current_option"]
    option_state = state["option_states"][option]
    searches = option_state.get("searches") or []
    if searches:
        evidence = []
        for search in searches:
            evidence.extend(
                retrieve_multi_doc(search["query"], search["target_doc_ids"], min(INITIAL_TOP_K, 2), "initial")
            )
    else:
        query = f"{maybe_fix_mojibake(state['question']['question'])}\n{option}. {option_state['claim']}"
        evidence = retrieve_multi_doc(query, option_state["target_doc_ids"], INITIAL_TOP_K, "initial")
    option_state["evidence"] = merge_evidence(option_state["evidence"], evidence)
    option_state["new_evidence_ids"] = [item["evidence_id"] for item in evidence]
    option_state["status"] = "retrieved"
    add_trace(state, f"{option}: initial retrieve {len(evidence)}")
    if VERBOSE_CONSOLE:
        if searches:
            print("[initial searches]", json.dumps(searches, ensure_ascii=False, indent=2), flush=True)
        print("[initial evidence]", flush=True)
        print_evidence_preview(evidence)
    return state


def audit_node(state: AgentState) -> AgentState:
    option = state["current_option"]
    option_state = state["option_states"][option]
    if DRY_RUN_WITHOUT_LLM:
        system, user = build_audit_prompt(state)
        audit = dry_json(system, user, "audit")
    else:
        system, user = build_audit_prompt(state)
        audit, usage = call_qwen_json(system, user)
        add_usage(state, usage)
    option_state["audit"] = audit
    option_state["memory_slots"] = merge_memory_slots(option_state.get("memory_slots", []), audit.get("filled_slots") or [])
    option_state["new_evidence_ids"] = []
    option_state["round"] = option_state.get("round", 0) + 1
    option_state["status"] = "ready_to_judge" if audit.get("can_judge") else "need_more_evidence"
    add_trace(state, f"{option}: audit can_judge={audit.get('can_judge')}")
    if VERBOSE_CONSOLE:
        public_audit = {key: audit.get(key) for key in ("can_judge", "filled_slots", "missing_slots")}
        print("[audit output]", json.dumps(public_audit, ensure_ascii=False, indent=2), flush=True)
        print("[memory slots]", json.dumps(option_state.get("memory_slots", []), ensure_ascii=False, indent=2), flush=True)
    return state


def route_after_audit(state: AgentState) -> str:
    option_state = state["option_states"][state["current_option"]]
    if option_state.get("status") == "ready_to_judge":
        return "judge"
    if option_state.get("round", 0) >= MAX_AUDIT_ROUNDS:
        return "judge"
    missing = option_state.get("audit", {}).get("missing_slots") or []
    return "retrieve_more" if missing else "judge"


def retrieve_more_node(state: AgentState) -> AgentState:
    option = state["current_option"]
    option_state = state["option_states"][option]
    missing = option_state.get("audit", {}).get("missing_slots") or []
    for item in missing:
        if isinstance(item, dict):
            query = item.get("followup_query") or item.get("query") or item.get("slot") or ""
            target_doc_ids = [doc_id for doc_id in item.get("target_doc_ids", []) if doc_id in state["doc_ids"]]
        else:
            query = str(item)
            target_doc_ids = []
        if not target_doc_ids:
            target_doc_ids = option_state.get("target_doc_ids") or state["doc_ids"]
        if not query:
            continue
        evidence = retrieve_multi_doc(query, target_doc_ids, FOLLOWUP_TOP_K, "followup")
        option_state["evidence"] = merge_evidence(option_state["evidence"], evidence)
        option_state["new_evidence_ids"] = list(
            dict.fromkeys(option_state.get("new_evidence_ids", []) + [item["evidence_id"] for item in evidence])
        )
        if VERBOSE_CONSOLE:
            print("[followup query]", query, flush=True)
            print("[followup target_doc_ids]", target_doc_ids, flush=True)
            print_evidence_preview(evidence)
    option_state["status"] = "retrieved_more"
    add_trace(state, f"{option}: followup retrieve {len(missing)} missing slots")
    return state


def judge_node(state: AgentState) -> AgentState:
    option = state["current_option"]
    option_state = state["option_states"][option]
    all_evidence = option_state.get("evidence", [])
    judge_evidence = select_judge_evidence(option_state)
    option_state["evidence"] = judge_evidence
    if VERBOSE_CONSOLE:
        print(f"[judge evidence] selected={len(judge_evidence)} total={len(all_evidence)}", flush=True)
        print(
            f"[judge input] {'memory_slots' if option_state.get('memory_slots') else 'evidence_text'}",
            flush=True,
        )
    try:
        if DRY_RUN_WITHOUT_LLM:
            system, user = build_judge_prompt(state)
            judgment = dry_json(system, user, "judge")
        else:
            system, user = build_judge_prompt(state)
            judgment, usage = call_qwen_json(system, user)
            add_usage(state, usage)
    finally:
        option_state["evidence"] = all_evidence
    option_state["judgment"] = judgment
    option_state["status"] = judgment.get("status", "done")
    add_trace(state, f"{option}: judge verdict={judgment.get('verdict')}")
    if VERBOSE_CONSOLE:
        public_judgment = {key: judgment.get(key) for key in ("verdict", "confidence", "evidence_ids")}
        print("[judge output]", json.dumps(public_judgment, ensure_ascii=False, indent=2), flush=True)
    return state


def advance_option_node(state: AgentState) -> AgentState:
    state["option_index"] += 1
    if state["option_index"] < len(state["option_order"]):
        state["current_option"] = state["option_order"][state["option_index"]]
        add_trace(state, f"advance to {state['current_option']}")
    else:
        add_trace(state, "all options processed")
    return state


def route_after_advance(state: AgentState) -> str:
    return "route_docs" if state["option_index"] < len(state["option_order"]) else "finalize"


def option_confidence(state: AgentState, option: str) -> float:
    judgment = state["option_states"].get(option, {}).get("judgment", {})
    try:
        return float(judgment.get("confidence", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def option_verdict(state: AgentState, option: str) -> bool | None:
    return state["option_states"].get(option, {}).get("judgment", {}).get("verdict")


def choose_single_answer(state: AgentState) -> str:
    option_order = [key for key in state.get("option_order", []) if key in state["option_states"]]
    true_options = [key for key in option_order if option_verdict(state, key) is True]
    if true_options:
        return max(true_options, key=lambda key: (option_confidence(state, key), -option_order.index(key)))

    judged_options = [key for key in option_order if option_verdict(state, key) is not None]
    if judged_options:
        fallback = max(judged_options, key=lambda key: (option_confidence(state, key), -option_order.index(key)))
        add_trace(state, f"single-answer fallback: no true option, choose highest-confidence judged option {fallback}")
        return fallback

    add_trace(state, "single-answer fallback: no judged option")
    return ""


def finalize_node(state: AgentState) -> AgentState:
    option_order = [key for key in state.get("option_order", list("ABCD")) if key in state["option_states"]]
    true_options = [key for key in option_order if option_verdict(state, key) is True]
    answer_format = state["question"].get("answer_format")
    question_type = state["question"].get("type")
    if state.get("task_reasoning_debug"):
        state["status"] = "done"
        add_trace(state, f"finalize format={answer_format} type={question_type} answer={state['final_answer']}")
        return state
    if answer_format == "multi":
        state["final_answer"] = "".join(true_options)
    elif answer_format == "tf":
        verdict = option_verdict(state, "T")
        if verdict is True:
            state["final_answer"] = "A"
        elif verdict is False:
            state["final_answer"] = "B"
        else:
            state["final_answer"] = ""
            add_trace(state, "tf-answer fallback: T was not judged")
    elif answer_format == "mcq":
        state["final_answer"] = choose_single_answer(state)
    else:
        state["final_answer"] = "".join(true_options)
    state["status"] = "done" if all(v.get("status") == "done" for v in state["option_states"].values()) else "partial"
    add_trace(state, f"finalize format={answer_format} answer={state['final_answer']}")
    return state


def reasoning_options_text(question: dict) -> str:
    return "\n".join(
        f"{key}. {maybe_fix_mojibake(value)}" for key, value in (question.get("options") or {}).items()
    )


def generalize_query_terms(text: str) -> list[str]:
    text = maybe_fix_mojibake(str(text or ""))
    queries: list[str] = []
    has_policy_year = bool(re.search(r"第?\s*\d+\s*(?:个)?\s*保单年度|第\s*\d+\s*年", text))
    if has_policy_year:
        queries.append("保单年度 区间 现金价值")
        queries.append("保单年度 范围 费用比例")
        queries.append("解除合同 现金价值 年度")

    has_age = bool(re.search(r"\d+\s*(?:周岁|岁)", text))
    if has_age:
        queries.append("年龄区间 给付比例 表")
        queries.append("周岁 范围 保险金比例")
        queries.append("年龄 分档 给付规则")

    if any(term in text for term in ("退保", "解除合同", "所得金额")):
        queries.append("现金价值 解除合同 所得金额")
        queries.append("现金价值 累计所交保险费 保单账户累计收益")
        queries.append("账户价值 退保费用 现金价值")
    return list(dict.fromkeys(query for query in queries if query))



def rewrite_followup_queries(
    state: AgentState,
    task: dict,
    missing_item: dict,
    previous_evidence: list[dict],
) -> list[str]:
    slot = maybe_fix_mojibake(str(missing_item.get("slot") or "")).strip()
    raw_query = maybe_fix_mojibake(str(missing_item.get("query") or missing_item.get("followup_query") or slot)).strip()
    base_text = " ".join(
        [
            maybe_fix_mojibake(state["question"]["question"]),
            maybe_fix_mojibake(str(task.get("task") or "")),
            slot,
            raw_query,
        ]
    )
    queries = [raw_query]
    queries.extend(generalize_query_terms(base_text))
    if DRY_RUN_WITHOUT_LLM:
        return list(dict.fromkeys(query for query in queries if query))[:5]

    failed_titles = []
    for evidence in previous_evidence[-10:]:
        title = " > ".join(evidence.get("heading_path") or evidence.get("selected_sections") or [])
        if title:
            failed_titles.append(title)
    system = (
        "你是检索词改写Agent。根据缺失槽位、题干中的具体年份/年龄/金额、以及失败证据标题，生成新的BM25检索词。"
        "如果题目给出具体第N年或具体年龄，要考虑原文可能写成范围区间。"
        "如果原检索词一直围绕手续/退保流程但任务问金额，应换到现金价值、账户价值、费用比例、年度区间、给付比例等视角。"
        "只输出JSON。"
    )
    user = f"""
题干：{maybe_fix_mojibake(state["question"]["question"])}
当前任务：{json.dumps(task, ensure_ascii=False)}
缺失槽位：{slot}
原始检索词：{raw_query}
失败证据标题：
{json.dumps(failed_titles, ensure_ascii=False, indent=2)}

请输出：
{{
  "queries": ["3到8个关键词", "考虑范围区间的关键词", "换视角关键词"]
}}
"""
    try:
        output, usage = call_qwen_json(system, user)
        add_usage(state, usage)
        for query in output.get("queries") or []:
            query = maybe_fix_mojibake(str(query or "")).strip()
            if query:
                queries.append(query)
    except Exception:
        pass
    return list(dict.fromkeys(query for query in queries if query))[:6]


# ==================== HUMAN TOC REASONING PATH START ====================
# Active reasoning route: task decomposition -> TOC routing -> section retrieval -> audit.


def build_task_reasoning_plan_prompt(state: AgentState) -> tuple[str, str]:
    doc_lines = "\n".join(f"- {alias}: {doc_id}" for doc_id, alias in state["doc_aliases"].items())
    valid_doc_ids = [str(doc_id) for doc_id in state.get("doc_ids", [])]
    system = (
        "你是推理/计算题任务拆解Agent。先整体分析题目和选项，再把整题拆成若干独立子任务。"
        "每个任务通常对应一个产品、对象、年份、指标或金额。"
        "每个任务必须给出题干已知事实known_facts，以及需要检索的事实槽位search_slots。"
        "任务还必须给出task_type：retrieve表示需要检索文档，summarize表示只汇总前面任务结果、不再检索文档。"
        "target_doc_ids必须使用真实doc_id，只能从题干给出的doc_ids中选择；不要输出doc_id_1、doc1、文档1等别名。"
        "如果任务只涉及一个产品，target_doc_ids通常只填该产品对应的一个文档。"
        "比较/排序汇总任务一般不需要检索文档，可不拆成检索任务。"
        "search query必须是3到8个关键词。只输出JSON。"
    )
    user = f"""
题干：{maybe_fix_mojibake(state["question"]["question"])}

选项：
{reasoning_options_text(state["question"])}

可用文档：
{doc_lines}

合法target_doc_ids只能取这些字符串：
{json.dumps(valid_doc_ids, ensure_ascii=False)}

请输出：
{{
  "tasks": [
    {{
      "task": "任务名",
      "task_type": "retrieve",
      "target_doc_ids": ["doc_id"],
      "known_facts": [{{"slot": "题干已知事实", "value": "数值或条件"}}],
      "search_slots": [{{"slot": "需要检索的条款事实", "query": "短关键词"}}]
    }}
  ]
}}
"""
    return system, user


def resolve_plan_doc_ids(raw_doc_ids: list, default_doc_ids: list[str], doc_aliases: dict[str, str] | None = None) -> list[str]:
    doc_aliases = doc_aliases or {}
    valid_ids = [str(doc_id) for doc_id in default_doc_ids]
    alias_to_id = {str(alias): str(doc_id) for doc_id, alias in doc_aliases.items()}
    resolved: list[str] = []
    for raw in raw_doc_ids or []:
        value = maybe_fix_mojibake(str(raw or "")).strip()
        if not value:
            continue
        doc_id = value if value in valid_ids else alias_to_id.get(value)
        if not doc_id:
            match = re.fullmatch(r"doc[_\-\s]*id[_\-\s]*(\d+)", value, flags=re.I) or re.fullmatch(r"doc[_\-\s]*(\d+)", value, flags=re.I)
            if match:
                index = int(match.group(1)) - 1
                if 0 <= index < len(valid_ids):
                    doc_id = valid_ids[index]
        if doc_id and doc_id in valid_ids and doc_id not in resolved:
            resolved.append(doc_id)
    return resolved


def normalize_task_reasoning_plan(plan: dict, default_doc_ids: list[str], doc_aliases: dict[str, str] | None = None) -> list[dict]:
    tasks: list[dict] = []
    for item in plan.get("tasks") or []:
        if not isinstance(item, dict):
            continue
        task_name = maybe_fix_mojibake(str(item.get("task") or "")).strip()
        if not task_name:
            continue
        task_type = maybe_fix_mojibake(str(item.get("task_type") or "")).strip().lower()
        if not task_type:
            if not item.get("search_slots") and re.search(r"(比较|排序|汇总|对比)", task_name):
                task_type = "summarize"
            else:
                task_type = "retrieve"
        target_doc_ids = resolve_plan_doc_ids(item.get("target_doc_ids", []), default_doc_ids, doc_aliases)
        if task_type != "summarize" and not target_doc_ids:
            target_doc_ids = default_doc_ids
        if task_type == "summarize":
            target_doc_ids = []
        known_facts = []
        for fact in item.get("known_facts") or []:
            if not isinstance(fact, dict):
                continue
            slot = maybe_fix_mojibake(str(fact.get("slot") or "")).strip()
            value = maybe_fix_mojibake(str(fact.get("value") or "")).strip()
            if slot and value:
                known_facts.append({"slot": slot[:80], "value": value[:160], "evidence_ids": ["题干"]})
        search_slots = []
        for slot_item in item.get("search_slots") or item.get("slots") or []:
            if not isinstance(slot_item, dict):
                continue
            slot = maybe_fix_mojibake(str(slot_item.get("slot") or "")).strip()
            if not slot or is_speculative_required_slot(slot):
                continue
            query = maybe_fix_mojibake(str(slot_item.get("query") or slot)).strip()
            search_slots.append(
                {
                    "slot": slot[:100],
                    "target_doc_ids": target_doc_ids,
                    "query": query[:120],
                }
            )
        if not search_slots:
            search_slots.append({"slot": task_name[:80], "target_doc_ids": target_doc_ids, "query": task_name[:120]})
        tasks.append(
            {
                "task": task_name[:120],
                "task_type": task_type,
                "target_doc_ids": target_doc_ids,
                "known_facts": known_facts,
                "search_slots": search_slots,
            }
        )
    if not tasks:
        tasks.append(
            {
                "task": "整题关键事实计算",
                "task_type": "retrieve",
                "target_doc_ids": default_doc_ids,
                "known_facts": [],
                "search_slots": [
                    {
                        "slot": "整题关键规则",
                        "target_doc_ids": default_doc_ids,
                        "query": "退保 现金价值 身故保险金 计算规则",
                    }
                ],
            }
        )
    return tasks


def build_task_reasoning_audit_prompt(
    state: AgentState,
    task: dict,
    task_memory: list[dict],
    evidence: list[dict],
) -> tuple[str, str]:
    system = (
        "你是单任务证据审查Agent。只判断当前任务是否已经具备计算/判断所需事实。"
        "题干已知事实可以直接使用；当前证据用于补齐条款规则、公式、比例、定义。"
        "如果证据足够，can_solve=true并提取filled_slots。"
        "filled_slots只能写规则、公式、比例、定义、区间条件、题干明确给出的已知事实；不要在审查阶段计算或填写最终金额。"
        "如果需要计算金额，只判断公式和输入数值是否齐全，把公式/比例作为filled_slots，最终计算留给单任务求解Agent。"
        "禁止把选项中的金额当作事实；禁止根据选项倒推出filled_slots。"
        "如果缺事实，missing_slots只写仍缺少的正向事实或计算依据，并给出3到8个关键词。"
        "证据不足时必须输出failure_analysis，说明当前证据为什么不够，以及下一轮应同章节换检索词还是回目录换章节。"
        "如果当前证据只是手续、流程、说明文字，但任务需要金额/公式/比例/区间，应明确建议回目录寻找价值、金额、账户、费用、比例、给付、领取等语义章节。"
        "只输出JSON。"
    )
    user = f"""
题干：{maybe_fix_mojibake(state["question"]["question"])}

当前任务：
{json.dumps(task, ensure_ascii=False, indent=2)}

已确认事实：
{format_memory_slots(task_memory)}

当前证据：
{format_evidence(evidence)}

请输出：
{{
  "can_solve": true,
  "filled_slots": [{{"slot": "规则/比例/公式/定义/题干已知事实", "value": "原文规则或题干明确数值，不要写最终计算金额", "evidence_ids": ["证据ID或题干"]}}],
  "expand_evidence_ids": ["需要向后补充的锚点证据ID"],
  "failure_analysis": {{"reason": "证据不足原因", "next_action": "同章节换词/回目录换章节", "avoid_titles": ["不应继续优先的标题"]}},
  "missing_slots": [{{"slot": "缺失事实槽位", "target_doc_ids": ["doc_id"], "followup_query": "短关键词"}}]
}}
"""
    return system, user


def build_task_reasoning_result_prompt(state: AgentState, task: dict, task_memory: list[dict], audit: dict) -> tuple[str, str]:
    system = (
        "你是单任务求解Agent。只基于题干和已确认事实完成当前任务。"
        "如果事实足以计算或判断，status=complete；如果真正缺关键公式、比例或数值，status=blocked。"
        "禁止根据选项倒推事实，禁止使用选项中的金额修正计算。"
        "计算题必须逐项写出公式、代入和算术过程；百分比要先换算成小数或明确乘法。"
        "如果发现已确认事实里存在互相冲突的计算结果，以题干数值和证据公式重新计算，并在warnings中说明冲突。"
        "只输出JSON。"
    )
    user = f"""
题干：{maybe_fix_mojibake(state["question"]["question"])}

当前任务：
{json.dumps(task, ensure_ascii=False, indent=2)}

已确认事实：
{format_memory_slots(task_memory)}

审查结果：{json.dumps(audit, ensure_ascii=False)}

请输出：
{{
  "task": "任务名",
  "status": "complete/blocked",
  "formula": "使用的公式；非计算任务可为空",
  "substitution": "代入题干数值；非计算任务可为空",
  "calculation": "逐步算术，例如 10 + 2 * 75% = 10 + 1.5 = 11.5",
  "result": "计算结果或事实结论",
  "basis": "简短依据",
  "evidence_ids": ["证据ID或题干"],
  "warnings": ["如发现冲突或忽略了错误中间结果，在这里说明"],
  "missing_slots": ["真正缺失的关键事实"]
}}
"""
    return system, user


def build_task_reasoning_final_prompt(state: AgentState, task_results: list[dict]) -> tuple[str, str]:
    system = (
        "你是最终作答Agent。只基于题干和各任务结果计算、排序并匹配选项。"
        "选项只能用于最后匹配，不能用于倒推缺失规则，不能用选项金额修正任务结果。"
        "如果任务结果中有formula/substitution/calculation，优先使用这些结构化计算。"
        "如果任务结果内部自相矛盾，要在missing_impact中说明冲突，不要直接用选项补救。"
        "只输出JSON。"
    )
    user = f"""
题干：{maybe_fix_mojibake(state["question"]["question"])}

选项：
{reasoning_options_text(state["question"])}

任务结果：
{json.dumps(task_results, ensure_ascii=False, indent=2)}

请输出：
{{
  "calculations": [{{"item": "对象", "amount": "金额/结论", "basis": "依据"}}],
  "option_checks": [{{"option": "A/B/C/D", "is_match": true, "reason": "简短原因"}}],
  "missing_impact": "无或说明缺失影响",
  "answer": "A/B/C/D"
}}
"""
    return system, user


def doc_root_sections(doc_id: str) -> list[dict]:
    return [section for section in DOC_SECTIONS_BY_DOC.get(doc_id, []) if not section.get("parent_id")]


def doc_child_sections(doc_id: str, parent_section_id: str) -> list[dict]:
    return [
        section
        for section in DOC_SECTIONS_BY_DOC.get(doc_id, [])
        if str(section.get("parent_id") or "") == str(parent_section_id)
    ]


def section_title(section: dict) -> str:
    return maybe_fix_mojibake(str(section.get("title") or " > ".join(section.get("heading_path") or []))).strip()


def task_query_bundle(task: dict) -> str:
    parts: list[str] = [maybe_fix_mojibake(str(task.get("task") or ""))]
    for slot in task.get("search_slots") or []:
        if not isinstance(slot, dict):
            continue
        parts.append(maybe_fix_mojibake(str(slot.get("slot") or "")))
        parts.append(maybe_fix_mojibake(str(slot.get("query") or "")))
    return " ".join(part for part in parts if part).strip()


def compact_doc_toc_for_llm(doc_id: str, max_sections: int = 120) -> list[dict]:
    sections = DOC_SECTIONS_BY_DOC.get(doc_id, [])
    compact: list[dict] = []
    for section in sections[:max_sections]:
        title = section_title(section)
        if not title:
            continue
        compact.append(
            {
                "section_id": section.get("section_id"),
                "title": title,
                "level": section.get("level"),
                "parent_id": section.get("parent_id"),
                "brief": maybe_fix_mojibake(str(section.get("preview") or "")).strip()[:80],
            }
        )
    return compact


def plan_section_queue_with_llm(
    state: AgentState,
    task: dict,
    doc_id: str,
    failed_titles: list[str] | None = None,
    used_queries: list[str] | None = None,
    max_sections: int = 5,
    focus_slot: str = "",
    focus_query: str = "",
) -> dict:
    toc = compact_doc_toc_for_llm(doc_id)
    if not toc:
        return {"information_need": task_query_bundle(task), "candidates": []}
    failed_titles = list(dict.fromkeys(failed_titles or []))[:10]
    used_queries = list(dict.fromkeys(used_queries or []))[:10]
    system = (
        "你是章节候选规划Agent。你只负责根据题目、选项、当前任务和doc目录，一次性选出最可能包含答案的章节队列。"
        "不要回答题目，不要计算最终答案。"
        "先判断当前任务真正需要的信息information_need，再从目录里按优先级选择候选章节。"
        "如果给出了当前缺失槽位/定向检索词，本轮必须优先围绕该槽位找证据，不要泛泛回到整个任务。"
        "候选章节要覆盖不同可能位置：主规则章节、可能引用章节、备选定义章节。"
        "如果任务是计算、金额、比例、排序、年龄或年度条件，优先考虑标题语义包含价值、金额、账户、费用、比例、给付、领取、现金价值、保险金、表格、条款定义的章节。"
        "如果缺失槽位或检索词含计划表、附表、表格、费率表、现金价值表、责任项赔付比例、载明、详见，应优先选择标题或简介含表、计划、附表、比例、费率、责任项的章节。"
        "如果题目中有具体年份、年龄、期限，检索词要考虑原文可能用范围、区间、分档表达，但不要编造具体范围。"
        "流程/手续章节只有在任务确实问办理手续时才排在最前；如果任务问金额，流程章节最多作为备选。"
        "如果提供了失败标题或已用检索词，本次候选和检索词应尽量换视角。"
        "只输出JSON。"
    )
    user = f"""
题干：{maybe_fix_mojibake(state["question"]["question"])}

选项：
{reasoning_options_text(state["question"])}

当前任务：
{json.dumps(task, ensure_ascii=False, indent=2)}

doc_id：{doc_id}

当前缺失槽位：{maybe_fix_mojibake(focus_slot)}
定向检索词：{maybe_fix_mojibake(focus_query)}

失败标题/失败原因：
{json.dumps(failed_titles, ensure_ascii=False, indent=2)}

已用检索词：
{json.dumps(used_queries, ensure_ascii=False, indent=2)}

目录：
{json.dumps(toc, ensure_ascii=False, indent=2)}

请输出：
{{
  "information_need": "当前任务真正需要找到的信息",
  "candidates": [
    {{
      "section_id": "section_id",
      "title": "章节标题",
      "why": "为什么这个章节可能有答案",
      "search_queries": ["1到2个章节内检索词"]
    }}
  ]
}}
"""
    if DRY_RUN_WITHOUT_LLM:
        fallback_query = " ".join(part for part in [focus_slot, focus_query, task_query_bundle(task)] if part)
        fallback = select_relevant_doc_sections(fallback_query, doc_id, max_sections)
        return {
            "information_need": fallback_query,
            "candidates": [
                {
                    "section_id": sec.get("section_id"),
                    "title": section_title(sec),
                    "why": "dry_run",
                    "search_queries": [fallback_query],
                }
                for sec in fallback
            ],
        }
    try:
        output, usage = call_qwen_json(system, user)
        add_usage(state, usage)
    except Exception:
        output = {"information_need": focus_query or task_query_bundle(task), "candidates": []}
    by_id = SECTION_BY_DOC_ID.get(doc_id, {})
    candidates: list[dict] = []
    for item in output.get("candidates") or []:
        if not isinstance(item, dict):
            continue
        section_id = str(item.get("section_id") or "").strip()
        section = by_id.get(section_id)
        if not section:
            continue
        queries = []
        for query in item.get("search_queries") or []:
            query = maybe_fix_mojibake(str(query or "")).strip()
            if query:
                queries.append(query)
        if not queries:
            queries = [focus_query or task_query_bundle(task)]
        candidates.append(
            {
                "section": section,
                "section_id": section_id,
                "title": section_title(section),
                "why": maybe_fix_mojibake(str(item.get("why") or "")).strip(),
                "search_queries": list(dict.fromkeys(queries))[:2],
            }
        )
        if len(candidates) >= max_sections:
            break
    if not candidates:
        fallback_query = " ".join(part for part in [focus_slot, focus_query, task_query_bundle(task)] if part)
        for section in select_relevant_doc_sections(fallback_query, doc_id, max_sections):
            candidates.append(
                {
                    "section": section,
                    "section_id": section.get("section_id"),
                    "title": section_title(section),
                    "why": "lexical fallback",
                    "search_queries": [fallback_query],
                }
            )
    return {
        "information_need": maybe_fix_mojibake(str(output.get("information_need") or focus_query or task_query_bundle(task))).strip(),
        "candidates": candidates,
    }


def retrieve_section_candidate(candidate: dict, doc_id: str, source: str, plan: dict, index: int) -> list[dict]:
    section = candidate.get("section")
    if not section:
        return []
    route_step = {
        "depth": 1,
        "parent_title": None,
        "selected_section_ids": [candidate.get("section_id")],
        "selected_titles": [candidate.get("title")],
        "planned_queries": candidate.get("search_queries") or [],
        "information_need": plan.get("information_need"),
        "retry_strategy": "如果该候选章节证据不足，程序直接尝试下一个候选章节。",
        "stop": True,
        "reason": candidate.get("why"),
        "candidate_index": index,
    }
    evidence: list[dict] = []
    for query in candidate.get("search_queries") or []:
        evidence = merge_evidence(
            evidence,
            retrieve_with_sections(query, doc_id, [section], source, [route_step]),
        )
    return evidence


def choose_child_sections_with_llm(
    state: AgentState,
    task: dict,
    doc_id: str,
    slot: str,
    query: str,
    parent_section: dict | None,
    child_sections: list[dict],
    failed_titles: list[str] | None = None,
    used_queries: list[str] | None = None,
) -> tuple[list[dict], bool, str, list[str], dict]:
    if not child_sections:
        return [], True, "no_children", [], {}
    parent_name = section_title(parent_section) if parent_section else "文档顶层目录"
    candidates = []
    for sec in child_sections[:24]:
        section_id = str(sec.get("section_id") or "")
        brief = maybe_fix_mojibake(str(sec.get("preview") or "")).strip()
        candidates.append(
            {
                "section_id": section_id,
                "title": section_title(sec),
                "level": sec.get("level"),
                "brief": brief[:120],
            }
        )
    if not candidates:
        return [], True, "all_children_excluded", [], {}
    failed_titles = list(dict.fromkeys(failed_titles or []))[:8]
    used_queries = list(dict.fromkeys(used_queries or []))[:8]
    system = (
        "你是目录检索规划Agent。你要像真人查资料一样，先理解题目和选项差异，再结合当前任务和目录标题判断答案可能在哪。"
        "不要回答题目。"
        "第一步输出information_need：说明当前任务真正缺什么信息，不要只复述表面关键词。"
        "第二步分析候选目录：选出最可能包含该信息的章节，可同时保留主候选和备选候选。"
        "如果候选标题已经足够具体，stop=true；如果还应继续看子标题，stop=false。"
        "第三步生成检索词：必须结合信息需求和所选标题生成，不要只复述原始检索词。"
        "如果上一轮章节或检索词失败，本轮必须说明retry_strategy，并尽量换章节或换检索视角；除非确有必要，不要重复相同章节和相同检索词。"
        "具体年份、年龄、期限、金额等点值在原文中可能写成范围、区间或分档，检索词应泛化为范围/区间/分档表达，但不要编造具体范围。"
        "计算、金额、比例、排序类任务应优先寻找定义、公式、比例表、范围条件、金额计算依据；流程/手续标题只有在槽位问手续时才优先。"
        "如果目录中同时存在流程类标题和价值/金额/比例/账户/给付/领取类标题，要说明二者分工，并优先选择能提供计算依据的章节。"
        "只输出JSON。"
    )
    user = f"""
题干：{maybe_fix_mojibake(state["question"]["question"])}
当前任务：{json.dumps(task, ensure_ascii=False, indent=2)}
槽位：{maybe_fix_mojibake(slot)}
原始检索意图：{maybe_fix_mojibake(query)}
doc_id：{doc_id}
当前父标题：{parent_name}
上一轮失败标题：
{json.dumps(failed_titles, ensure_ascii=False, indent=2)}
已用检索词：
{json.dumps(used_queries, ensure_ascii=False, indent=2)}
候选目录：
{json.dumps(candidates, ensure_ascii=False, indent=2)}

请输出：
{{
  "information_need": "当前任务真正需要找到的信息",
  "candidate_sections": [
    {{"section_id": "候选section_id", "why": "为什么该标题可能包含答案"}}
  ],
  "selected_section_ids": ["section_id"],
  "search_queries": ["结合所选章节的3到8个关键词", "换视角关键词"],
  "stop": true,
  "reason": "简短说明",
  "retry_strategy": "如果本轮仍证据不足，下一轮应如何换章节或换检索视角"
}}
"""
    if DRY_RUN_WITHOUT_LLM:
        chosen = child_sections[:1]
        return chosen, False, "dry_run", [query], {}
    try:
        output, usage = call_qwen_json(system, user)
        add_usage(state, usage)
    except Exception:
        return child_sections[:1], False, "fallback", [query], {}
    by_id = {str(sec.get("section_id")): sec for sec in child_sections}
    chosen = []
    for section_id in output.get("selected_section_ids") or []:
        sec = by_id.get(str(section_id).strip())
        if sec and sec not in chosen:
            chosen.append(sec)
        if len(chosen) >= 2:
            break
    if not chosen:
        chosen = child_sections[:1]
    planned_queries = []
    for item in output.get("search_queries") or []:
        planned_query = maybe_fix_mojibake(str(item or "")).strip()
        if planned_query:
            planned_queries.append(planned_query)
    if not planned_queries:
        planned_queries = [query]
    return (
        chosen,
        bool(output.get("stop")),
        maybe_fix_mojibake(str(output.get("reason") or "")).strip(),
        list(dict.fromkeys(planned_queries))[:3],
        {
            "information_need": maybe_fix_mojibake(str(output.get("information_need") or "")).strip(),
            "candidate_sections": output.get("candidate_sections") or [],
            "retry_strategy": maybe_fix_mojibake(str(output.get("retry_strategy") or "")).strip(),
        },
    )


def route_sections_human(
    state: AgentState,
    task: dict,
    doc_id: str,
    slot: str,
    query: str,
    start_section: dict | None = None,
    failed_titles: list[str] | None = None,
    used_queries: list[str] | None = None,
) -> tuple[list[dict], list[dict], list[str]]:
    route_steps: list[dict] = []
    parent_section: dict | None = start_section
    chosen: list[dict] = []
    planned_queries: list[str] = []
    for depth in range(MAX_TOC_ROUTE_DEPTH):
        children = doc_root_sections(doc_id) if parent_section is None else doc_child_sections(doc_id, parent_section.get("section_id"))
        if not children:
            break
        chosen, stop, reason, step_queries, route_plan = choose_child_sections_with_llm(
            state,
            task,
            doc_id,
            slot,
            query,
            parent_section,
            children,
            failed_titles=failed_titles,
            used_queries=used_queries,
        )
        planned_queries.extend(step_queries)
        route_steps.append(
            {
                "depth": depth + 1,
                "parent_section_id": parent_section.get("section_id") if parent_section else None,
                "parent_title": section_title(parent_section) if parent_section else None,
                "candidates": [sec.get("section_id") for sec in children[:24]],
                "selected_section_ids": [sec.get("section_id") for sec in chosen],
                "selected_titles": [section_title(sec) for sec in chosen],
                "planned_queries": step_queries,
                "information_need": route_plan.get("information_need"),
                "candidate_sections": route_plan.get("candidate_sections"),
                "retry_strategy": route_plan.get("retry_strategy"),
                "stop": stop,
                "reason": reason,
            }
        )
        if stop or not chosen:
            break
        first = chosen[0]
        grand_children = doc_child_sections(doc_id, first.get("section_id"))
        parent_section = first
        if not grand_children:
            break
    planned_queries = list(dict.fromkeys(query for query in planned_queries if query))[:3]
    if not planned_queries:
        planned_queries = [query]
    return chosen, route_steps, planned_queries


def retrieve_with_sections(
    query: str,
    doc_id: str,
    sections: list[dict],
    source: str,
    route_steps: list[dict],
    top_k: int = FOLLOWUP_TOP_K,
) -> list[dict]:
    if not sections:
        return []
    allowed_page_ids = {
        page_id
        for sec in sections
        for page_id in (sec.get("descendant_page_ids") or sec.get("page_ids") or [])
    }
    hits = [
        (score, page)
        for score, page in BM25_INDEX.score(query, allowed_doc_ids={doc_id})
        if page.get("page_id") in allowed_page_ids
    ][:top_k]
    section_names = [section_title(sec) for sec in sections]
    section_ids = [sec.get("section_id") for sec in sections]
    evidence = [page_to_evidence(score, page, source + "_section") for score, page in hits]
    for item in evidence:
        item["selected_section_ids"] = section_ids
        item["selected_sections"] = section_names
        item["section_route_steps"] = route_steps
    context = section_context_evidence(sections, source + "_section", section_ids, section_names)
    for item in context:
        item["section_route_steps"] = route_steps
    combined = merge_evidence(evidence, context)
    pointer_expanded = expand_title_pointer_evidence(combined, source + "_section")
    for item in pointer_expanded:
        item["section_route_steps"] = route_steps
    return merge_evidence(combined, pointer_expanded)


def collect_route_steps_from_evidence(evidence: list[dict]) -> list[dict]:
    seen: set[str] = set()
    steps: list[dict] = []
    for ev in evidence:
        for step in ev.get("section_route_steps") or []:
            key = json.dumps(
                [
                    step.get("depth"),
                    step.get("parent_section_id"),
                    step.get("selected_section_ids"),
                    step.get("planned_queries"),
                ],
                ensure_ascii=False,
                sort_keys=True,
            )
            if key in seen:
                continue
            seen.add(key)
            steps.append(
                {
                    "depth": step.get("depth"),
                    "parent_title": step.get("parent_title"),
                    "selected_titles": step.get("selected_titles"),
                    "planned_queries": step.get("planned_queries"),
                    "information_need": step.get("information_need"),
                    "retry_strategy": step.get("retry_strategy"),
                    "stop": step.get("stop"),
                    "reason": step.get("reason"),
                }
            )
    return steps


def human_toc_reasoning_node(state: AgentState) -> AgentState:
    system, user = build_task_reasoning_plan_prompt(state)
    if DRY_RUN_WITHOUT_LLM:
        plan = dry_json(system, user, "router")
    else:
        plan, usage = call_qwen_json(system, user)
        add_usage(state, usage)
    tasks = normalize_task_reasoning_plan(plan, state.get("doc_ids", []), state.get("doc_aliases", {}))
    task_results: list[dict] = []
    all_memory: list[dict] = []
    task_debug: dict = {"plan": plan, "tasks": [], "path": "human_toc"}
    add_trace(state, f"human toc plan tasks={len(tasks)}")

    for task_index, task in enumerate(tasks, start=1):
        task_memory = list(task.get("known_facts") or [])
        task_evidence: list[dict] = []
        task_log: dict = {"task_index": task_index, "task": task, "section_plans": [], "rounds": []}
        task_type = str(task.get("task_type") or "retrieve").strip().lower()
        if task_type == "summarize":
            task_result = {
                "task": task.get("task", ""),
                "status": "complete",
                "result": "",
                "basis": "汇总型任务，不单独检索文档，由最终作答阶段基于前序任务结果完成。",
                "evidence_ids": [],
                "missing_slots": [],
            }
            task_results.append(task_result)
            all_memory.extend(task_memory)
            task_log["result"] = task_result
            task_log["summary_only"] = True
            task_debug["tasks"].append(task_log)
            continue
        used_queries: list[str] = []
        failed_titles: list[str] = []
        failure_notes: list[str] = []
        section_candidate_queue: list[dict] = []
        for doc_id in task.get("target_doc_ids") or state.get("doc_ids", []):
            section_plan = plan_section_queue_with_llm(state, task, doc_id, used_queries=used_queries)
            task_log["section_plans"].append(
                {
                    "doc_id": doc_id,
                    "task_type": task_type,
                    "information_need": section_plan.get("information_need"),
                    "candidates": [
                        {
                            "section_id": item.get("section_id"),
                            "title": item.get("title"),
                            "why": item.get("why"),
                            "search_queries": item.get("search_queries"),
                        }
                        for item in section_plan.get("candidates") or []
                    ],
                }
            )
            for candidate in section_plan.get("candidates") or []:
                section_candidate_queue.append({"doc_id": doc_id, "plan": section_plan, "candidate": candidate})
        next_candidate_index = 0
        if section_candidate_queue:
            item = section_candidate_queue[next_candidate_index]
            next_candidate_index += 1
            for query in item["candidate"].get("search_queries") or []:
                if query not in used_queries:
                    used_queries.append(query)
            task_evidence = merge_evidence(
                task_evidence,
                retrieve_section_candidate(
                    item["candidate"],
                    item["doc_id"],
                    "human_toc",
                    item["plan"],
                    next_candidate_index,
                ),
            )
        else:
            for search in task.get("search_slots") or []:
                query = maybe_fix_mojibake(str(search.get("query") or search.get("slot") or "")).strip()
                slot = maybe_fix_mojibake(str(search.get("slot") or "")).strip()
                for doc_id in task.get("target_doc_ids") or state.get("doc_ids", []):
                    sections, route_steps, planned_queries = route_sections_human(
                        state,
                        task,
                        doc_id,
                        slot,
                        query,
                        used_queries=used_queries,
                    )
                    for planned_query in planned_queries:
                        if planned_query in used_queries:
                            continue
                        used_queries.append(planned_query)
                        evidence = retrieve_with_sections(planned_query, doc_id, sections, "human_toc", route_steps)
                        task_evidence = merge_evidence(task_evidence, evidence)
        audit: dict = {}
        for round_index in range(1, MAX_REASONING_AUDIT_ROUNDS + 1):
            system, user = build_task_reasoning_audit_prompt(state, task, task_memory, task_evidence)
            if DRY_RUN_WITHOUT_LLM:
                audit = dry_json(system, user, "audit")
            else:
                audit, usage = call_qwen_json(system, user)
                add_usage(state, usage)
            raw_filled_slots = audit.get("filled_slots") or []
            filtered_filled_slots = filter_task_audit_slots(raw_filled_slots)
            task_memory = merge_memory_slots(task_memory, filtered_filled_slots)
            round_log = {
                "round": round_index,
                "audit": audit,
                "filtered_filled_slots": filtered_filled_slots,
                "dropped_filled_slots": [
                    item for item in raw_filled_slots if item not in filtered_filled_slots
                ],
                "memory_slots": task_memory,
                "route_steps": collect_route_steps_from_evidence(task_evidence),
                "evidence": task_evidence,
            }
            task_log["rounds"].append(round_log)
            if audit.get("can_solve"):
                break
            if round_index >= MAX_REASONING_AUDIT_ROUNDS:
                break
            if audit.get("expand_evidence_ids"):
                expanded = expand_forward_evidence_ids(task_evidence, audit.get("expand_evidence_ids") or [], forward=4)
                task_evidence = merge_evidence(task_evidence, expanded)
                round_log["anchor_expanded"] = expanded
            missing = []
            for item in audit.get("missing_slots") or []:
                if not isinstance(item, dict):
                    continue
                missing.append(
                    {
                        "slot": maybe_fix_mojibake(str(item.get("slot") or "")).strip(),
                        "target_doc_ids": [doc_id for doc_id in item.get("target_doc_ids", []) if doc_id in state.get("doc_ids", [])] or task.get("target_doc_ids", []),
                        "query": maybe_fix_mojibake(str(item.get("followup_query") or item.get("query") or item.get("slot") or "")).strip(),
                    }
                )
            if not missing:
                break
            followup_evidence: list[dict] = []
            failure_analysis = audit.get("failure_analysis") if isinstance(audit.get("failure_analysis"), dict) else {}
            if failure_analysis:
                note = "；".join(
                    part
                    for part in [
                        maybe_fix_mojibake(str(failure_analysis.get("reason") or "")).strip(),
                        maybe_fix_mojibake(str(failure_analysis.get("next_action") or "")).strip(),
                    ]
                    if part
                )
                if note and note not in failure_notes:
                    failure_notes.append(note)
                for title in failure_analysis.get("avoid_titles") or []:
                    title = maybe_fix_mojibake(str(title or "")).strip()
                    if title and title not in failed_titles:
                        failed_titles.append(title)
            for evidence_item in task_evidence[-12:]:
                title = " > ".join(evidence_item.get("heading_path") or evidence_item.get("selected_sections") or [])
                if title and title not in failed_titles:
                    failed_titles.append(title)
            tried_candidates: list[dict] = []
            directed_candidates: list[dict] = []
            for missing_item in missing[:1]:
                for doc_id in missing_item["target_doc_ids"][:1]:
                    directed_query = missing_item.get("query") or missing_item.get("slot") or ""
                    directed_plan = plan_section_queue_with_llm(
                        state,
                        task,
                        doc_id,
                        failed_titles=(failure_notes + failed_titles)[:8],
                        used_queries=used_queries + [directed_query],
                        max_sections=3,
                        focus_slot=missing_item.get("slot", ""),
                        focus_query=directed_query,
                    )
                    task_log["section_plans"].append(
                        {
                            "doc_id": doc_id,
                            "task_type": task_type,
                            "information_need": directed_plan.get("information_need"),
                            "reason": "directed_replan_for_missing_slot",
                            "missing_slot": missing_item.get("slot"),
                            "missing_query": directed_query,
                            "candidates": [
                                {
                                    "section_id": candidate.get("section_id"),
                                    "title": candidate.get("title"),
                                    "why": candidate.get("why"),
                                    "search_queries": candidate.get("search_queries"),
                                }
                                for candidate in directed_plan.get("candidates") or []
                            ],
                        }
                    )
                    for candidate in directed_plan.get("candidates") or []:
                        directed_candidates.append({"doc_id": doc_id, "plan": directed_plan, "candidate": candidate})
            if directed_candidates:
                item = directed_candidates[0]
                for query in item["candidate"].get("search_queries") or []:
                    if query not in used_queries:
                        used_queries.append(query)
                tried_candidates.append(
                    {
                        "doc_id": item["doc_id"],
                        "section_id": item["candidate"].get("section_id"),
                        "title": item["candidate"].get("title"),
                        "search_queries": item["candidate"].get("search_queries"),
                        "reason": "directed_missing_slot",
                    }
                )
                followup_evidence = merge_evidence(
                    followup_evidence,
                    retrieve_section_candidate(
                        item["candidate"],
                        item["doc_id"],
                        "human_toc_followup",
                        item["plan"],
                        next_candidate_index + 1,
                    ),
                )
            elif next_candidate_index < len(section_candidate_queue):
                item = section_candidate_queue[next_candidate_index]
                next_candidate_index += 1
                for query in item["candidate"].get("search_queries") or []:
                    if query not in used_queries:
                        used_queries.append(query)
                tried_candidates.append(
                    {
                        "doc_id": item["doc_id"],
                        "section_id": item["candidate"].get("section_id"),
                        "title": item["candidate"].get("title"),
                        "search_queries": item["candidate"].get("search_queries"),
                        "reason": "next_initial_candidate",
                    }
                )
                followup_evidence = merge_evidence(
                    followup_evidence,
                    retrieve_section_candidate(
                        item["candidate"],
                        item["doc_id"],
                        "human_toc_followup",
                        item["plan"],
                        next_candidate_index,
                    ),
                )
            else:
                for missing_item in missing[:1]:
                    for doc_id in missing_item["target_doc_ids"][:1]:
                        rewritten_queries = rewrite_followup_queries(state, task, missing_item, task_evidence)[:2]
                        fallback_plan = plan_section_queue_with_llm(
                            state,
                            task,
                            doc_id,
                            failed_titles=(failure_notes + failed_titles)[:8],
                            used_queries=used_queries + rewritten_queries,
                            max_sections=3,
                            focus_slot=missing_item.get("slot", ""),
                            focus_query=(rewritten_queries[0] if rewritten_queries else missing_item.get("query", "")),
                        )
                        task_log["section_plans"].append(
                            {
                                "doc_id": doc_id,
                                "task_type": task_type,
                                "information_need": fallback_plan.get("information_need"),
                                "reason": "fallback_replan_after_queue_exhausted",
                                "candidates": [
                                    {
                                        "section_id": candidate.get("section_id"),
                                        "title": candidate.get("title"),
                                        "why": candidate.get("why"),
                                        "search_queries": candidate.get("search_queries"),
                                    }
                                    for candidate in fallback_plan.get("candidates") or []
                                ],
                            }
                        )
                        for candidate in fallback_plan.get("candidates") or []:
                            section_candidate_queue.append({"doc_id": doc_id, "plan": fallback_plan, "candidate": candidate})
                        if next_candidate_index < len(section_candidate_queue):
                            item = section_candidate_queue[next_candidate_index]
                            next_candidate_index += 1
                            for query in item["candidate"].get("search_queries") or []:
                                if query not in used_queries:
                                    used_queries.append(query)
                            tried_candidates.append(
                                {
                                    "doc_id": item["doc_id"],
                                    "section_id": item["candidate"].get("section_id"),
                                    "title": item["candidate"].get("title"),
                                    "search_queries": item["candidate"].get("search_queries"),
                                }
                            )
                            followup_evidence = merge_evidence(
                                followup_evidence,
                                retrieve_section_candidate(
                                    item["candidate"],
                                    item["doc_id"],
                                    "human_toc_followup",
                                    item["plan"],
                                    next_candidate_index,
                                ),
                            )
                        missing_item["rewritten_queries"] = rewritten_queries
                        missing_item["failed_titles"] = failed_titles[:8]
                        missing_item["failure_notes"] = failure_notes[-4:]
                        missing_item["used_queries"] = used_queries[-8:]
            task_evidence = merge_evidence(task_evidence, followup_evidence)
            round_log["missing_slots_used"] = missing
            round_log["tried_candidates"] = tried_candidates
            round_log["followup_route_steps"] = collect_route_steps_from_evidence(followup_evidence)
            round_log["followup_evidence"] = followup_evidence
        if DRY_RUN_WITHOUT_LLM:
            system, user = build_task_reasoning_result_prompt(state, task, task_memory, audit)
            task_result = dry_json(system, user, "judge")
        else:
            system, user = build_task_reasoning_result_prompt(state, task, task_memory, audit)
            task_result, usage = call_qwen_json(system, user)
            add_usage(state, usage)
        if not task_result.get("task"):
            task_result["task"] = task.get("task", "")
        task_results.append(task_result)
        all_memory.extend(task_memory)
        task_log["result"] = task_result
        task_debug["tasks"].append(task_log)

    system, user = build_task_reasoning_final_prompt(state, task_results)
    if DRY_RUN_WITHOUT_LLM:
        output = dry_json(system, user, "judge")
    else:
        output, usage = call_qwen_json(system, user)
        add_usage(state, usage)
    answer = str(output.get("answer") or "").strip().upper()
    if answer not in "ABCD":
        answer = ""
    state["final_answer"] = answer
    state["status"] = "done"
    state["task_reasoning_plan"] = plan
    state["task_reasoning_tasks"] = tasks
    state["task_reasoning_results"] = task_results
    state["task_reasoning_debug"] = task_debug
    state["reasoning_memory_slots"] = all_memory
    state["reasoning_judgment"] = output
    add_trace(state, f"human toc final answer={answer}")
    return state


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("init", init_state)
    graph.add_node("human_toc_reasoning", human_toc_reasoning_node)
    graph.add_node("route_docs", route_docs_node)
    graph.add_node("retrieve_initial", retrieve_initial_node)
    graph.add_node("audit", audit_node)
    graph.add_node("retrieve_more", retrieve_more_node)
    graph.add_node("judge", judge_node)
    graph.add_node("advance_option", advance_option_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("init")
    graph.add_conditional_edges(
        "init",
        lambda state: "human_toc_reasoning" if use_task_reasoning_path(state["question"]) else "route_docs",
        {"human_toc_reasoning": "human_toc_reasoning", "route_docs": "route_docs"},
    )
    graph.add_edge("human_toc_reasoning", "finalize")
    graph.add_edge("route_docs", "retrieve_initial")
    graph.add_edge("retrieve_initial", "audit")
    graph.add_conditional_edges("audit", route_after_audit, {"retrieve_more": "retrieve_more", "judge": "judge"})
    graph.add_edge("retrieve_more", "audit")
    graph.add_edge("judge", "advance_option")
    graph.add_conditional_edges("advance_option", route_after_advance, {"route_docs": "route_docs", "finalize": "finalize"})
    graph.add_edge("finalize", END)
    return graph.compile()


def normalize_answer(answer: str, answer_format: str) -> str:
    letters = [char for char in answer.upper() if char in "ABCD"]
    if answer_format == "multi":
        return "".join(sorted(set(letters), key="ABCD".index))
    if answer_format in {"mcq", "tf"}:
        return letters[0] if letters else ""
    return "".join(sorted(set(letters), key="ABCD".index))


def upsert_answer_csv(state: AgentState) -> None:
    path = Path(ANSWER_CSV)
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


def write_outputs(state: AgentState) -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / f"langgraph_{state['qid']}_state.json"
    output_md = output_dir / f"langgraph_{state['qid']}.md"
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
                    for ev in round_log.get("evidence", []):
                        heading = " > ".join(ev.get("heading_path") or [])
                        text = re.sub(r"\s+", " ", ev.get("text", "")).strip()
                        lines.append(f"  - `{ev['doc_id']}` `{ev['page_id']}` score={ev['score']:.3f} source={ev['source']}")
                        lines.append(f"    - heading: {heading}")
                        if ev.get("selected_sections"):
                            lines.append(f"    - selected_sections: {json.dumps(ev.get('selected_sections'), ensure_ascii=False)}")
                        lines.append(f"    - text: {text[:400]}")
                if round_log.get("followup_evidence"):
                    lines.append("- followup_evidence:")
                    for ev in round_log.get("followup_evidence", []):
                        heading = " > ".join(ev.get("heading_path") or [])
                        text = re.sub(r"\s+", " ", ev.get("text", "")).strip()
                        lines.append(f"  - `{ev['doc_id']}` `{ev['page_id']}` score={ev['score']:.3f} source={ev['source']}")
                        lines.append(f"    - heading: {heading}")
                        if ev.get("selected_sections"):
                            lines.append(f"    - selected_sections: {json.dumps(ev.get('selected_sections'), ensure_ascii=False)}")
                        lines.append(f"    - text: {text[:400]}")
                if round_log.get("anchor_expanded"):
                    lines.append("- anchor_expanded:")
                    for ev in round_log.get("anchor_expanded", []):
                        heading = " > ".join(ev.get("heading_path") or [])
                        text = re.sub(r"\s+", " ", ev.get("text", "")).strip()
                        lines.append(f"  - `{ev['doc_id']}` `{ev['page_id']}` score={ev['score']:.3f} source={ev['source']}")
                        lines.append(f"    - heading: {heading}")
                        if ev.get("selected_sections"):
                            lines.append(f"    - selected_sections: {json.dumps(ev.get('selected_sections'), ensure_ascii=False)}")
                        lines.append(f"    - text: {text[:400]}")
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
        for key in ("router", "audit", "judgment"):
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


def main() -> None:
    questions = load_json(QUESTIONS_PATH)
    question = next(item for item in questions if item["qid"] == TARGET_QID)
    app = build_graph()
    final_state = app.invoke({"question": question})
    print(f"qid={final_state['qid']}")
    print(f"answer={final_state.get('final_answer', '')}")
    print(f"status={final_state.get('status')}")
    print(f"token_usage={json.dumps(final_state.get('token_usage', {}), ensure_ascii=False)}")
    if APPEND_ANSWER_CSV:
        upsert_answer_csv(final_state)
    if SAVE_DEBUG_OUTPUTS:
        write_outputs(final_state)


if __name__ == "__main__":
    main()
