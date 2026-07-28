from __future__ import annotations

import csv
import hashlib
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


DOMAIN_PAGE_INDEX_PATHS = {
    "financial_contracts": "processed/page_index_financial_contracts/page_index.jsonl",
    "financial_reports": "processed/page_index_financial_reports/page_index.jsonl",
    "insurance": "processed/page_index_insurance/page_index.jsonl",
    "regulatory": "processed/page_index_regulatory/page_index.jsonl",
    "research": "processed/page_index_research/page_index.jsonl",
}


def read_atomic_domain_for_startup(path: str, atomic_id: str) -> str:
    data_path = Path(path)
    if not data_path.exists():
        return ""
    try:
        if data_path.suffix.lower() == ".jsonl":
            with data_path.open("r", encoding="utf-8-sig") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if str(item.get("atomic_id") or item.get("qid") or "") == atomic_id:
                        return str(item.get("domain") or "").strip()
        else:
            data = json.loads(data_path.read_text(encoding="utf-8-sig"))
            rows = data if isinstance(data, list) else data.get("questions") or data.get("data") or data.get("items") or []
            for item in rows:
                if str(item.get("atomic_id") or item.get("qid") or "") == atomic_id:
                    return str(item.get("domain") or "").strip()
    except (OSError, json.JSONDecodeError, AttributeError):
        return ""
    return ""


TARGET_QID = os.environ.get("ATOMIC_ID", os.environ.get("TARGET_QID", "fin_b_007_A"))
ATOMIC_QUESTIONS_PATH = os.environ.get(
    "ATOMIC_QUESTIONS_PATH",
    "enterprise_qa_agent/data/atomic_questions/atomic_questions_all.jsonl",
)
QUESTIONS_PATH = ATOMIC_QUESTIONS_PATH
STARTUP_ATOMIC_DOMAIN = read_atomic_domain_for_startup(ATOMIC_QUESTIONS_PATH, TARGET_QID)
PAGE_INDEX_PATH = os.environ.get(
    "PAGE_INDEX_PATH",
    DOMAIN_PAGE_INDEX_PATHS.get(STARTUP_ATOMIC_DOMAIN, "processed/page_index_research/page_index.jsonl"),
)

ANSWER_CSV = os.environ.get("ENTERPRISE_RESULT_JSONL", "enterprise_qa_agent/outputs/results/enterprise_results.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "enterprise_qa_agent/outputs/debug")

INITIAL_TOP_K = 4
FOLLOWUP_TOP_K = 3
MAX_AUDIT_ROUNDS = 3
MAX_REASONING_AUDIT_ROUNDS = 3
DOC_REROUTE_TOP_K = 2
DOC_REROUTE_MIN_SCORE = 7.0
DOC_REROUTE_MARGIN = 0.6
REQUEST_TIMEOUT_SECONDS = 180
REQUEST_RETRY_TIMES = 3
VERBOSE_CONSOLE = True
PRINT_EVIDENCE_TEXT = True
EVIDENCE_TEXT_PREVIEW_CHARS = 700
COMPACT_HTML_EVIDENCE = True
SECTION_CONTEXT_PER_SECTION = 3
SECTION_CONTEXT_MAX_TOTAL = 6
SECTION_CONTEXT_MIN_CHARS = 20
INNER_FOCUS_MIN_CHARS = 1200
INNER_FOCUS_MAX_SNIPPETS = 3
INNER_FOCUS_SNIPPET_CHARS = 900
LOCAL_SUBSEARCH_MIN_CHARS = 1500
LOCAL_SUBSEARCH_MAX_SNIPPETS = 3
LOCAL_SUBSEARCH_SNIPPET_CHARS = 760
REVIEW_ORIGINAL_MAX_SOURCE_IDS = 3
INITIAL_SUMMARY_CANDIDATES_PER_DOC = 2
MAX_TOC_ROUTE_DEPTH = 4
MAX_TOC_CHILDREN = 24
SAVE_DEBUG_OUTPUTS = True
APPEND_ANSWER_CSV = False
DRY_RUN_WITHOUT_LLM = os.environ.get("DRY_RUN_WITHOUT_LLM", "0").lower() in {"1", "true", "yes"}
DASHSCOPE_API_KEY_ENV = "DEEPSEEK_API_KEY"
DASHSCOPE_API_KEY_FILE = "api_ds"
DASHSCOPE_BASE_URL = "https://api.deepseek.com"
# DASHSCOPE_API_KEY_ENV = "DASHSCOPE_API_KEY"
# DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# QWEN_MODEL = "qwen3.6-plus"
# QWEN_MODEL = "qwen3.5-plus-2026-04-20"


# 1. 鐩存帴鏄惧紡鍐欏叆鎮ㄧ殑璁鏄熻景 API Key锛堣鏇挎崲涓烘偍鐪熷疄鐨?Key锛?SPARKAI_API_KEY = "a24abac9ef09acd39638685937f0a999:NWM0ZTE1ODBlMmYyNDI4MzkxYmJlMmE3"
# 2. 璁鏄熻景 OpenAI 鍏煎妯″紡鐨?Base URL
SPARKAI_API_KEY_ENV = DASHSCOPE_API_KEY_ENV
SPARKAI_API_KEY_FILE = DASHSCOPE_API_KEY_FILE
SPARKAI_BASE_URL = DASHSCOPE_BASE_URL
SPARKAI_CHAT_URL = f"{SPARKAI_BASE_URL.rstrip('/')}/chat/completions"
# 3. 瑕佷娇鐢ㄧ殑妯″瀷鍚嶇О锛堜互 Qwen3.6 涓轰緥锛岃鏍规嵁骞冲彴瀹為檯鏀寔鐨勫悕绉板～鍐欙級
QWEN_MODEL = os.environ.get("QWEN_MODEL", "deepseek-chat")


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
    mojibake = tuple(chr(code) for code in (0x951B, 0x7ED7, 0x93C9, 0x95B2, 0x9429, 0x7039, 0x5F42, 0x20AC))
    mojibake_markers = mojibake + ("锛", "鈥", "鐨", "骞", "鏈", "鍙", "鎵", "涓", "瀹", "绗", "妯")
    if not any(marker in text for marker in mojibake_markers):
        return text
    try:
        fixed = text.encode("gb18030").decode("utf-8")
    except UnicodeError:
        return text
    common = ("\u7684", "\u7b2c", "\u516c\u53f8", "\u53d1\u884c", "\u503a\u5238", "\u4fe1\u606f", "\u62a5\u544a")
    fixed_score = sum(fixed.count(word) for word in common) * 3 - sum(fixed.count(word) for word in mojibake)
    text_score = sum(text.count(word) for word in common) * 3 - sum(text.count(word) for word in mojibake)
    return fixed if fixed_score > text_score else text


def load_json(path: str) -> Any:
    return json.loads(maybe_fix_mojibake(Path(path).read_text(encoding="utf-8-sig")))


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_records(path: str) -> list[dict]:
    if str(path).lower().endswith(".jsonl"):
        return load_jsonl(path)
    data = load_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("questions", "data", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        return [data]
    return []


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


LOW_VALUE_PAGE_TERMS = (
    "内容目录",
    "图表目录",
    "目录",
    "联系人",
    "联系方式",
    "公司基本数据",
    "评级说明",
    "免责声明",
    "法律声明",
)

SUMMARY_PAGE_TERMS = (
    "投资要点",
    "核心观点",
    "报告摘要",
    "摘要",
    "核心结论",
    "主要观点",
    "行业观点",
    "事件点评",
    "首页",
)


def evidence_surface_text(item: dict) -> str:
    return maybe_fix_mojibake(
        " ".join(
            [
                " > ".join(item.get("heading_path") or []),
                str(item.get("title") or ""),
                str(item.get("index_text") or ""),
                str(item.get("text") or ""),
            ]
        )
    )


def page_surface_text(page: dict) -> str:
    return maybe_fix_mojibake(
        " ".join(
            [
                " > ".join(page.get("heading_path") or []),
                str(page.get("title") or ""),
                str(page.get("index_text") or ""),
                str(page.get("text") or ""),
            ]
        )
    )


def is_low_value_evidence(item: dict) -> bool:
    heading = compact_text(" > ".join(item.get("heading_path") or []))
    text_head = compact_text(str(item.get("text") or "")[:180])
    surface = heading + " " + text_head
    if any(term in surface for term in LOW_VALUE_PAGE_TERMS):
        return True
    if text_head.count("................") >= 2:
        return True
    return False


def is_summary_like_page(page: dict) -> bool:
    surface = compact_text(page_surface_text(page)[:1200])
    if any(term in surface for term in SUMMARY_PAGE_TERMS):
        return True
    page_no = page.get("page_no")
    try:
        return int(page_no) <= 2 and not is_heading_only_page(page)
    except Exception:
        return False


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
    if is_low_value_evidence(item):
        score -= 12.0
    elif any(term in compact_text(surface[:1200]) for term in SUMMARY_PAGE_TERMS):
        score += 2.0
    return score


def rerank_initial_evidence(evidence: list[dict], query: str) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
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
    terms = normalize_key_terms(key_terms or [], " ".join([slot, query]), max_terms=12)
    score = evidence_query_score(item, query)
    covered = 0
    for term in terms:
        compact_term = compact_text(term)
        if not compact_term:
            continue
        if compact_term in compact_surface:
            score += 4.0
            covered += 1
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
    evidence: list[dict],
    slot: str,
    query: str,
    key_terms: list[str] | None = None,
    max_items: int | None = None,
) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for item in evidence:
        evidence_id = str(item.get("evidence_id") or "")
        if not evidence_id or evidence_id in seen:
            continue
        seen.add(evidence_id)
        unique.append(item)
    scored = [
        (evidence_slot_score(item, slot, query, key_terms), idx, item)
        for idx, item in enumerate(unique)
    ]
    scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
    ranked = [item for _score, _idx, item in scored]
    return ranked[:max_items] if max_items is not None else ranked


def summary_candidate_evidence(doc_ids: list[str], query: str, source: str = "initial_summary") -> list[dict]:
    candidates: list[dict] = []
    for doc_id in doc_ids:
        scored: list[tuple[float, dict]] = []
        for page in PAGES_BY_DOC.get(doc_id, [])[:8]:
            if not page or is_heading_only_page(page) or not is_summary_like_page(page):
                continue
            item = page_to_evidence(0.0, page, source)
            score = evidence_query_score(item, query)
            if score <= 0:
                continue
            scored.append((score, item))
        scored.sort(key=lambda row: row[0], reverse=True)
        candidates.extend(item for _score, item in scored[:INITIAL_SUMMARY_CANDIDATES_PER_DOC])
    return candidates


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


def section_pages_as_evidence_ranked(
    section: dict,
    query: str,
    source: str,
    selected_section_ids: list[str],
    selected_section_names: list[str],
    max_pages: int = 2,
) -> list[dict]:
    page_ids = section.get("descendant_page_ids") or section.get("page_ids") or []
    pages = [PAGE_BY_ID.get(page_id) for page_id in page_ids]
    pages = [page for page in pages if page and not is_heading_only_page(page)]
    if not pages:
        return []
    query_tokens = set(tokenize(query or section_title(section)))
    scored: list[tuple[float, dict]] = []
    for page in pages:
        heading_text = " > ".join(page.get("heading_path") or [])
        if any(term in compact_text(heading_text + " " + page.get("text", "")[:80]) for term in ("内容目录", "图表目录")):
            continue
        text = " ".join(
            [
                heading_text,
                page.get("index_text", ""),
                page.get("text", ""),
            ]
        )
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


def retrieve_by_key_term_coverage(
    doc_ids: list[str],
    key_terms: list[str],
    source: str,
    max_per_doc: int = 3,
) -> list[dict]:
    terms = [term for term in normalize_key_terms(key_terms or [], max_terms=12) if len(compact_text(term)) >= 2]
    if len(terms) < 2:
        return []
    evidence: list[dict] = []
    allowed = set(doc_ids)
    for doc_id in doc_ids:
        scored: list[tuple[float, dict]] = []
        for page in PAGES:
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


def expand_auto_anchor_evidence(
    evidence: list[dict],
    forward: int = 2,
    source_suffix: str = "_auto_anchor_expand",
) -> list[dict]:
    anchor_ids = [
        item.get("evidence_id")
        for item in evidence
        if item.get("evidence_id") and is_expandable_anchor_evidence(item)
    ]
    return expand_forward_evidence_ids(evidence, anchor_ids, forward=forward, source_suffix=source_suffix)


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


def merge_evidence(existing: list[dict], incoming: list[dict], excluded_ids: set[str] | None = None) -> list[dict]:
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


REFERENCE_TARGET_RE = re.compile(
    r"(?:详见|见|参见|载明于|以|按照|按)([^。；;\n]{0,40}?(?:附表\s*\d*|计划表|保险计划表|费率表|现金价值表|利益表|比例表|限额表|表格?))"
)


def extract_reference_targets_from_evidence(evidence: list[dict]) -> list[str]:
    targets: list[str] = []
    for item in evidence[-12:]:
        text = maybe_fix_mojibake(str(item.get("text") or ""))
        for match in REFERENCE_TARGET_RE.finditer(text):
            target = compact_text(match.group(1))
            target = re.sub(r"^(?:本合同|本主险合同|本计划|本保险|的|中|所附|对应的)+", "", target).strip()
            if any(term in target for term in ("疾病清单", "分级表", "分期表")):
                continue
            if target and target not in targets:
                targets.append(target[:40])
    return targets[:4]


def reference_followup_queries(state: AgentState, task: dict, missing_item: dict, evidence: list[dict]) -> list[str]:
    slot = maybe_fix_mojibake(str(missing_item.get("slot") or "")).strip()
    raw_query = maybe_fix_mojibake(str(missing_item.get("query") or missing_item.get("followup_query") or "")).strip()
    question = maybe_fix_mojibake(state["question"]["question"])
    task_text = maybe_fix_mojibake(str(task.get("task") or ""))
    targets = extract_reference_targets_from_evidence(evidence)
    if any(term in (slot + raw_query) for term in ("比例", "责任项", "赔付")):
        targets = [
            target for target in targets
            if any(term in target for term in ("计划表", "保险计划表", "比例表", "费率表", "限额表"))
        ] or targets[:1]
    if not targets:
        return []
    queries: list[str] = []
    for target in targets:
        queries.append(" ".join(part for part in [target, slot, raw_query] if part))
        queries.append(" ".join(part for part in [target, slot, task_text] if part))
    if any(term in (slot + raw_query) for term in ("比例", "责任项", "赔付")):
        queries.append(" ".join(part for part in ["计划表", "附表", slot, "计划一", "赔付比例"] if part))
    if "计划一" in question and "计划一" not in " ".join(queries):
        queries.append(" ".join(part for part in ["计划一", slot, raw_query] if part))
    return list(dict.fromkeys(query for query in queries if query))[:5]


def retrieve_reference_followup(
    state: AgentState,
    task: dict,
    missing_item: dict,
    evidence: list[dict],
    source: str,
) -> list[dict]:
    queries = reference_followup_queries(state, task, missing_item, evidence)
    if not queries:
        return []
    found: list[dict] = []
    targets = extract_reference_targets_from_evidence(evidence)
    for doc_id in missing_item.get("target_doc_ids") or task.get("target_doc_ids") or state.get("doc_ids", []):
        if doc_id not in state.get("doc_ids", []):
            continue
        for query in queries[:2]:
            hits = retrieve_one_doc(query, doc_id, 2, source + "_reference")
            for item in hits:
                item["reference_queries"] = queries
                item["reference_targets"] = targets
            found = merge_evidence(found, hits)
    return found[:4]


def normalize_reference_slot(text: str) -> str:
    fixed = maybe_fix_mojibake(str(text or ""))
    fixed = fixed.replace("Ａ", "A").replace("Ｂ", "B").replace("Ｃ", "C").replace("Ｄ", "D")
    return re.sub(r"[\s:：,，。；;()（）]+", "", fixed)


def extract_referenced_table_slots(evidence: list[dict]) -> list[dict]:
    slots: list[dict] = []
    for item in evidence[-12:]:
        text = maybe_fix_mojibake(str(item.get("text") or ""))
        for match in re.finditer(r"(?:详见|见|参见)([^。；;\n]{0,30}?(?:附表\s*\d*|计划表|保险计划表|费率表|现金价值表|利益表|比例表|限额表|表格?))", text):
            target = compact_text(match.group(1))
            if any(term in target for term in ("疾病清单", "分级表", "分期表")):
                continue
            prefix = text[max(0, match.start() - 80):match.start()]
            candidates = re.findall(r"[\u4e00-\u9fffA-Za-z]{1,18}(?:比例|费率|金额|限额|系数|标准)\s*[A-ZＡ-Ｚ]?", prefix)
            if not candidates:
                continue
            slot = compact_text(candidates[-1])
            if (
                not re.search(r"[A-ZＡ-Ｚ]\s*$", slot)
                and "比例" not in slot
                and not any(term in target for term in ("现金价值表", "利益表", "费率表"))
            ):
                continue
            normalized = normalize_reference_slot(slot)
            if not normalized:
                continue
            record = {
                "slot": normalized,
                "target": target,
                "evidence_id": item.get("evidence_id"),
                "doc_id": item.get("doc_id"),
            }
            if not any(existing.get("slot") == record["slot"] and existing.get("target") == record["target"] for existing in slots):
                slots.append(record)
    return slots[:4]


def evidence_has_concrete_reference_slot(evidence: list[dict], slot: str) -> bool:
    slot_key = normalize_reference_slot(slot)
    if not slot_key:
        return False
    short_slot_key = re.sub(r"^(?:各项|对应的|责任对应的)+", "", slot_key)
    for item in evidence:
        text = maybe_fix_mojibake(str(item.get("text") or ""))
        compact = normalize_reference_slot(text)
        if slot_key not in compact and short_slot_key not in compact:
            continue
        for key in (slot_key, short_slot_key):
            if not key:
                continue
            for match in re.finditer(re.escape(key), compact):
                tail = compact[match.end():match.end() + 60]
                if re.search(r"(?:详见|参见|见)[^。；;\n]{0,30}(?:表|附表|计划表)", tail):
                    continue
                value_match = re.search(r"[:：]?(?:[^。；;\n]{0,30})?(?:\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?\s*(?:元|万元|倍))", tail)
                if not value_match:
                    continue
                before_value = tail[:value_match.start()]
                if any(term in before_value for term in ("详见", "参见", "未给出", "未知", "需从", "需要")):
                    continue
                return True
    return False


def filled_slots_have_concrete_reference_slot(filled_slots: list[dict], slot: str) -> bool:
    slot_key = normalize_reference_slot(slot)
    if not slot_key:
        return False
    for item in filled_slots or []:
        if not isinstance(item, dict):
            continue
        raw_value = maybe_fix_mojibake(str(item.get("value") or ""))
        if any(term in raw_value for term in ("假设", "未给出", "未知", "需从", "需要", "详见", "参见")):
            continue
        combined = normalize_reference_slot(str(item.get("slot") or "") + raw_value)
        evidence_ids = [evidence_id for evidence_id in item.get("evidence_ids") or [] if evidence_id]
        if slot_key in combined and evidence_ids and re.search(r"\d+(?:\.\d+)?(?:%|元|万元|倍)", combined):
            return True
    return False


def enforce_reference_missing_slots(audit: dict, evidence: list[dict], task: dict) -> dict:
    referenced_slots = extract_referenced_table_slots(evidence)
    if not referenced_slots:
        return audit
    filled_slots = audit.get("filled_slots") or []
    missing_slots = audit.get("missing_slots") or []
    existing_missing = {
        normalize_reference_slot(str(item.get("slot") if isinstance(item, dict) else item))
        for item in missing_slots
    }
    for ref in referenced_slots:
        slot = ref.get("slot") or ""
        if not slot or slot in existing_missing:
            continue
        if evidence_has_concrete_reference_slot(evidence, slot):
            continue
        if filled_slots_have_concrete_reference_slot(filled_slots, slot):
            continue
        missing_slots.append(
            {
                "slot": slot,
                "target_doc_ids": task.get("target_doc_ids", []),
                "followup_query": " ".join(part for part in [ref.get("target"), slot] if part),
                "reference_target": ref.get("target"),
                "source_evidence_id": ref.get("evidence_id"),
            }
        )
        existing_missing.add(slot)
    if missing_slots:
        audit["missing_slots"] = missing_slots
        audit["can_solve"] = False
    return audit


ALLOCATION_METHOD_TERMS = ("按比例", "按占比", "占比例", "占比", "分摊", "分配", "应占比例")
ALLOCATION_SUPPORT_TERMS = ("按比例", "按占比", "占比例", "占比", "分摊", "分配", "依比例", "比例分配")


def result_contains_unsupported_assumption(
    task_result: dict,
    task: dict | None = None,
    task_memory: list[dict] | None = None,
    evidence: list[dict] | None = None,
    question_text: str = "",
) -> bool:
    text = maybe_fix_mojibake(
        " ".join(
            str(task_result.get(key) or "")
            for key in ("formula", "substitution", "calculation", "basis", "result")
        )
    )
    if any(term in text for term in ALLOCATION_METHOD_TERMS):
        support_parts = [question_text, json.dumps(task or {}, ensure_ascii=False)]
        support_parts.extend(json.dumps(item, ensure_ascii=False) for item in task_memory or [])
        support_parts.extend(str(item.get("text") or "") for item in evidence or [] if isinstance(item, dict))
        support_text = maybe_fix_mojibake(" ".join(support_parts))
        if not any(term in support_text for term in ALLOCATION_SUPPORT_TERMS):
            return True
    if not any(term in text for term in ("假设", "未给出", "未知", "需从", "需要补充")):
        return False
    return any(term in text for term in ("比例", "费率", "金额", "赔付", "给付", "责任项"))


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


EVIDENCE_NOTE_TERMS = (
    "保险责任",
    "责任免除",
    "不负责赔偿",
    "不承担",
    "不予赔偿",
    "赔付",
    "赔偿",
    "给付",
    "保险金",
    "免赔",
    "比例",
    "限额",
    "公式",
    "计算",
    "现金价值",
    "退保",
    "费用",
    "门诊",
    "住院",
    "详见",
    "如下",
    "包括",
    "不包括",
)


def task_note_query_text(state: AgentState, task: dict) -> str:
    parts = [maybe_fix_mojibake(state["question"]["question"]), str(task.get("task") or "")]
    for item in task.get("known_facts") or []:
        if isinstance(item, dict):
            parts.append(str(item.get("slot") or ""))
            parts.append(str(item.get("value") or ""))
    for item in task.get("search_slots") or []:
        if isinstance(item, dict):
            parts.append(str(item.get("slot") or ""))
            parts.append(str(item.get("query") or ""))
    return maybe_fix_mojibake(" ".join(part for part in parts if part))


def split_note_sentences(text: str) -> list[str]:
    text = compact_html_text(text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    raw_parts = re.split(r"(?<=[。；;！？!?])\s*|\n+", text)
    parts: list[str] = []
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        if len(part) <= 260:
            parts.append(part)
            continue
        for chunk in re.split(r"(?<=</tr>)|(?<=</td>)|(?<=[，,])", part):
            chunk = chunk.strip()
            if chunk:
                parts.append(chunk[:260])
    return parts


def evidence_note_score(sentence: str, query_tokens: set[str], query_text: str) -> float:
    sentence_fixed = maybe_fix_mojibake(sentence)
    sentence_tokens = set(tokenize(sentence_fixed))
    score = float(len(query_tokens & sentence_tokens) * 2)
    for term in EVIDENCE_NOTE_TERMS:
        if term in sentence_fixed:
            score += 3.0
    for year in re.findall(r"(?:19|20)\d{2}", query_text):
        if year in sentence_fixed:
            score += 2.0
    for number in re.findall(r"\d+(?:\.\d+)?\s*%?", query_text):
        if number and number.replace(" ", "") in sentence_fixed.replace(" ", ""):
            score += 1.0
    if len(sentence_fixed) < 12:
        score -= 2.0
    return score


def extract_evidence_notes_for_item(state: AgentState, task: dict, item: dict, max_notes: int = 1) -> list[dict]:
    query_text = task_note_query_text(state, task)
    query_tokens = set(tokenize(query_text))
    heading = " > ".join(item.get("heading_path") or [])
    sentences = split_note_sentences(item.get("text", ""))
    if not sentences:
        return []
    scored: list[tuple[float, int, str]] = []
    for idx, sentence in enumerate(sentences):
        score = evidence_note_score(sentence, query_tokens, query_text)
        if score > 0:
            scored.append((score, idx, sentence))
    if not scored and len(compact_text(item.get("text", ""))) <= 320:
        scored.append((1.0, 0, compact_text(item.get("text", ""))))
    scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
    notes: list[dict] = []
    seen_text: set[str] = set()
    for score, _idx, sentence in scored:
        note_text = compact_text(sentence)[:260]
        if not note_text or note_text in seen_text:
            continue
        seen_text.add(note_text)
        notes.append(
            {
                "source_evidence_id": item.get("evidence_id"),
                "doc_id": item.get("doc_id"),
                "heading": heading,
                "note": note_text,
                "score": round(score, 3),
            }
        )
        if len(notes) >= max_notes:
            break
    return notes


def update_task_evidence_notes(
    state: AgentState,
    task: dict,
    evidence: list[dict],
    notes: list[dict],
    noted_ids: set[str],
    max_total: int = 10,
) -> list[dict]:
    new_notes: list[dict] = []
    for item in evidence:
        evidence_id = str(item.get("evidence_id") or "")
        if not evidence_id or evidence_id in noted_ids:
            continue
        noted_ids.add(evidence_id)
        new_notes.extend(extract_evidence_notes_for_item(state, task, item))
    merged = notes + new_notes
    if len(merged) > max_total:
        merged = merged[-max_total:]
    return merged


def format_evidence_notes(notes: list[dict], max_chars: int = 1800) -> str:
    if not notes:
        return "[]"
    lines: list[str] = []
    used = 0
    for note in notes:
        text = compact_text(str(note.get("note") or ""))
        if not text:
            continue
        line = (
            f"- [{note.get('source_evidence_id')}] doc={note.get('doc_id')} "
            f"heading={note.get('heading')}: {text}"
        )
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines) if lines else "[]"


def print_evidence_preview(evidence: list[dict]) -> None:
    for ev in evidence:
        heading = " > ".join(ev.get("heading_path") or [])
        print(f"- {ev['doc_id']} {ev['page_id']} score={ev['score']:.3f} heading={heading}", flush=True)
        if PRINT_EVIDENCE_TEXT:
            text = re.sub(r"\s+", " ", ev.get("text", "")).strip()
            preview = text[:EVIDENCE_TEXT_PREVIEW_CHARS].encode("gbk", errors="replace").decode("gbk")
            print(f"  text: {preview}", flush=True)


def missing_query_text(missing_slots: list) -> str:
    parts: list[str] = []
    for item in missing_slots or []:
        if isinstance(item, dict):
            parts.extend(str(item.get(key) or "") for key in ("slot", "followup_query", "query"))
        else:
            parts.append(str(item))
    return maybe_fix_mojibake(" ".join(part for part in parts if part)).strip()


RATIO_SLOT_TERMS = ("占比", "比例", "比重", "强度", "率")
FINANCIAL_METRIC_TERMS = (
    "经营活动产生的现金流量净额",
    "归属于上市公司股东的净利润",
    "现金流量净额",
    "营业收入",
    "研发投入",
    "研发费用",
    "现金分红",
    "归母净利润",
    "净利润",
    "营业成本",
    "总资产",
    "净资产",
    "收入",
    "利润",
    "投入",
    "费用",
    "成本",
    "资产",
    "负债",
)


def computed_ratio_followup_queries(query: str, max_queries: int = 3) -> list[str]:
    text = maybe_fix_mojibake(query or "")
    if not text or not any(term in text for term in RATIO_SLOT_TERMS):
        return []
    cleaned = re.sub(r"\d+(?:\.\d+)?%", " ", text)
    found: list[str] = []
    for term in FINANCIAL_METRIC_TERMS:
        if term in cleaned and term not in found:
            found.append(term)
    expanded: list[str] = []
    for metric in found:
        if metric in ("营业收入", "收入"):
            candidate = "营业收入 合计"
        elif metric in ("研发投入", "研发费用", "投入", "费用"):
            candidate = f"{metric} 金额"
        elif metric in ("净利润", "归母净利润", "归属于上市公司股东的净利润", "利润"):
            candidate = f"{metric} 金额"
        elif metric in ("经营活动产生的现金流量净额", "现金流量净额", "现金流"):
            candidate = f"{metric} 金额"
        else:
            candidate = metric
        if candidate not in expanded:
            expanded.append(candidate)
    return expanded[:max_queries]


def ratio_metric_terms_from_query(query: str) -> list[str]:
    text = maybe_fix_mojibake(query or "")
    found = [term for term in FINANCIAL_METRIC_TERMS if term in text]
    found.sort(key=len, reverse=True)
    kept: list[str] = []
    for term in found:
        if any(term != other and term in other for other in kept):
            continue
        kept.append(term)
    return kept[:4]


def order_ratio_metrics_from_query(query: str, metrics: list[str]) -> list[str]:
    text = maybe_fix_mojibake(query or "")
    if len(metrics) < 2:
        return metrics
    marker_positions = [pos for marker in ("占", "占比", "占营业", "占收入") for pos in [text.find(marker)] if pos >= 0]
    if not marker_positions:
        return metrics
    marker_pos = min(marker_positions)
    before = [metric for metric in metrics if text.rfind(metric, 0, marker_pos) >= 0]
    after = [metric for metric in metrics if text.find(metric, marker_pos) >= 0]
    if before and after:
        numerator = max(before, key=lambda metric: text.rfind(metric, 0, marker_pos))
        denominator = min(after, key=lambda metric: text.find(metric, marker_pos))
        ordered = [numerator, denominator]
        ordered.extend(metric for metric in metrics if metric not in ordered)
        return ordered
    return metrics


def metric_number_snippet(text: str, metric: str, max_chars: int = 280) -> str:
    normalized = compact_html_text(maybe_fix_mojibake(text or ""))
    if not normalized or metric not in normalized:
        return ""
    number_pattern = r"\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|亿元|百万元|千元|元)?"
    best = ""
    for match in re.finditer(re.escape(metric), normalized):
        start = max(0, match.start() - 120)
        end = min(len(normalized), match.end() + 220)
        snippet = normalized[start:end].strip()
        if re.search(number_pattern, snippet):
            best = snippet
            break
    if not best:
        return ""
    best = re.sub(r"\s+", " ", best)
    return trim_snippet_around_query(best, metric, max_chars=max_chars)


def parse_metric_amount_from_snippet(snippet: str, metric: str) -> tuple[float | None, str]:
    text = compact_html_text(maybe_fix_mojibake(snippet or ""))
    if not text:
        return None, ""
    pos = text.find(metric)
    if pos < 0:
        pos = 0
    local = text[pos : pos + 220]
    number_re = re.compile(r"(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*(亿元|百万元|千元|元)?")
    for match in number_re.finditer(local):
        raw = match.group(1)
        unit = match.group(2) or ""
        if "%" in local[match.start() : match.end() + 1]:
            continue
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        if value == 0:
            continue
        multiplier = 1.0
        if unit == "亿元":
            multiplier = 100000000.0
        elif unit == "百万元":
            multiplier = 1000000.0
        elif unit == "千元":
            multiplier = 1000.0
        return value * multiplier, (raw + unit)
    return None, ""


def computed_ratio_line(metric_values: list[tuple[str, float, str]]) -> str:
    if len(metric_values) < 2:
        return ""
    numerator = metric_values[0]
    denominator = metric_values[1]
    if denominator[1] == 0:
        return ""
    ratio = numerator[1] / denominator[1] * 100
    return (
        "候选计算："
        f"{numerator[0]}({numerator[2]}) / {denominator[0]}({denominator[2]}) = {ratio:.2f}%"
    )


def build_computed_ratio_hint_evidence(option_state: dict, query: str, source_evidence: list[dict]) -> list[dict]:
    if not any(term in maybe_fix_mojibake(query or "") for term in RATIO_SLOT_TERMS):
        return []
    metrics = ratio_metric_terms_from_query(query)
    if len(metrics) < 2:
        return []
    metrics = order_ratio_metrics_from_query(query, metrics)
    lines: list[str] = []
    metric_values: list[tuple[str, float, str]] = []
    evidence_ids: list[str] = []
    for metric in metrics:
        for item in source_evidence:
            snippet = metric_number_snippet(str(item.get("text") or ""), metric)
            if not snippet:
                continue
            lines.append(f"{metric}: {snippet}")
            amount, raw_amount = parse_metric_amount_from_snippet(snippet, metric)
            if amount is not None and raw_amount:
                metric_values.append((metric, amount, raw_amount))
            evidence_id = str(item.get("evidence_id") or "")
            if evidence_id and evidence_id not in evidence_ids:
                evidence_ids.append(evidence_id)
            break
    if len(lines) < 2:
        return []
    ratio_line = computed_ratio_line(metric_values)
    if ratio_line:
        lines.append(ratio_line)
    digest = hashlib.md5(("|".join(lines) + "|" + query).encode("utf-8", errors="ignore")).hexdigest()[:10]
    doc_ids = [str(item.get("doc_id") or "") for item in source_evidence if item.get("doc_id")]
    doc_id = doc_ids[0] if doc_ids else (option_state.get("target_doc_ids") or [""])[0]
    return [
        {
            "evidence_id": f"computed_ratio_hint_{digest}",
            "doc_id": doc_id,
            "page_id": f"computed_ratio_hint_{digest}",
            "heading_path": ["computed ratio evidence"],
            "text": "比例/占比类槽位的候选原始数值：\n" + "\n".join(lines),
            "score": 0.0,
            "source": "computed_ratio_hint",
            "supporting_evidence_ids": evidence_ids,
        }
    ]


def query_years(query: str) -> set[str]:
    return set(re.findall(r"(?:19|20)\d{2}", maybe_fix_mojibake(query or "")))


def trim_snippet_around_query(snippet: str, query: str, max_chars: int = INNER_FOCUS_SNIPPET_CHARS) -> str:
    snippet = snippet.strip()
    if len(snippet) <= max_chars:
        return snippet
    query_tokens = [token for token in tokenize(query) if len(token) >= 2]
    years = query_years(query)
    anchors = list(years) + query_tokens
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


def focus_prefix_text(prefix_text: str, query: str, max_chars: int = 420) -> str:
    prefix_text = compact_text(prefix_text)
    if not prefix_text:
        return ""
    if len(prefix_text) <= max_chars:
        return prefix_text
    return trim_snippet_around_query(prefix_text, query, max_chars=max_chars)


def focus_markdown_table_snippet(snippet: str, query: str, max_chars: int = INNER_FOCUS_SNIPPET_CHARS) -> str | None:
    lines = snippet.splitlines()
    if sum(1 for line in lines if "|" in line) < 3:
        return None
    years = query_years(query)
    query_tokens = {token for token in tokenize(query) if len(token) >= 2}
    table_start = next((idx for idx, line in enumerate(lines) if "|" in line), -1)
    if table_start < 0:
        return None
    prefix_text = focus_prefix_text("\n".join(lines[:table_start]), query)
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


def focus_inner_snippet(snippet: str, query: str) -> str:
    table_focus = focus_markdown_table_snippet(snippet, query)
    if table_focus:
        return table_focus
    return trim_snippet_around_query(snippet, query)


def split_inner_evidence_text(text: str, query: str = "") -> list[tuple[int, int, str]]:
    text = compact_html_text(maybe_fix_mojibake(text or ""))
    if len(text) <= INNER_FOCUS_MIN_CHARS:
        return []
    starts = {0}
    for match in re.finditer(r"(?:^|\s)(?:图|表)\s*[\d一二三四五六七八九十]+[：:]", text):
        starts.add(match.start())
    for match in re.finditer(r"<table\b|(?:^|\n)\s*\|.+\|\s*(?:\n|$)", text, flags=re.I):
        starts.add(match.start())
    ordered = sorted(starts)
    spans: list[tuple[int, int, str]] = []
    for idx, start in enumerate(ordered):
        end = ordered[idx + 1] if idx + 1 < len(ordered) else len(text)
        if end - start < 80:
            continue
        snippet_start = start
        if start > 0:
            context_start = max(0, start - 900)
            prefix_context = text[context_start:start]
            paragraph_break = max(prefix_context.rfind("\n\n"), prefix_context.rfind("。"))
            if paragraph_break >= 0:
                snippet_start = context_start + paragraph_break + 1
            else:
                snippet_start = context_start
        snippet = text[snippet_start:end].strip()
        if len(snippet) > INNER_FOCUS_SNIPPET_CHARS:
            snippet = focus_inner_snippet(snippet, query)
        spans.append((snippet_start, min(end, snippet_start + len(snippet)), snippet))
    if len(spans) <= 1:
        spans = []
        window = INNER_FOCUS_SNIPPET_CHARS
        step = max(400, window // 2)
        for start in range(0, len(text), step):
            snippet = focus_inner_snippet(text[start : start + window * 2].strip(), query)
            if len(snippet) >= 120:
                spans.append((start, start + len(snippet), snippet))
    return spans


def inner_focus_score(snippet: str, query: str) -> float:
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
    return score


def build_inner_focus_evidence(option_state: dict, missing_slots: list) -> list[dict]:
    query = missing_query_text(missing_slots)
    if not query:
        return []
    scored: list[tuple[float, dict]] = []
    for ev in prioritize_evidence_for_review(option_state):
        text = ev.get("text", "")
        if len(compact_text(text)) <= INNER_FOCUS_MIN_CHARS:
            continue
        for idx, (_start, _end, snippet) in enumerate(split_inner_evidence_text(text, query)):
            score = inner_focus_score(snippet, query)
            if score <= 0:
                continue
            focused = dict(ev)
            focused["evidence_id"] = f"{ev.get('evidence_id')}_focus_{idx}"
            focused["page_id"] = focused["evidence_id"]
            focused["text"] = snippet
            focused["score"] = round(score, 6)
            focused["source"] = "evidence_inner_focus"
            focused["focused_from"] = ev.get("evidence_id")
            scored.append((score, focused))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item for _score, item in scored[:INNER_FOCUS_MAX_SNIPPETS]]


def local_subsearch_query(state: AgentState, option_state: dict) -> str:
    parts = [
        option_claim(state),
        option_slot_query_text(option_state),
        missing_query_text(option_state.get("audit", {}).get("missing_slots") or []),
    ]
    for search in option_state.get("searches") or []:
        if isinstance(search, dict):
            parts.append(str(search.get("query") or ""))
    parts.append(maybe_fix_mojibake(state["question"].get("question", ""))[:240])
    return maybe_fix_mojibake(" ".join(part for part in parts if part)).strip()


def option_slot_query_text(option_state: dict) -> str:
    parts: list[str] = []
    for item in option_state.get("slots") or []:
        if not isinstance(item, dict):
            continue
        parts.append(str(item.get("slot") or ""))
        parts.append(str(item.get("expected") or ""))
        for query in item.get("queries") or []:
            parts.append(str(query))
    return maybe_fix_mojibake(" ".join(part for part in parts if part)).strip()


def split_long_evidence_for_subsearch(text: str) -> list[str]:
    text = compact_html_text(maybe_fix_mojibake(text or ""))
    text = re.sub(r"[ \t]+", " ", text)
    if len(compact_text(text)) <= LOCAL_SUBSEARCH_MIN_CHARS:
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
        if len(segment) <= LOCAL_SUBSEARCH_SNIPPET_CHARS:
            chunks.append(segment)
            continue
        table_focus = focus_markdown_table_snippet(segment, "", max_chars=LOCAL_SUBSEARCH_SNIPPET_CHARS)
        if table_focus and len(table_focus) >= 80:
            chunks.append(table_focus)
            continue
        step = max(360, LOCAL_SUBSEARCH_SNIPPET_CHARS // 2)
        for start in range(0, len(segment), step):
            window = segment[start : start + LOCAL_SUBSEARCH_SNIPPET_CHARS]
            if len(compact_text(window)) >= 100:
                chunks.append(window.strip())
            if start + LOCAL_SUBSEARCH_SNIPPET_CHARS >= len(segment):
                break
    return chunks


def local_subsearch_score(snippet: str, query: str) -> float:
    score = inner_focus_score(snippet, query)
    query_text = maybe_fix_mojibake(query or "")
    snippet_text = maybe_fix_mojibake(snippet or "")
    query_numbers = set(re.findall(r"\d+(?:\.\d+)?%?|\d{4}", query_text))
    snippet_numbers = set(re.findall(r"\d+(?:\.\d+)?%?|\d{4}", snippet_text))
    score += len(query_numbers & snippet_numbers) * 2.0
    if any(term in snippet_text for term in ("同比", "增速", "增长", "下降", "收入", "利润", "费用", "现金价值", "比例")):
        score += 0.8
    if re.search(r"(?:图|表)\s*[\d一二三四五六七八九十]+[：:]", snippet_text):
        score += 0.5
    return score


def subsearch_long_new_evidence(state: AgentState, source: str) -> list[dict]:
    option = state["current_option"]
    option_state = state["option_states"][option]
    query = local_subsearch_query(state, option_state)
    if not query:
        return []
    evidence_by_id = {item.get("evidence_id"): item for item in option_state.get("evidence", []) or []}
    scored: list[tuple[float, dict]] = []
    for evidence_id in option_state.get("new_evidence_ids") or []:
        ev = evidence_by_id.get(evidence_id)
        if not ev:
            continue
        ev_source = str(ev.get("source") or "")
        if ev_source in {"evidence_extract", "evidence_inner_focus", "evidence_local_subsearch"}:
            continue
        text = ev.get("text", "")
        if len(compact_text(text)) <= LOCAL_SUBSEARCH_MIN_CHARS:
            continue
        for idx, snippet in enumerate(split_long_evidence_for_subsearch(text)):
            focused_text = focus_inner_snippet(snippet, query)
            score = local_subsearch_score(focused_text, query)
            if score <= 0:
                continue
            focused = dict(ev)
            focused["evidence_id"] = f"{evidence_id}_sub_{idx}"
            focused["page_id"] = focused["evidence_id"]
            focused["text"] = focused_text[:LOCAL_SUBSEARCH_SNIPPET_CHARS]
            focused["score"] = round(score, 6)
            focused["source"] = "evidence_local_subsearch"
            focused["focused_from"] = evidence_id
            focused["subsearch_source"] = source
            scored.append((score, focused))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item for _score, item in scored[:LOCAL_SUBSEARCH_MAX_SNIPPETS]]


def apply_local_subsearch_to_new_evidence(state: AgentState, source: str) -> None:
    option = state["current_option"]
    option_state = state["option_states"][option]
    original_new_ids = list(option_state.get("new_evidence_ids") or [])
    focused = subsearch_long_new_evidence(state, source)
    if not focused:
        return
    option_state["evidence"] = merge_evidence(option_state.get("evidence", []), focused)
    focused_ids = [item["evidence_id"] for item in focused]
    option_state["new_evidence_ids"] = focused_ids
    option_state["recent_evidence_ids"] = list(dict.fromkeys(focused_ids + original_new_ids))
    option_state["local_subsearch"] = {
        "source": source,
        "source_evidence_ids": original_new_ids,
        "focused_evidence_ids": focused_ids,
    }
    add_trace(state, f"{option}: local subsearch {len(focused)}")
    if VERBOSE_CONSOLE:
        print("[local subsearch evidence]", flush=True)
        print_evidence_preview(focused)


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


def normalize_key_terms(values: list | None, fallback_text: str = "", max_terms: int = 10) -> list[str]:
    terms: list[str] = []
    for value in values or []:
        term = maybe_fix_mojibake(str(value or "")).strip()
        term = re.sub(r"\s+", "", term)
        if len(term) < 2:
            continue
        if re.fullmatch(r"[，。；、,.。:：]+", term):
            continue
        if term not in terms:
            terms.append(term[:40])
    if not terms and fallback_text:
        fallback = maybe_fix_mojibake(fallback_text)
        for token in tokenize(fallback):
            token = maybe_fix_mojibake(str(token or "")).strip()
            if len(token) < 2:
                continue
            if token not in terms:
                terms.append(token[:40])
            if len(terms) >= max_terms:
                break
    return terms[:max_terms]


def remembered_evidence_ids(option_state: dict) -> set[str]:
    remembered = slot_evidence_ids(option_state.get("memory_slots") or [])
    remembered.update(slot_evidence_ids(option_state.get("evidence_facts") or []))
    audit = option_state.get("audit") or {}
    remembered.update(slot_evidence_ids(audit.get("filled_slots") or []))
    return remembered


INSUFFICIENT_SLOT_VALUE_TERMS = (
    "未找到",
    "未在证据中找到",
    "未提供",
    "没有找到",
    "无法确定",
    "无法判断",
    "证据不足",
    "缺少证据",
    "不明确",
)


def numeric_tokens_for_support(text: str) -> set[str]:
    text = maybe_fix_mojibake(text)
    tokens: set[str] = set()
    for match in re.finditer(r"\d+(?:,\d{3})*(?:\.\d+)?%?", text):
        token = match.group(0)
        compact = token.replace(",", "")
        is_percent = compact.endswith("%")
        if (not is_percent) and len(compact.rstrip("%").replace(".", "")) < 2:
            continue
        tokens.add(token)
        tokens.add(compact)
    return tokens


def evidence_text_for_ids(option_state: dict, evidence_ids: list[str]) -> str:
    evidence_by_id = {item.get("evidence_id"): item for item in option_state.get("evidence", []) or []}
    parts: list[str] = []
    for evidence_id in evidence_ids:
        item = evidence_by_id.get(evidence_id)
        if not item:
            continue
        parts.append(maybe_fix_mojibake(" > ".join(item.get("heading_path") or [])))
        parts.append(maybe_fix_mojibake(str(item.get("text") or "")))
    return "\n".join(parts)


def filled_value_looks_option_copied_without_evidence(item: dict, option_state: dict) -> bool:
    value = maybe_fix_mojibake(str(item.get("value") or ""))
    claim = maybe_fix_mojibake(str(option_state.get("claim") or ""))
    evidence_ids = [evidence_id for evidence_id in item.get("evidence_ids") or [] if evidence_id]
    if not value or not claim or not evidence_ids:
        return False
    value_tokens = numeric_tokens_for_support(value)
    if not value_tokens:
        return False
    claim_tokens = numeric_tokens_for_support(claim)
    copied_tokens = value_tokens & claim_tokens
    if not copied_tokens:
        return False
    evidence_tokens = numeric_tokens_for_support(evidence_text_for_ids(option_state, evidence_ids))
    return not any(token in evidence_tokens for token in copied_tokens)


def filled_slot_is_incomplete_for_judgment(item: dict) -> bool:
    slot = maybe_fix_mojibake(str(item.get("slot") or "")).strip()
    value = maybe_fix_mojibake(str(item.get("value") or "")).strip()
    if not value:
        return True
    incomplete_terms = (
        "?",
        "？",
        "未知",
        "未给出",
        "未直接给出",
        "缺失",
        "缺少",
        "无法",
        "不能",
        "待定",
        "需计算",
        "需要计算",
        "另行计算",
        "需另行计算",
        "需要另行计算",
        "另行确认",
        "需另行确认",
        "需要另行确认",
        "另行补充",
        "需另行补充",
        "需进一步",
        "需要进一步",
        "待补充",
        "待计算",
        "待确认",
    )
    if any(term in value for term in incomplete_terms):
        return True
    relation_terms = ("高于", "低于", "大于", "小于", "超过", "不足", "均", "比较", "占比", "比例", "比重")
    if any(term in slot for term in relation_terms):
        if any(term in value for term in ("已比较", "已确认事实", "可反推")):
            return True
        if re.search(r"[<>＞＜]\s*$", value):
            return True
    return False


def normalize_audit_slots(audit: dict, option_state: dict) -> dict:
    filled_slots = audit.get("filled_slots") or []
    missing_slots = audit.get("missing_slots") or audit.get("matching_slots") or audit.get("matched_slots") or []
    normalized_filled: list[dict] = []
    normalized_missing: list[dict] = []
    seen_missing: set[str] = set()

    def add_missing(item: dict | str) -> None:
        if isinstance(item, dict):
            slot = maybe_fix_mojibake(str(item.get("slot") or "")).strip()
            query = maybe_fix_mojibake(str(item.get("followup_query") or item.get("query") or slot)).strip()
            target_doc_ids = [
                doc_id for doc_id in item.get("target_doc_ids", []) if doc_id in (option_state.get("target_doc_ids") or [])
            ]
            key_terms = normalize_key_terms(item.get("key_terms") or [], " ".join([slot, query]))
        else:
            slot = maybe_fix_mojibake(str(item or "")).strip()
            query = slot
            target_doc_ids = []
            key_terms = normalize_key_terms([], slot)
        if not slot and not query:
            return
        key = f"{slot}|{query}|{'/'.join(target_doc_ids)}"
        if key in seen_missing:
            return
        seen_missing.add(key)
        normalized_missing.append(
            {
                "slot": slot or query,
                "target_doc_ids": target_doc_ids or list(option_state.get("target_doc_ids") or []),
                "followup_query": query or slot,
                "key_terms": key_terms,
            }
        )

    for item in missing_slots:
        add_missing(item)

    for item in filled_slots:
        if not isinstance(item, dict):
            continue
        value = maybe_fix_mojibake(str(item.get("value") or "")).strip()
        evidence_ids = [evidence_id for evidence_id in item.get("evidence_ids") or [] if evidence_id]
        if filled_slot_is_incomplete_for_judgment(item):
            add_missing(
                {
                    "slot": item.get("slot") or "",
                    "target_doc_ids": item.get("target_doc_ids") or option_state.get("target_doc_ids") or [],
                    "followup_query": item.get("followup_query") or item.get("query") or item.get("slot") or "",
                    "key_terms": item.get("key_terms") or [],
                }
            )
            continue
        if (not evidence_ids) and any(term in value for term in INSUFFICIENT_SLOT_VALUE_TERMS):
            add_missing(
                {
                    "slot": item.get("slot") or "",
                    "target_doc_ids": item.get("target_doc_ids") or option_state.get("target_doc_ids") or [],
                    "followup_query": item.get("followup_query") or item.get("query") or item.get("slot") or "",
                    "key_terms": item.get("key_terms") or [],
                }
            )
            continue
        if filled_value_looks_option_copied_without_evidence(item, option_state):
            add_missing(
                {
                    "slot": item.get("slot") or "",
                    "target_doc_ids": item.get("target_doc_ids") or option_state.get("target_doc_ids") or [],
                    "followup_query": item.get("followup_query") or item.get("query") or item.get("slot") or "",
                    "key_terms": item.get("key_terms") or [],
                }
            )
            continue
        normalized_filled.append(item)

    audit["filled_slots"] = normalized_filled
    audit["missing_slots"] = normalized_missing
    if normalized_missing:
        audit["can_judge"] = False
    return audit


def filter_task_audit_slots(filled_slots: list[dict]) -> list[dict]:
    """Keep audit memory to rules and inputs; final task results are computed later."""
    blocked_slot_terms = ("最终", "结果", "答案", "实际退保金额", "退保金额", "所得金额", "计算结果", "排序")
    kept: list[dict] = []
    for item in filled_slots or []:
        if not isinstance(item, dict):
            continue
        slot = maybe_fix_mojibake(str(item.get("slot") or "")).strip()
        value = maybe_fix_mojibake(str(item.get("value") or "")).strip()
        evidence_ids = [evidence_id for evidence_id in item.get("evidence_ids") or [] if evidence_id]
        if any(term in slot for term in blocked_slot_terms):
            continue
        if not evidence_ids and any(term in value for term in ("缺少", "未提供", "需从", "需要", "待定", "未知", "无法")):
            continue
        if filled_slot_is_incomplete_for_judgment(item):
            continue
        if any(term in value for term in ("选项A", "选项B", "选项C", "选项D", "以选项")):
            continue
        if filled_slot_is_bare_relation_conclusion(item):
            continue
        kept.append(item)
    return kept


def filled_slot_is_bare_relation_conclusion(item: dict) -> bool:
    slot = maybe_fix_mojibake(str(item.get("slot") or "")).strip()
    value = maybe_fix_mojibake(str(item.get("value") or "")).strip()
    if not slot or not value:
        return False
    if numeric_tokens_for_support(value):
        return False
    relation_terms = (
        "是否",
        "较",
        "相比",
        "比",
        "高于",
        "低于",
        "大于",
        "小于",
        "多于",
        "少于",
        "增长",
        "下降",
        "上升",
        "减少",
        "增加",
        "提升",
        "降低",
    )
    bare_values = (
        "是",
        "否",
        "正确",
        "错误",
        "增长",
        "下降",
        "上升",
        "减少",
        "增加",
        "提升",
        "降低",
        "高于",
        "低于",
        "大于",
        "小于",
        "多于",
        "少于",
    )
    return any(term in slot for term in relation_terms) and any(term == value or term in value for term in bare_values)


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


def evidence_by_ids_strict(evidence: list[dict], evidence_ids: list[str]) -> list[dict]:
    if not evidence_ids:
        return []
    wanted = set(evidence_ids)
    return [item for item in evidence if item.get("evidence_id") in wanted]


def current_evidence_for_read(option_state: dict, max_items: int = 6) -> list[dict]:
    batch_ids = normalize_evidence_id_list(option_state.get("evidence_read_current_ids") or [])
    if not batch_ids:
        batch_ids = normalize_evidence_id_list(option_state.get("new_evidence_ids") or option_state.get("recent_evidence_ids") or [])
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


def should_toc_repair(state: AgentState) -> bool:
    option_state = state["option_states"][state["current_option"]]
    audit = option_state.get("audit") or {}
    missing = audit.get("missing_slots") or []
    if not missing:
        return False
    if option_state.get("toc_repair_used"):
        return False
    if option_state.get("round", 0) < 2:
        return False
    return True


def tried_section_summaries(option_state: dict, max_items: int = 8) -> list[dict]:
    summaries: list[dict] = []
    seen: set[str] = set()
    for item in prioritize_evidence_for_review(option_state):
        heading = maybe_fix_mojibake(" > ".join(item.get("heading_path") or [])).strip()
        if not heading:
            selected = item.get("selected_sections") or item.get("selected_titles") or []
            heading = maybe_fix_mojibake(" > ".join(str(part) for part in selected)).strip()
        if not heading:
            continue
        key = normalize_heading_key(heading)
        if not key or key in seen:
            continue
        seen.add(key)
        summaries.append(
            {
                "doc_id": item.get("doc_id"),
                "heading": heading[:260],
                "source": item.get("source"),
            }
        )
        if len(summaries) >= max_items:
            break
    return summaries


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


def build_toc_repair_prompt(state: AgentState) -> tuple[str, str]:
    option = state["current_option"]
    option_state = state["option_states"][option]
    system = (
        "你是目录补证Agent。当前普通检索证据不足，你需要根据题干、当前选项、缺失槽位和文档目录，"
        "选择最可能包含答案的章节、图或表。只输出JSON。"
        "可以从全部候选文档中补选文档，不要局限于之前router选中的文档。"
        "每个缺失事实优先选择最具体的图表标题或章节标题。"
    )
    recent = []
    for item in prioritize_evidence_for_review(option_state)[:6]:
        recent.append(
            {
                "evidence_id": item.get("evidence_id"),
                "doc_id": item.get("doc_id"),
                "heading": " > ".join(item.get("heading_path") or []),
                "preview": compact_text(item.get("text", ""))[:160],
            }
        )
    user = f"""
题干：{maybe_fix_mojibake(state["question"]["question"])}
当前选项：{option}. {option_claim(state)}

缺失槽位：
{json.dumps(option_state.get("audit", {}).get("missing_slots") or [], ensure_ascii=False)}

已有证据摘要：
{json.dumps(recent, ensure_ascii=False, indent=2)}

已尝试但仍未补齐缺失槽位的章节：
{json.dumps(tried_section_summaries(option_state), ensure_ascii=False, indent=2)}

候选文档目录：
{compact_toc_for_docs(state.get("doc_ids") or [])}

请输出：
{{
  "section_queries": [
    {{"doc_id": "doc_id", "section_hint": "目录中的章节/图/表标题", "query": "进入该章节后用于核验证据的短关键词"}}
  ],
  "reason": "为什么这些章节能补齐缺失槽位"
}}
最多输出3个section_queries。doc_id必须来自候选文档。section_hint应尽量原样复制目录标题。
如果已尝试章节只命中相近但不直接回答的内容，必须改选同一文档内其他更相关章节、相邻条款或上级目录下的其他子条款。
"""
    return system, user


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
    candidates = [
        section_title(section),
        " > ".join(section.get("heading_path") or []),
        section.get("preview", ""),
    ]
    for text in candidates:
        text_key = normalize_heading_key(str(text or "")[:3000])
        if key and text_key and (key in text_key or text_key in key):
            return True
    return False


def retrieve_by_toc_hint(doc_id: str, section_hint: str, query: str, max_items: int = 3) -> list[dict]:
    evidence: list[dict] = []
    selected_sections = [
        section for section in DOC_SECTIONS_BY_DOC.get(doc_id, []) if section_matches_hint(section, section_hint)
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
            title_evidence = section_title_as_evidence(
                section,
                "toc_repair_title",
                [section_id],
                [section_name],
            )
            if title_evidence:
                section_evidence.append(title_evidence)
        evidence.extend(section_evidence)
    if len(evidence) >= max_items:
        return evidence[:max_items]

    seen = {item.get("evidence_id") for item in evidence}
    for page in PAGES_BY_DOC.get(doc_id, []):
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


def doc_profile_text(doc_id: str, max_chars: int = 420) -> str:
    parts: list[str] = []
    tree_doc = TREE_DOC_BY_ID.get(doc_id) or {}
    for key in ("title", "doc_title", "source_path"):
        value = tree_doc.get(key)
        if value:
            parts.append(Path(str(value)).stem if key == "source_path" else str(value))
    tree_text = DOC_TREE_TEXT_BY_ID.get(doc_id, "")
    if tree_text:
        parts.append(tree_text[:1200])
    profile_pages = list(PAGES_BY_DOC.get(doc_id, [])[:3])
    for page in PAGES_BY_DOC.get(doc_id, []):
        heading = " > ".join(page.get("heading_path") or [])
        title = str(page.get("title") or "")
        if any(term in f"{title}\n{heading}" for term in ("目录", "图表目录", "内容目录")):
            if page not in profile_pages:
                profile_pages.append(page)
        if len(profile_pages) >= 6:
            break
    for page in profile_pages:
        title = str(page.get("title") or "")
        heading = " > ".join(page.get("heading_path") or [])
        text = str(page.get("text") or "")[:520 if "目录" in f"{title}{heading}" else 180]
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
                if is_doc_hint_phrase(phrase_key) and phrase_key in normalized_profile:
                    if phrase_key not in phrase_hits:
                        phrase_hits.append(phrase_key)
                if len(phrase_hits) >= 3:
                    break
            if len(phrase_hits) >= 3:
                break
        for phrase_key in phrase_hits:
            score += min(8.0, len(phrase_key) / 1.5)
        scored.append((score, doc_id, overlap))
    return sorted(scored, key=lambda item: item[0], reverse=True)


def build_missing_doc_reroute_query(state: AgentState, missing_item: dict | str, option_state: dict) -> str:
    parts = [
        maybe_fix_mojibake(str(state.get("question", {}).get("question") or "")),
        option_claim(state),
        option_slot_query_text(option_state),
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
    query = maybe_fix_mojibake(query)
    query_terms = [
        token
        for token in dict.fromkeys(tokenize(query))
        if is_doc_hint_token(token)
    ]
    if not query_terms:
        return []
    scored: list[tuple[float, str, list[str]]] = []
    phrase_terms = [
        compact_text(term)
        for term in normalize_key_terms([], query, max_terms=16)
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


def reroute_docs_for_missing_slot(state: AgentState, missing_item: dict | str, option_state: dict) -> list[str]:
    attempted = set(option_state.setdefault("doc_reroute_attempted_slots", []))
    query = build_missing_doc_reroute_query(state, missing_item, option_state)
    query_key = hashlib.md5(query.encode("utf-8", errors="ignore")).hexdigest()[:12]
    if query_key in attempted:
        return []
    attempted.add(query_key)
    option_state["doc_reroute_attempted_slots"] = list(attempted)[-12:]

    current_doc_ids = list(dict.fromkeys(option_state.get("target_doc_ids") or state.get("doc_ids") or []))
    all_doc_ids = sorted(PAGES_BY_DOC.keys())
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
        if score < DOC_REROUTE_MIN_SCORE:
            continue
        if current_best and score < current_best * DOC_REROUTE_MARGIN:
            continue
        selected.append(doc_id)
        if len(selected) >= DOC_REROUTE_TOP_K:
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


def has_doc_reroute_candidate(state: AgentState) -> bool:
    option_state = state.get("option_states", {}).get(state.get("current_option", ""), {})
    missing = option_state.get("audit", {}).get("missing_slots") or []
    if not missing:
        return False
    if len(state.get("doc_ids") or []) >= len(PAGES_BY_DOC):
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
        global_scores = doc_reroute_scores(query, sorted(PAGES_BY_DOC.keys()))
        if not global_scores:
            continue
        if query not in current_scores_cache:
            current_scores = doc_reroute_scores(query, current_doc_ids)
            current_scores_cache[query] = current_scores[0][0] if current_scores else 0.0
        current_best = current_scores_cache[query]
        for score, doc_id, _overlap in global_scores:
            if doc_id in current_doc_ids:
                continue
            if score < DOC_REROUTE_MIN_SCORE:
                continue
            if current_best and score < current_best * DOC_REROUTE_MARGIN:
                continue
            return True
    return False


def deterministic_doc_hints(state: AgentState, max_docs: int = 2) -> list[str]:
    claim = option_claim(state)
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


def option_claim(state: AgentState) -> str:
    option = state["current_option"]
    option_state = state.get("option_states", {}).get(option, {})
    if "claim" in option_state:
        return maybe_fix_mojibake(option_state["claim"])
    return maybe_fix_mojibake(state["question"]["options"][option])


UNIVERSAL_CLAIM_TERMS = ("都", "均", "全部", "所有", "每个", "各", "皆")


def atomic_entities_for_state(state: AgentState) -> list[str]:
    atomic = state.get("question", {}).get("enterprise_atomic") or {}
    entities: list[str] = []
    for value in atomic.get("entities") or []:
        entity = maybe_fix_mojibake(str(value or "")).strip()
        if len(entity) < 2:
            continue
        if entity not in entities:
            entities.append(entity)
    return entities


def needs_entity_coverage_slots(state: AgentState) -> bool:
    entities = atomic_entities_for_state(state)
    if len(entities) < 2:
        return False
    claim = option_claim(state)
    question = maybe_fix_mojibake(str(state.get("question", {}).get("question") or ""))
    constraints = " ".join(
        maybe_fix_mojibake(str(item or ""))
        for item in (state.get("question", {}).get("enterprise_atomic") or {}).get("constraints") or []
    )
    surface = claim + "\n" + question + "\n" + constraints
    if "需要覆盖" in constraints and "所有" in constraints:
        return True
    return any(term in surface for term in UNIVERSAL_CLAIM_TERMS)


def slot_surface(slot: dict) -> str:
    parts = [
        str(slot.get("slot") or ""),
        str(slot.get("expected") or ""),
        " ".join(str(item) for item in slot.get("queries") or []),
        " ".join(str(item) for item in slot.get("key_terms") or []),
    ]
    return maybe_fix_mojibake(" ".join(parts))


def ensure_entity_coverage_slots(state: AgentState, slots: list[dict]) -> list[dict]:
    if not needs_entity_coverage_slots(state):
        return slots
    entities = atomic_entities_for_state(state)
    if not entities:
        return slots
    existing_surfaces = [slot_surface(slot) for slot in slots]
    core_terms: list[str] = []
    entity_keys = {compact_text(entity) for entity in entities}
    for slot in slots:
        for term in slot.get("key_terms") or []:
            term = maybe_fix_mojibake(str(term or "")).strip()
            if not term or compact_text(term) in entity_keys:
                continue
            if term not in core_terms:
                core_terms.append(term)
    if not core_terms:
        core_terms = [
            term for term in normalize_key_terms([], option_claim(state), max_terms=8)
            if compact_text(term) not in entity_keys
        ]
    enhanced = list(slots)
    for entity in entities:
        entity_key = compact_text(entity)
        if any(entity_key and entity_key in compact_text(surface) for surface in existing_surfaces):
            continue
        key_terms = normalize_key_terms([entity] + core_terms, " ".join([entity, option_claim(state)]), max_terms=8)
        query = " ".join(key_terms[:8]) or entity
        enhanced.append(
            {
                "slot": f"{entity}是否支持当前选项中的全称陈述"[:100],
                "expected": "",
                "queries": [query[:160]],
                "key_terms": key_terms,
            }
        )
        existing_surfaces.append(slot_surface(enhanced[-1]))
    return enhanced


CAUSE_SPLIT_RE = re.compile(r"(?:因为|由于|原因是|源于)([^。；;\n]+)")
CAUSE_SEPARATOR_RE = re.compile(r"(?:且|并且|以及|同时|、|，|,)")
CLAIM_FOCUS_RE = re.compile(r"(?:集中在|位于|面向|针对|依托|通过|基于)([^，,。；;因由且并及同时]+)")


def claim_focus_for_reason(claim: str) -> str:
    before_cause = re.split(r"因为|由于|原因是|源于", claim, maxsplit=1)[0]
    matches = CLAIM_FOCUS_RE.findall(before_cause)
    if not matches:
        return ""
    focus = maybe_fix_mojibake(matches[-1]).strip()
    focus = re.sub(r"^(?:该|当地|相关|上述|这些)", "", focus).strip()
    return focus[:40]


def extract_claim_reason_parts(claim: str) -> list[str]:
    claim = maybe_fix_mojibake(claim)
    parts: list[str] = []
    focus = claim_focus_for_reason(claim)
    for match in CAUSE_SPLIT_RE.finditer(claim):
        cause_text = match.group(1)
        cause_text = re.split(r"[。；;\n]", cause_text, maxsplit=1)[0]
        for raw_part in CAUSE_SEPARATOR_RE.split(cause_text):
            part = maybe_fix_mojibake(raw_part).strip(" ：:，,。；; ")
            if len(part) < 2:
                continue
            if focus and part.startswith(("该地区", "当地", "该地", "该区域")):
                part = re.sub(r"^(?:该地区|当地|该地|该区域)", focus, part)
            elif focus and not any(term in part for term in (focus, "地区", "市场", "区域")):
                part = f"{focus}{part}"
            if part and part not in parts:
                parts.append(part[:80])
    return parts[:4]


def ensure_compound_reason_slots(state: AgentState, slots: list[dict]) -> list[dict]:
    claim = option_claim(state)
    reason_parts = extract_claim_reason_parts(claim)
    if len(reason_parts) <= 1:
        return slots
    enhanced = list(slots)
    existing_surfaces = [slot_surface(slot) for slot in enhanced]
    for reason in reason_parts:
        reason_key = compact_text(reason)
        if any(reason_key and reason_key in compact_text(surface) for surface in existing_surfaces):
            continue
        key_terms = normalize_key_terms([], reason, max_terms=8)
        query = " ".join(key_terms[:8]) or reason
        enhanced.append(
            {
                "slot": f"当前选项原因是否有证据支持：{reason}"[:100],
                "expected": "是",
                "queries": [query[:160]],
                "key_terms": key_terms,
            }
        )
        existing_surfaces.append(slot_surface(enhanced[-1]))
    return enhanced


def build_doc_router_prompt(state: AgentState) -> tuple[str, str]:
    option = state["current_option"]
    doc_lines = "\n".join(f"- {alias}: {doc_id}" for doc_id, alias in state["doc_aliases"].items())
    profile_lines = doc_profile_lines(state.get("doc_ids") or [])
    hint_docs = deterministic_doc_hints(state)
    system = (
        "你是Doc Router。判断当前选项需要检索哪些参考文档。只输出JSON。"
        "如果当前选项出现明确产品名、机构名、法规名，应优先选择文档摘要中直接包含该名称或高度相近名称的文档。"
    )
    user = f"""
题干：{maybe_fix_mojibake(state["question"]["question"])}
当前选项：{option}. {option_claim(state)}

可用文档：
{doc_lines}

文档摘要：
{profile_lines}

程序预匹配的候选文档（若非空，通常应优先考虑）：
{json.dumps(hint_docs, ensure_ascii=False)}

请输出：
{{
  "target_doc_ids": ["doc_id"],
  "doc_reasons": {{"doc_id": "为什么需要该文档"}},
  "reason": "简短说明"
}}
"""
    return system, user


def build_slot_planner_prompt(state: AgentState) -> tuple[str, str]:
    option = state["current_option"]
    entities = atomic_entities_for_state(state)
    system = (
        "你是选项槽位规划Agent。只根据题干和当前选项，拆出验证该选项所需的最小事实槽位，"
        "并为每个槽位生成用于全文检索的短关键词。不要选择文档，不要判断选项对错。只输出JSON。"
    )
    user = f"""
题干：{maybe_fix_mojibake(state["question"]["question"])}
题型：{state["question"].get("answer_format")}
当前选项：{option}. {option_claim(state)}
题目实体：{json.dumps(entities, ensure_ascii=False)}

请输出：
{{
  "slots": [
    {{
      "slot": "需要验证的事实",
      "expected": "选项声称的值/关系，可为空",
      "queries": ["语义检索短关键词", "数值/时间检索短关键词"],
      "key_terms": ["当前槽位必须覆盖的对象/指标/条件/关系关键词"]
    }}
  ]
}}
要求：
1. 只拆当前选项本身需要验证的事实，不要加入无关背景。
2. query 应包含对象、时间、指标；如果选项含数值或方向，至少一个 query 包含该数值或方向。
3. 必须保留选项中的关键口径词，不要改成相近但不同的指标；例如“研发投入占营业收入比例”不要改成“研发费用 营业收入 比例”，“每股”不要改成“每10股”，“现金分红金额占归母净利润”不要改成“本次利润分配中占比”。
4. query 不要写成长句，使用空格分隔关键词。
5. key_terms 来自当前题干和选项，不要写行业通用词表；每个槽位3-8个。
6. 一般输出1-2个slots，每个slot最多2个queries。
7. 如果当前选项是“都/均/全部/所有/每个”等全称断言，且题目实体包含多个对象，必须为每个对象分别输出槽位；判断支持时需要全部对象都有证据，判断不支持时可由任一明确反例支持。
8. 如果当前选项包含“因为/由于/原因是”等因果解释，必须把结论和每个原因分别拆成槽位；“且/并且/以及/同时”连接的多个原因不能合并为一个槽位。
"""
    return system, user


def build_audit_prompt(state: AgentState) -> tuple[str, str]:
    option = state["current_option"]
    option_state = state["option_states"][option]
    system = (
        "你是槽位状态审查Agent。判断已抽取事实和当前证据是否足以验证当前选项；不足则给出缺失槽位和短检索词。"
        "优先使用已抽取事实；当前证据正文只用于核对和补充。"
        "单位、期间、分母、统计口径必须一致或明确可换算；“每10股”和“每股”、“研发费用”和“研发投入”、“年度现金分红”和“年度+特别现金分红”不能直接视为同一口径。"
        "如果原文同时给出组成项和合计项，必须按当前选项措辞对应的最小口径抽取；选项未写“合计/总额/含其他方式/特别”等词时，不要自行扩大为合计口径。"
        "例如“同比下降18.97%”可以支持“增速=-18.97%”，“同比增长14%”可以支持“增速=14%”。"
        "filled_slots中的value必须来自证据原文、题干已知事实或基于证据中数字的明确计算；禁止把当前选项声称的数值直接当作已确认事实。"
        "你必须先在内部明确验证当前选项所需的最小事实集合；只有这些事实都能由题干、已确认事实或当前证据支持时，can_judge才能为true。"
        "只要任一必要事实缺失，就必须把该事实写入missing_slots，并设置can_judge=false。"
        "filled_slots禁止写半截结论或待计算结论；value中不得出现问号、未知、未直接给出、需计算、可反推、已比较等不完整表述。"
        "比较/比例类选项必须拿到双方对象的可比较数值或可直接支持的原文比例，缺任一方都必须进入missing_slots。"
        "因果解释类选项必须同时核对结论和每个原因；任一原因没有直接证据，或证据把该原因归属于其他对象/地区/期间/口径，都不能视为已填充。"
        "含“主要、集中、重点、核心、首要”等程度词的槽位，证据必须支持该程度；若原文是多个并列对象或多个核心区域，不能支持“主要集中在其中一个对象”。"
        "证据中的限定对象必须和槽位对象一致；例如某因素属于A地区，不能迁移为B地区的原因。"
        "禁止用常识、估算、行业经验、选项倾向、排除法或未经证据支持的假设来补足缺失事实。"
        "不要判断证据块是否永久无关，不要输出需要丢弃的证据ID。"
        "只输出JSON，且只能使用can_judge、filled_slots、missing_slots三个顶层字段。"
    )
    user = f"""
题干：{maybe_fix_mojibake(state["question"]["question"])}
当前选项：{option}. {option_claim(state)}
目标文档：{json.dumps(option_state.get("target_doc_ids", []), ensure_ascii=False)}

待验证槽位：
{json.dumps(option_state.get("slots", []), ensure_ascii=False, indent=2)}

已确认事实：
{format_memory_slots(option_state.get("memory_slots", []))}

本轮从证据中抽取的事实：
{format_memory_slots(option_state.get("evidence_facts", []))}

当前证据：
{format_evidence(prioritize_evidence_for_review(option_state), max_chars=3600)}

请输出：
{{
  "can_judge": true,
  "filled_slots": [{{"slot": "事实槽位", "value": "事实值", "evidence_ids": ["证据ID"]}}],
  "missing_slots": [{{"slot": "缺失槽位", "target_doc_ids": ["doc_id"], "followup_query": "短关键词", "key_terms": ["下一轮证据必须覆盖的关键词"]}}]
}}
不要输出matching_slots、matched_slots、matched_evidence等其他字段。若证据表述为“同比下降18.97%”，可将事实值写为“-18.97%（同比下降18.97%）”。
如果当前选项需要比较、计算或判断多个对象，请分别检查每个对象的必要事实是否都有证据。缺少任何一项时，不要进入最终判断。
如果当前选项需要计算占比、比例、比重、强度或率，分子和分母可以来自不同证据块；只要对象、年份、单位和口径一致，就应合并计算并填入filled_slots。
如果只拿到了分子但没有分母，或只拿到一方比例但另一方比例未知，必须把缺少的分母/比例写入missing_slots，不得把“高于/低于/已比较”写入filled_slots。
不要因为槽位名称和原文措辞不完全一致就判为缺失；如果原文能够等价支持该事实，应填入filled_slots并引用证据ID。
如果选项中的数值没有出现在证据中，且不能由证据中的原始数字直接计算得到，必须判为缺失或填入证据中的实际值，不得照抄选项数值。
如果证据只支持相近但不同口径的事实，应把正确口径写入filled_slots；若还缺少当前选项口径，则写入missing_slots继续检索。
如果同一证据同时列出单项、特别项、其他方式、合计/总额等多种口径，应分别识别；当前选项没有明确要求合计时，优先抽取与选项文字最贴近的单项或原文同名项目。
如果选项用“因为/由于/且/并且”给出多个原因，必须逐一填充每个原因；原因A有证据不能替代原因B。若原文把某原因写给其他对象、地区或市场，应把当前对象下该原因写入missing_slots。
如果选项声称“主要集中/重点集中/主要目的地”，但证据只说“两个核心区域/多个重点/其中之一”，不得填为“是”，应写入missing_slots或给出实际关系。
missing_slots 的 key_terms 必须来自缺失槽位和当前题目，不要写行业通用词表；用于后续证据排序。
"""
    return system, user


def build_judge_prompt(state: AgentState) -> tuple[str, str]:
    option = state["current_option"]
    option_state = state["option_states"][option]
    system = (
        "你是做题Agent。只能基于给定证据判断当前选项陈述本身是否正确。"
        "verdict=true表示当前选项陈述正确，verdict=false表示当前选项陈述错误或证据不足。只输出JSON。"
        "已确认事实是审查Agent从证据中抽取的结构化线索；证据原文和基于同口径数字的可复算结果具有更高优先级。"
        "如果已确认事实与证据原文、表格数字或你重新计算出的关系冲突，必须以证据原文和重新计算结果为准，并在reason中说明冲突。"
        "如果已确认事实中已经包含某对象、指标、年份或数值，不要因为同一证据块还包含其他对象的信息而忽略或否定该事实。"
        "如果当前选项包含高于、低于、多于、少于、大于、小于、快于、慢于、均为、是否等关系，必须先识别选项声称的关系，再用证据计算或抽取实际关系，最后检查二者是否一致。"
        "如果证据显示的关系与选项声称的方向相反，verdict必须为false。"
        "必须逐字核对单位和统计口径；每股、每10股、每手、每百元、含税/不含税、年度现金分红、特别现金分红、合计现金分红、研发费用、研发投入、占营业收入比例等口径不一致时，除非证据明确给出换算并换算后相同，否则verdict=false。"
        "如果证据同时存在组成项和合计项，应先按当前选项文字选择最贴近的同名/同义项目；选项未明确写合计、总额、含其他方式、年度+特别等词时，不要主动改用合计项否定它。"
        "如果已确认事实来自相近但不同口径的证据，不能据此判当前选项为true。"
        "当题干和当前选项已经给出明确比较对象和指标时，不要改用其他年份、其他口径或额外背景事实推翻当前证据支持的同口径判断。"
        "当前选项若包含多个原因、并列条件或因果解释，必须逐项核对；任一原因无直接证据或归属对象不一致，verdict不能为true。"
        "当前选项若包含主要、集中、重点、核心等程度词，证据必须支持该程度；证据只说多个并列对象、多个核心区域或其中之一时，不能支持“主要集中在某一个对象”。"
        "如果审查结果仍有missing_slots，说明当前必要事实未齐全；不得用题干背景、没有反例、行业常识或猜测把缺失事实补成true。"
        "题干背景只能提供待核验范围，不能替代证据证明选项中的新增关系、数值或结论。"
    )
    user = f"""
题干：{maybe_fix_mojibake(state["question"]["question"])}
题型：{state["question"].get("answer_format")}
当前选项：{option}. {option_claim(state)}

已确认事实：
{format_memory_slots(option_state.get("memory_slots", []))}

证据阅读事实：
{format_memory_slots(option_state.get("evidence_facts", []))}

审查结果：
{json.dumps(option_state.get("audit", {}), ensure_ascii=False, indent=2)}

证据：
{format_evidence(select_judge_evidence(option_state), max_chars=6000)}

请输出：
{{
  "verdict": true,
  "confidence": 0.0,
  "reason": "简短依据",
  "answer": "若当前选项正确则输出当前选项字母，否则输出空字符串；判断题可输出A或B"
}}
注意：不要把“已经完成判断”理解为verdict=true；verdict只表示当前选项陈述是否正确。若理由中认为当前选项错误，verdict必须为false。
reason必须说明：选项声称的关系是什么、证据得到的关系是什么、二者是否一致。比较方向相反时不要输出当前选项字母。
reason还必须说明单位和口径是否一致；例如证据为“每10股派息45.53元”而选项为“每股45.53元”时，应判为false。
判断时可以先使用“已确认事实”中的对象-指标-数值作为线索，但遇到比较/比例/增长下降/是否高低这类关系时，必须用证据中的原始数值或明确同比值复核一遍。
如果一个选项可由证据中的同名组成项直接支持，不要因为同一段证据还存在更大的合计项就判为错误；只有选项明确要求合计口径时才用合计项。
如果审查结果中missing_slots非空，除非证据部分已经逐项补齐这些missing_slots并在reason中引用证据ID，否则verdict必须为false。
“题干已经提到该领域/对象”“没有找到反例”“可能存在类似案例”都不能作为verdict=true的理由。
如果选项说“因为A且B”，reason必须分别说明A和B的证据；不能用A的证据替代B，也不能把属于其他对象/地区/期间的原因迁移到当前对象。
如果证据原文是“两个核心区域/多个重点/一是X、二是Y”，而选项说“主要集中在X”，应判为false或证据不足，除非证据另有明确主次排序。
"""
    return system, user


def enforce_judgment_missing_slot_guard(judgment: dict, option_state: dict, option: str, answer_format: str) -> dict:
    audit = option_state.get("audit") or {}
    missing = audit.get("missing_slots") or []
    if not missing or judgment.get("verdict") is not True:
        return judgment
    judgment["verdict"] = False
    judgment["confidence"] = min(float(judgment.get("confidence") or 0.0), 0.5)
    judgment["missing_slots"] = missing
    prior_reason = maybe_fix_mojibake(str(judgment.get("reason") or "")).strip()
    judgment["reason"] = (
        "审查结果仍存在未补齐槽位，不能用题干背景、无反例或推测替代证据判为支持。"
        + (f" 原模型理由：{prior_reason}" if prior_reason else "")
    )[:1200]
    if answer_format == "tf":
        judgment["answer"] = "F"
    else:
        judgment["answer"] = ""
    return judgment


def make_option_state(claim: str) -> dict:
    return {
        "claim": maybe_fix_mojibake(claim),
        "status": "pending",
        "round": 0,
        "slots": [],
        "target_doc_ids": [],
        "searches": [],
        "router": {},
        "evidence": [],
        "new_evidence_ids": [],
        "recent_evidence_ids": [],
        "evidence_facts": [],
        "evidence_read_batches": [],
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
            "current_option": order[0] if order else "",
            "option_states": options,
            "status": "running",
            "trace": [],
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    )
    add_trace(state, "init")
    return state


def slot_planner_node(state: AgentState) -> AgentState:
    option = state["current_option"]
    option_state = state["option_states"][option]
    if DRY_RUN_WITHOUT_LLM:
        system, user = build_slot_planner_prompt(state)
        plan = dry_json(system, user, "slot_plan")
    else:
        system, user = build_slot_planner_prompt(state)
        plan, usage = call_qwen_json(system, user)
        add_usage(state, usage)
    slots: list[dict] = []
    for item in plan.get("slots") or []:
        if not isinstance(item, dict):
            continue
        slot = maybe_fix_mojibake(str(item.get("slot") or "")).strip()
        expected = maybe_fix_mojibake(str(item.get("expected") or "")).strip()
        queries = []
        for query in item.get("queries") or []:
            query = maybe_fix_mojibake(str(query or "")).strip()
            if query and query not in queries:
                queries.append(query[:160])
        key_terms = normalize_key_terms(
            item.get("key_terms") or [],
            " ".join([slot, expected, " ".join(queries)]),
            max_terms=8,
        )
        if slot or expected or queries:
            slots.append(
                {
                    "slot": slot[:100],
                    "expected": expected[:120],
                    "queries": queries[:2],
                    "key_terms": key_terms,
                }
            )
    if not slots:
        slots = [
            {
                "slot": option_claim(state)[:100],
                "expected": "",
                "queries": [option_claim(state)[:160]],
                "key_terms": normalize_key_terms([], option_claim(state), max_terms=8),
            }
        ]
    slots = ensure_entity_coverage_slots(state, slots)
    slots = ensure_compound_reason_slots(state, slots)
    reason_slot_count = len(extract_claim_reason_parts(option_claim(state)))
    max_slots = 2
    if needs_entity_coverage_slots(state):
        max_slots = min(6, max(max_slots, len(atomic_entities_for_state(state))))
    if reason_slot_count > 1:
        max_slots = min(6, max(max_slots, 1 + reason_slot_count))
    option_state["slot_plan"] = plan
    option_state["slots"] = slots[:max_slots]
    option_state["status"] = "slot_planned"
    add_trace(state, f"{option}: slot plan {len(option_state['slots'])}")
    if VERBOSE_CONSOLE:
        print("[slot plan]", json.dumps(option_state["slots"], ensure_ascii=False, indent=2), flush=True)
    return state


def route_docs_node(state: AgentState) -> AgentState:
    option = state["current_option"]
    option_state = state["option_states"][option]
    hint_targets = deterministic_doc_hints(state)
    if DRY_RUN_WITHOUT_LLM:
        system, user = build_doc_router_prompt(state)
        router = dry_json(system, user, "router")
    else:
        system, user = build_doc_router_prompt(state)
        router, usage = call_qwen_json(system, user)
        add_usage(state, usage)
    targets = [doc_id for doc_id in router.get("target_doc_ids", []) if doc_id in state["doc_ids"]]
    if hint_targets:
        targets = list(dict.fromkeys(hint_targets + targets))
        router["pre_matched_doc_ids"] = hint_targets
        router["pre_match_scores"] = [
            {"doc_id": doc_id, "score": round(score, 3), "overlap": overlap}
            for score, doc_id, overlap in doc_hint_scores(option_claim(state), state.get("doc_ids") or [])[:4]
        ]
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
    slot_queries: list[str] = []
    for slot in option_state.get("slots") or []:
        if not isinstance(slot, dict):
            continue
        for query in slot.get("queries") or []:
            query = maybe_fix_mojibake(str(query or "")).strip()
            if query and query not in slot_queries:
                slot_queries.append(query)
    searches = option_state.get("searches") or []
    target_doc_ids = option_state.get("target_doc_ids") or state.get("doc_ids") or []
    query_parts = [option_claim(state), option_slot_query_text(option_state)]
    query_parts.extend(slot_queries[:4])
    for search in searches:
        if isinstance(search, dict):
            query_parts.append(str(search.get("query") or ""))
    retrieval_query = maybe_fix_mojibake(" ".join(part for part in query_parts if part)).strip()
    if searches:
        evidence = []
        for search in searches:
            evidence.extend(
                retrieve_multi_doc(search["query"], search["target_doc_ids"], min(INITIAL_TOP_K, 2), "initial")
            )
    elif slot_queries:
        evidence = []
        for query in slot_queries[:4]:
            evidence.extend(
                retrieve_multi_doc(query, target_doc_ids, min(INITIAL_TOP_K, 2), "initial")
            )
    else:
        query = f"{maybe_fix_mojibake(state['question']['question'])}\n{option}. {option_state['claim']}"
        retrieval_query = query
        evidence = retrieve_multi_doc(query, target_doc_ids, INITIAL_TOP_K, "initial")
    if (slot_queries or searches) and retrieval_query:
        evidence = merge_evidence(evidence, summary_candidate_evidence(target_doc_ids, retrieval_query))
        before_rerank_ids = [item["evidence_id"] for item in evidence]
        evidence = rerank_initial_evidence(evidence, retrieval_query)
        max_items = max(INITIAL_TOP_K + 2, min(10, len(target_doc_ids) * (INITIAL_TOP_K + 2)))
        evidence = evidence[:max_items]
        option_state["initial_rerank"] = {
            "query": retrieval_query[:500],
            "before_ids": before_rerank_ids,
            "after_ids": [item["evidence_id"] for item in evidence],
        }
    option_state["evidence"] = merge_evidence(option_state["evidence"], evidence)
    option_state["new_evidence_ids"] = [item["evidence_id"] for item in evidence]
    option_state["recent_evidence_ids"] = option_state["new_evidence_ids"]
    apply_local_subsearch_to_new_evidence(state, "initial")
    option_state["status"] = "retrieved"
    add_trace(state, f"{option}: initial retrieve {len(evidence)}")
    if VERBOSE_CONSOLE:
        if slot_queries:
            print("[slot queries]", json.dumps(slot_queries, ensure_ascii=False, indent=2), flush=True)
        if searches:
            print("[initial searches]", json.dumps(searches, ensure_ascii=False, indent=2), flush=True)
        print("[initial evidence]", flush=True)
        print_evidence_preview(evidence)
    return state


def build_evidence_read_prompt(state: AgentState) -> tuple[str, str]:
    option = state["current_option"]
    option_state = state["option_states"][option]
    system = (
        "你是证据阅读Agent。你的任务只是在候选证据中抽取与待验证槽位有关的事实。"
        "不要判断选项最终对错，不要决定是否丢弃证据，不要生成下一轮检索词。"
        "如果一个证据只能支持某个槽位的一部分，也要抽取出来，并在coverage写partial。"
        "事实必须来自证据原文、题干已知事实或基于证据数字的明确计算；禁止照抄选项声称当事实。"
        "只输出JSON。"
    )
    user = f"""
题干：{maybe_fix_mojibake(state["question"]["question"])}
当前选项：{option}. {option_claim(state)}

待验证槽位：
{json.dumps(option_state.get("slots", []), ensure_ascii=False, indent=2)}

已有压缩事实：
{format_memory_slots(option_state.get("memory_slots", []))}

本轮/近期证据：
{format_evidence(current_evidence_for_read(option_state), max_chars=3600)}

请输出：
{{
  "facts": [
    {{
      "slot": "该事实对应的槽位或子槽位",
      "value": "从证据中抽取的事实、数值、关系、公式或边界条件，最多180字",
      "coverage": "full|partial",
      "evidence_ids": ["证据ID"]
    }}
  ]
}}
要求：
1. 每条事实必须引用至少一个证据ID。
2. 能证明部分对象、部分指标、分子/分母之一、关系的一侧，也要输出coverage=partial。
3. 如果证据只含泛泛背景且完全不能支持任何槽位，facts输出空数组。
4. 最多输出6条facts，优先覆盖不同槽位和不同目标对象；不要重复输出已有压缩事实。
"""
    return system, user


def merge_evidence_facts(existing: list[dict], incoming: list[dict]) -> list[dict]:
    merged = list(existing or [])
    seen = {
        (
            str(item.get("slot") or ""),
            str(item.get("value") or ""),
            ",".join(item.get("evidence_ids") or []),
        )
        for item in merged
        if isinstance(item, dict)
    }
    for item in incoming or []:
        if not isinstance(item, dict):
            continue
        slot = maybe_fix_mojibake(str(item.get("slot") or "")).strip()
        value = maybe_fix_mojibake(str(item.get("value") or "")).strip()
        evidence_ids = normalize_evidence_id_list(item.get("evidence_ids") or [])
        if not slot or not value or not evidence_ids:
            continue
        compact = {
            "slot": slot[:100],
            "value": value[:220],
            "coverage": str(item.get("coverage") or "partial")[:20],
            "evidence_ids": evidence_ids[:5],
        }
        key = (compact["slot"], compact["value"], ",".join(compact["evidence_ids"]))
        if key in seen:
            continue
        seen.add(key)
        merged.append(compact)
    return merged[-24:]


def evidence_read_node(state: AgentState) -> AgentState:
    option = state["current_option"]
    option_state = state["option_states"][option]
    batch_ids = normalize_evidence_id_list(option_state.get("new_evidence_ids") or option_state.get("recent_evidence_ids") or [])
    if not batch_ids:
        option_state["status"] = "evidence_read_skipped"
        return state
    batch_key = "|".join(batch_ids)
    if batch_key in set(option_state.get("evidence_read_batches") or []):
        option_state["status"] = "evidence_read_already_done"
        return state
    batches = list(option_state.get("evidence_read_batches") or [])
    batches.append(batch_key)
    option_state["evidence_read_batches"] = batches[-12:]
    option_state["evidence_read_current_ids"] = batch_ids
    if DRY_RUN_WITHOUT_LLM:
        result = {"facts": []}
    else:
        system, user = build_evidence_read_prompt(state)
        result, usage = call_qwen_json(system, user)
        add_usage(state, usage)
    facts = result.get("facts") or []
    option_state["evidence_facts"] = merge_evidence_facts(option_state.get("evidence_facts", []), facts)
    option_state["evidence_read_current_ids"] = []
    option_state["status"] = "evidence_read"
    add_trace(state, f"{option}: evidence read {len(facts)} facts")
    if VERBOSE_CONSOLE:
        print("[evidence facts]", json.dumps(option_state.get("evidence_facts", []), ensure_ascii=False, indent=2), flush=True)
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
    if "missing_slots" not in audit:
        audit["missing_slots"] = audit.get("matching_slots") or audit.get("matched_slots") or []
    if "filled_slots" not in audit:
        audit["filled_slots"] = []
    audit = normalize_audit_slots(audit, option_state)
    option_state["recent_evidence_ids"] = option_state.get("new_evidence_ids", [])
    option_state["audit"] = audit
    option_state["memory_slots"] = merge_memory_slots(
        option_state.get("memory_slots", []),
        merge_evidence_facts(option_state.get("evidence_facts", []), filter_task_audit_slots(audit.get("filled_slots") or [])),
    )
    option_state["new_evidence_ids"] = []
    option_state["round"] = option_state.get("round", 0) + 1
    option_state["status"] = "ready_to_judge" if audit.get("can_judge") else "need_more_evidence"
    add_trace(state, f"{option}: audit can_judge={audit.get('can_judge')}")
    if VERBOSE_CONSOLE:
        public_audit = {key: audit.get(key) for key in ("can_judge", "filled_slots", "missing_slots")}
        print("[audit output]", json.dumps(public_audit, ensure_ascii=False, indent=2), flush=True)
        print("[memory slots]", json.dumps(option_state.get("memory_slots", []), ensure_ascii=False, indent=2), flush=True)
    return state


DEGREE_CLAIM_TERMS = ("主要", "集中", "重点", "核心", "首要", "唯一", "均", "都", "全部", "所有")
PARALLEL_RELATION_TERMS = (
    "两大",
    "两个",
    "多大",
    "多个",
    "多类",
    "并列",
    "之一",
    "一是",
    "二是",
    "三是",
    "同时",
    "分别",
)
NEGATIVE_RELATION_TERMS = ("并非", "不是", "不属于", "不一致", "相反", "无法支持")
INSUFFICIENT_EVIDENCE_TERMS = (
    "未包含",
    "未提及",
    "未找到",
    "没找到",
    "没有找到",
    "没有检索到",
    "未检索到",
    "证据不足",
    "缺少证据",
    "无法确定",
    "无法判断",
    "仅提及",
)


def same_slot_key(left: str, right: str) -> bool:
    left_key = compact_text(left)
    right_key = compact_text(right)
    if not left_key or not right_key:
        return False
    return left_key in right_key or right_key in left_key


def audit_has_decisive_contradiction(option_state: dict) -> bool:
    audit = option_state.get("audit") or {}
    missing_slots = audit.get("missing_slots") or []
    filled_slots = list(audit.get("filled_slots") or [])
    filled_slots.extend(option_state.get("memory_slots") or [])
    filled_slots.extend(option_state.get("evidence_facts") or [])
    if not missing_slots or not filled_slots:
        return False
    planned_slots = [
        maybe_fix_mojibake(str(item.get("slot") or ""))
        for item in option_state.get("slots") or []
        if isinstance(item, dict)
    ]
    direct_negative_terms = (
        "否",
        "不成立",
        "不支持",
        "不符合",
        "并非",
        "不是",
        "相反",
        "无法支持",
    )
    for filled in filled_slots:
        if not isinstance(filled, dict):
            continue
        filled_slot = maybe_fix_mojibake(str(filled.get("slot") or ""))
        value = maybe_fix_mojibake(str(filled.get("value") or "")).strip()
        if not filled_slot or not value:
            continue
        if planned_slots and not any(same_slot_key(filled_slot, planned) for planned in planned_slots):
            continue
        if any(term in value for term in INSUFFICIENT_EVIDENCE_TERMS):
            continue
        if value == "否" or any(term in value for term in direct_negative_terms[1:]):
            return True
    for missing in missing_slots:
        if not isinstance(missing, dict):
            continue
        missing_slot = maybe_fix_mojibake(str(missing.get("slot") or ""))
        if not missing_slot:
            continue
        for filled in filled_slots:
            if not isinstance(filled, dict):
                continue
            filled_slot = maybe_fix_mojibake(str(filled.get("slot") or ""))
            value = maybe_fix_mojibake(str(filled.get("value") or ""))
            if not same_slot_key(missing_slot, filled_slot):
                continue
            if any(term in value for term in INSUFFICIENT_EVIDENCE_TERMS):
                continue
            if any(term in value for term in NEGATIVE_RELATION_TERMS):
                return True
            has_degree_claim = any(term in missing_slot for term in DEGREE_CLAIM_TERMS)
            has_parallel_value = any(term in value for term in PARALLEL_RELATION_TERMS)
            if has_degree_claim and has_parallel_value:
                return True
    return False


def route_after_audit(state: AgentState) -> str:
    option = state["current_option"]
    option_state = state["option_states"][state["current_option"]]
    missing = option_state.get("audit", {}).get("missing_slots") or []
    if not missing and option_state.get("status") == "ready_to_judge":
        return "judge"
    if audit_has_decisive_contradiction(option_state):
        option_state["early_judge_reason"] = "audit_filled_slot_contradicts_missing_core_claim"
        add_trace(state, f"{option}: early judge on decisive contradiction")
        return "judge"
    if missing and has_doc_reroute_candidate(state):
        add_trace(state, f"{option}: doc reroute before local repair")
        return "retrieve_more"
    if should_review_original_after_subsearch(state):
        return "review_original_after_subsearch"
    if should_inner_focus(state):
        return "focus_evidence"
    if should_toc_repair(state):
        return "toc_repair"
    if option_state.get("round", 0) >= MAX_AUDIT_ROUNDS:
        return "judge"
    return "retrieve_more" if missing else "judge"


def should_review_original_after_subsearch(state: AgentState) -> bool:
    option_state = state["option_states"][state["current_option"]]
    audit = option_state.get("audit") or {}
    if not (audit.get("missing_slots") or []):
        return False
    local_subsearch = option_state.get("local_subsearch") or {}
    source_ids = [evidence_id for evidence_id in local_subsearch.get("source_evidence_ids") or [] if evidence_id]
    focused_ids = set(local_subsearch.get("focused_evidence_ids") or [])
    if not source_ids or not focused_ids:
        return False
    batch_key = "|".join(source_ids)
    if batch_key in set(option_state.get("original_review_after_subsearch_batches") or []):
        return False
    evidence_by_id = {item.get("evidence_id"): item for item in option_state.get("evidence", []) or []}
    return any(evidence_id in evidence_by_id for evidence_id in source_ids)


def filter_review_original_source_ids(option_state: dict, source_ids: list[str]) -> list[str]:
    source_ids = list(dict.fromkeys(source_ids))
    if len(source_ids) <= REVIEW_ORIGINAL_MAX_SOURCE_IDS:
        return source_ids
    missing = option_state.get("audit", {}).get("missing_slots") or []
    query = missing_query_text(missing)
    key_terms: list[str] = []
    for item in missing:
        if isinstance(item, dict):
            for term in normalize_key_terms(
                item.get("key_terms") or [],
                " ".join([str(item.get("slot") or ""), str(item.get("followup_query") or item.get("query") or "")]),
                max_terms=8,
            ):
                if term not in key_terms:
                    key_terms.append(term)
    query_tokens = {token for token in tokenize(query) if len(token) >= 2}
    evidence_by_id = {item.get("evidence_id"): item for item in option_state.get("evidence", []) or []}
    remembered = remembered_evidence_ids(option_state)
    scored: list[tuple[float, int, str]] = []
    for idx, evidence_id in enumerate(source_ids):
        item = evidence_by_id.get(evidence_id)
        if not item:
            continue
        heading = maybe_fix_mojibake(" > ".join(item.get("heading_path") or []))
        text = maybe_fix_mojibake(compact_html_text(str(item.get("text") or "")))
        searchable = f"{heading}\n{text}"
        score = 0.0
        score += sum(4.0 for term in key_terms if term and term in searchable)
        score += len(query_tokens & set(tokenize(searchable))) * 1.0
        score += len(query_tokens & set(tokenize(heading))) * 1.5
        if evidence_id in remembered:
            score += 2.0
        if re.search(r"(?:图|表)\s*[\d一二三四五六七八九十]+[：:]|\|.+\|", searchable):
            score += 0.5
        if score > 0:
            scored.append((score, -idx, evidence_id))
    if not scored:
        return source_ids[:REVIEW_ORIGINAL_MAX_SOURCE_IDS]
    scored.sort(reverse=True)
    return [evidence_id for _score, _neg_idx, evidence_id in scored[:REVIEW_ORIGINAL_MAX_SOURCE_IDS]]


def review_original_after_subsearch_node(state: AgentState) -> AgentState:
    option = state["current_option"]
    option_state = state["option_states"][option]
    local_subsearch = option_state.get("local_subsearch") or {}
    evidence_by_id = {item.get("evidence_id"): item for item in option_state.get("evidence", []) or []}
    source_ids = [
        evidence_id
        for evidence_id in local_subsearch.get("source_evidence_ids") or []
        if evidence_id in evidence_by_id
    ]
    original_source_count = len(source_ids)
    original_batch_key = "|".join(source_ids)
    source_ids = filter_review_original_source_ids(option_state, source_ids)
    batches = list(option_state.get("original_review_after_subsearch_batches") or [])
    if original_batch_key and original_batch_key not in batches:
        batches.append(original_batch_key)
    option_state["original_review_after_subsearch_batches"] = batches
    option_state["original_review_after_subsearch_used"] = True
    option_state["review_original_filter"] = {
        "before_count": original_source_count,
        "after_count": len(source_ids),
        "source_evidence_ids": source_ids,
    }
    option_state["new_evidence_ids"] = source_ids
    option_state["recent_evidence_ids"] = source_ids
    option_state["status"] = "review_original_after_subsearch"
    add_trace(state, f"{option}: review original after subsearch {len(source_ids)}/{original_source_count}")
    if VERBOSE_CONSOLE and source_ids:
        print("[review original after subsearch]", source_ids, flush=True)
    return state


def should_inner_focus(state: AgentState) -> bool:
    option_state = state["option_states"][state["current_option"]]
    audit = option_state.get("audit") or {}
    missing = audit.get("missing_slots") or []
    if not missing:
        return False
    if option_state.get("inner_focus_used"):
        return False
    return any(len(compact_text(item.get("text", ""))) > INNER_FOCUS_MIN_CHARS for item in prioritize_evidence_for_review(option_state))


def focus_evidence_node(state: AgentState) -> AgentState:
    option = state["current_option"]
    option_state = state["option_states"][option]
    missing = option_state.get("audit", {}).get("missing_slots") or []
    focused = build_inner_focus_evidence(option_state, missing)
    option_state["inner_focus_used"] = True
    if focused:
        option_state["evidence"] = merge_evidence(option_state.get("evidence", []), focused)
        option_state["new_evidence_ids"] = [item["evidence_id"] for item in focused]
        option_state["recent_evidence_ids"] = option_state["new_evidence_ids"]
        option_state["inner_focus"] = {
            "missing_slots": missing,
            "focused_evidence_ids": option_state["new_evidence_ids"],
        }
    option_state["status"] = "inner_focused" if focused else "inner_focus_empty"
    add_trace(state, f"{option}: inner focus {len(focused)}")
    if VERBOSE_CONSOLE and focused:
        print("[inner focus evidence]", flush=True)
        print_evidence_preview(focused)
    return state


def should_extract_evidence(option_state: dict) -> bool:
    if option_state.get("extract_attempted_for") == option_state.get("new_evidence_ids"):
        return False
    evidence_by_id = {item.get("evidence_id"): item for item in option_state.get("evidence", []) or []}
    for evidence_id in option_state.get("new_evidence_ids") or []:
        item = evidence_by_id.get(evidence_id)
        if item and item.get("source") != "evidence_extract" and len(compact_text(item.get("text", ""))) > INNER_FOCUS_MIN_CHARS:
            return True
    return False


def build_evidence_extract_prompt(state: AgentState) -> tuple[str, str]:
    option = state["current_option"]
    option_state = state["option_states"][option]
    evidence_by_id = {item.get("evidence_id"): item for item in option_state.get("evidence", []) or []}
    evidence_items = []
    for evidence_id in option_state.get("new_evidence_ids") or []:
        item = evidence_by_id.get(evidence_id)
        if not item or item.get("source") == "evidence_extract":
            continue
        text = maybe_fix_mojibake(item.get("text", ""))
        if len(compact_text(text)) <= INNER_FOCUS_MIN_CHARS:
            continue
        evidence_items.append(
            {
                "evidence_id": evidence_id,
                "doc_id": item.get("doc_id"),
                "heading": " > ".join(item.get("heading_path") or []),
                "text": text[:9000],
            }
        )
    system = (
        "你是块内证据摘录Agent。你只能从给定证据块原文中摘录能验证当前选项的句子、短段落或表格行。"
        "不要解释、不要改写、不要推理、不要补充原文没有的内容。只输出JSON。"
    )
    user = f"""
题干：{maybe_fix_mojibake(state["question"]["question"])}
当前选项：{option}. {option_claim(state)}

上一轮缺失槽位：
{json.dumps(option_state.get("audit", {}).get("missing_slots") or [], ensure_ascii=False)}

证据块：
{json.dumps(evidence_items, ensure_ascii=False, indent=2)}

请输出：
{{
  "extractions": [
    {{"source_evidence_id": "原证据ID", "text": "从原文逐字摘录的相关句子/表格行，最多500字", "reason": "对应哪个事实槽位"}}
  ]
}}
要求：
1. text 必须是原文连续片段或表格的原始行组合，不得改写。
2. 优先摘录同时包含选项中的实体、年份、指标、数值的句子。
3. 正文中已有精确数值时，优先摘正文句子；正文没有时再摘表格行。
4. 最多输出4条；如果没有相关原文，输出空数组。
"""
    return system, user


def evidence_extract_node(state: AgentState) -> AgentState:
    option = state["current_option"]
    option_state = state["option_states"][option]
    original_new_ids = list(option_state.get("new_evidence_ids") or [])
    if not should_extract_evidence(option_state):
        option_state["status"] = "extract_skipped"
        return state
    option_state["extract_attempted_for"] = original_new_ids
    if DRY_RUN_WITHOUT_LLM:
        result = {"extractions": []}
    else:
        system, user = build_evidence_extract_prompt(state)
        result, usage = call_qwen_json(system, user)
        add_usage(state, usage)
    evidence_by_id = {item.get("evidence_id"): item for item in option_state.get("evidence", []) or []}
    extracted: list[dict] = []
    for idx, item in enumerate(result.get("extractions") or []):
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_evidence_id") or "").strip()
        source = evidence_by_id.get(source_id)
        text = maybe_fix_mojibake(str(item.get("text") or "")).strip()
        if not source or not text:
            continue
        source_text = maybe_fix_mojibake(source.get("text", ""))
        if text not in source_text:
            text = compact_text(text)
            if text not in compact_text(source_text):
                continue
        extracted.append(
            {
                "evidence_id": f"{source_id}_extract_{idx}",
                "doc_id": source.get("doc_id"),
                "page_id": f"{source_id}_extract_{idx}",
                "heading_path": source.get("heading_path") or [],
                "text": text[:700],
                "score": source.get("score", 0.0),
                "source": "evidence_extract",
                "focused_from": source_id,
                "extract_reason": str(item.get("reason") or "")[:120],
            }
        )
    if extracted:
        option_state["evidence"] = merge_evidence(option_state.get("evidence", []), extracted)
        option_state["new_evidence_ids"] = [item["evidence_id"] for item in extracted]
        option_state["recent_evidence_ids"] = option_state["new_evidence_ids"]
        option_state["evidence_extract"] = {
            "source_evidence_ids": original_new_ids,
            "extracted_evidence_ids": [item["evidence_id"] for item in extracted],
        }
    else:
        option_state["new_evidence_ids"] = original_new_ids
    option_state["status"] = "evidence_extracted" if extracted else "extract_empty"
    add_trace(state, f"{option}: evidence extract {len(extracted)}")
    if VERBOSE_CONSOLE and extracted:
        print("[evidence extract]", flush=True)
        print_evidence_preview(extracted)
    return state


def retrieve_more_node(state: AgentState) -> AgentState:
    option = state["current_option"]
    option_state = state["option_states"][option]
    missing = option_state.get("audit", {}).get("missing_slots") or []
    for item in missing:
        if isinstance(item, dict):
            query = item.get("followup_query") or item.get("query") or item.get("slot") or ""
            slot_text = maybe_fix_mojibake(str(item.get("slot") or query)).strip()
            key_terms = normalize_key_terms(item.get("key_terms") or [], " ".join([slot_text, str(query)]), max_terms=10)
            target_doc_ids = [doc_id for doc_id in item.get("target_doc_ids", []) if doc_id in state["doc_ids"]]
        else:
            query = str(item)
            slot_text = query
            key_terms = normalize_key_terms([], query, max_terms=10)
            target_doc_ids = []
        if not target_doc_ids:
            target_doc_ids = option_state.get("target_doc_ids") or state["doc_ids"]
        added_doc_ids = reroute_docs_for_missing_slot(state, item, option_state)
        if added_doc_ids:
            target_doc_ids = list(dict.fromkeys(list(target_doc_ids) + added_doc_ids))
        if not query:
            continue
        followup_queries = [query]
        for extra_query in computed_ratio_followup_queries(query):
            if extra_query not in followup_queries:
                followup_queries.append(extra_query)
        evidence: list[dict] = []
        for idx, followup_query in enumerate(followup_queries):
            source = "followup" if idx == 0 else "followup_computed_part"
            evidence = merge_evidence(
                evidence,
                retrieve_multi_doc(followup_query, target_doc_ids, FOLLOWUP_TOP_K, source),
            )
        coverage_evidence = retrieve_by_key_term_coverage(target_doc_ids, key_terms, "followup_key_coverage")
        if coverage_evidence:
            evidence = merge_evidence(evidence, coverage_evidence)
        ratio_hint_evidence = build_computed_ratio_hint_evidence(option_state, query, evidence)
        if ratio_hint_evidence:
            evidence = merge_evidence(ratio_hint_evidence, evidence)
        before_rerank_ids = [item["evidence_id"] for item in evidence]
        max_followup_items = max(FOLLOWUP_TOP_K + 2, min(8, len(target_doc_ids) * (FOLLOWUP_TOP_K + 1)))
        evidence = rerank_evidence_for_slot(
            evidence,
            slot_text,
            query,
            key_terms=key_terms,
            max_items=max_followup_items,
        )
        option_state.setdefault("followup_rerank", []).append(
            {
                "slot": slot_text[:160],
                "query": str(query)[:160],
                "key_terms": key_terms,
                "coverage_ids": [item["evidence_id"] for item in coverage_evidence],
                "before_ids": before_rerank_ids,
                "after_ids": [item["evidence_id"] for item in evidence],
            }
        )
        option_state["evidence"] = merge_evidence(option_state["evidence"], evidence)
        option_state["new_evidence_ids"] = list(
            dict.fromkeys(option_state.get("new_evidence_ids", []) + [item["evidence_id"] for item in evidence])
        )
        option_state["recent_evidence_ids"] = option_state["new_evidence_ids"]
        apply_local_subsearch_to_new_evidence(state, "followup")
        if VERBOSE_CONSOLE:
            print("[followup query]", query, flush=True)
            print("[followup key_terms]", json.dumps(key_terms, ensure_ascii=False), flush=True)
            if len(followup_queries) > 1:
                print("[followup computed queries]", json.dumps(followup_queries[1:], ensure_ascii=False), flush=True)
            print("[followup target_doc_ids]", target_doc_ids, flush=True)
            if added_doc_ids:
                print("[doc reroute added]", added_doc_ids, flush=True)
            print_evidence_preview(evidence)
    option_state["status"] = "retrieved_more"
    add_trace(state, f"{option}: followup retrieve {len(missing)} missing slots")
    return state


def toc_repair_node(state: AgentState) -> AgentState:
    option = state["current_option"]
    option_state = state["option_states"][option]
    if DRY_RUN_WITHOUT_LLM:
        plan = {"section_queries": []}
    else:
        system, user = build_toc_repair_prompt(state)
        plan, usage = call_qwen_json(system, user)
        add_usage(state, usage)
    section_queries = plan.get("section_queries") or []
    evidence: list[dict] = []
    audit_key_terms: list[str] = []
    for missing_item in option_state.get("audit", {}).get("missing_slots") or []:
        if isinstance(missing_item, dict):
            for term in normalize_key_terms(
                missing_item.get("key_terms") or [],
                " ".join([str(missing_item.get("slot") or ""), str(missing_item.get("followup_query") or "")]),
            ):
                if term not in audit_key_terms:
                    audit_key_terms.append(term)
    for item in section_queries[:3]:
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("doc_id") or "").strip()
        if doc_id not in state.get("doc_ids", []):
            continue
        section_hint = maybe_fix_mojibake(str(item.get("section_hint") or "")).strip()
        query = maybe_fix_mojibake(str(item.get("query") or section_hint).strip())
        if not section_hint and not query:
            continue
        section_evidence = retrieve_by_toc_hint(doc_id, section_hint or query, query, max_items=3)
        key_terms = normalize_key_terms(
            item.get("key_terms") or audit_key_terms,
            " ".join([section_hint, query]),
            max_terms=10,
        )
        before_rerank_ids = [candidate["evidence_id"] for candidate in section_evidence]
        section_evidence = rerank_evidence_for_slot(
            section_evidence,
            section_hint,
            query,
            key_terms=key_terms,
            max_items=3,
        )
        option_state.setdefault("toc_rerank", []).append(
            {
                "doc_id": doc_id,
                "section_hint": section_hint[:160],
                "query": query[:160],
                "key_terms": key_terms,
                "before_ids": before_rerank_ids,
                "after_ids": [candidate["evidence_id"] for candidate in section_evidence],
            }
        )
        evidence.extend(section_evidence)
    option_state["evidence"] = merge_evidence(option_state.get("evidence", []), evidence)
    option_state["new_evidence_ids"] = [item["evidence_id"] for item in evidence]
    option_state["recent_evidence_ids"] = option_state["new_evidence_ids"]
    apply_local_subsearch_to_new_evidence(state, "toc_repair")
    option_state["toc_repair"] = plan
    option_state["toc_repair_used"] = True
    option_state["status"] = "toc_repaired"
    add_trace(state, f"{option}: toc repair retrieve {len(evidence)}")
    if VERBOSE_CONSOLE:
        print("[toc repair plan]", json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        print("[toc repair evidence]", flush=True)
        print_evidence_preview(evidence)
    return state


def normalize_judgment(judgment: dict, option: str, answer_format: str) -> dict:
    answer = str(judgment.get("answer") or "").strip().upper()
    if answer_format == "tf":
        if answer in {"T", "TRUE", "Y", "YES", "A"}:
            judgment["verdict"] = True
            judgment["answer"] = "T"
        elif answer in {"F", "FALSE", "N", "NO", "B"}:
            judgment["verdict"] = False
            judgment["answer"] = "F"
        elif judgment.get("verdict") is True:
            judgment["answer"] = "T"
        elif judgment.get("verdict") is False:
            judgment["answer"] = "F"
    else:
        if answer in {"F", "FALSE", "N", "NO"}:
            judgment["verdict"] = False
            judgment["answer"] = ""
        elif answer in {"T", "TRUE", "Y", "YES"}:
            judgment["verdict"] = True
            judgment["answer"] = option
        elif answer in "ABCD":
            judgment["verdict"] = answer == option
            judgment["answer"] = option if answer == option else ""
        elif judgment.get("verdict") is False:
            judgment["answer"] = ""
        elif judgment.get("verdict") is True:
            judgment["answer"] = option
    return judgment


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
    judgment = normalize_judgment(judgment, option, state["question"].get("answer_format", ""))
    judgment = enforce_judgment_missing_slot_guard(
        judgment,
        option_state,
        option,
        state["question"].get("answer_format", ""),
    )
    option_state["judgment"] = judgment
    option_state["status"] = judgment.get("status", "done")
    add_trace(state, f"{option}: judge verdict={judgment.get('verdict')}")
    if VERBOSE_CONSOLE:
        public_judgment = {key: judgment.get(key) for key in ("verdict", "answer", "confidence", "evidence_ids")}
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
        add_trace(state, "single-answer fallback blocked: no true option")
        return ""

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


def split_option_claim_parts(text: str) -> list[str]:
    text = maybe_fix_mojibake(str(text or "")).strip()
    if not text:
        return []
    parts = re.split(r"[；;。\n]+", text)
    cleaned: list[str] = []
    for part in parts:
        part = re.sub(r"\s+", " ", part).strip(" ，,、")
        if part:
            cleaned.append(part)
    return cleaned or [text]


def extract_claim_amounts(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\d+(?:\.\d+)?\s*(?:亿元|万元|元|%|倍|万|亿|人|天|年)", text or "")))


def classify_claim_polarity(text: str) -> str:
    if re.search(r"不予赔|不赔付|不赔|均不赔|无法赔|不承担|不属于", text or ""):
        return "negative/no_pay"
    if re.search(r"赔付|赔偿|给付|赔\d|赔\s*\d", text or ""):
        return "positive/pay_or_amount"
    if re.search(r"高于|低于|大于|小于|超过|不超过|排序|从高到低|从低到高|>", text or ""):
        return "comparison/ranking"
    return "statement"


def build_option_claims(question: dict) -> list[dict]:
    claims: list[dict] = []
    for option, raw in (question.get("options") or {}).items():
        raw_text = maybe_fix_mojibake(raw)
        parts = split_option_claim_parts(raw_text)
        claims.append({
            "option": option,
            "raw_text": raw_text,
            "claims": [
                {
                    "claim_index": index,
                    "raw_claim": part,
                    "amounts": extract_claim_amounts(part),
                    "polarity": classify_claim_polarity(part),
                }
                for index, part in enumerate(parts, start=1)
            ],
        })
    return claims


def compact_evidence_notes_for_final(notes: list[dict], max_notes: int = 6) -> list[dict]:
    compact: list[dict] = []
    for note in notes[-max_notes:]:
        compact.append({
            "source_evidence_id": note.get("source_evidence_id"),
            "doc_id": note.get("doc_id"),
            "heading": note.get("heading"),
            "note": maybe_fix_mojibake(str(note.get("note") or ""))[:360],
        })
    return compact

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
    evidence_notes: list[dict] | None = None,
) -> tuple[str, str]:
    system = (
        "你是单任务证据审查Agent。只判断当前任务是否已经具备计算/判断所需事实。"
        "题干已知事实可以直接使用；证据笔记和当前证据原文用于补齐条款规则、公式、比例、定义。"
        "证据笔记是从原文证据块中抽取的关键信息，必须保留其source_evidence_id作为溯源。"
        "若证据笔记已足够支持判断，优先使用证据笔记；当前证据原文只用于核对和补充。"
        "已确认事实中的槽位不要重复验证；当前轮只需要判断新证据能否补齐缺失槽位或修正冲突。"
        "对于是否属于保险责任/是否赔付类任务，如果证据完整列举了可赔责任、费用类型或适用场景，而题干事项不在列举范围内，可以作为反向证据判断不属于责任范围；不必必须找到原文直接写不赔。"
        "当任务首先需要判断是否赔付/是否属于责任范围时，封闭列举的责任范围或费用范围就是关键证据；若题干事项不在范围内，can_solve=true，filled_slots写明不属于范围及依据，不要继续把免赔额、赔付比例、限额或计算公式列为missing_slots。"
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

证据笔记：
{format_evidence_notes(evidence_notes or [])}

当前证据原文（仅最近/必要片段，用于核对笔记；若笔记为空则为候选原文）：
{format_evidence(evidence[-2:] if evidence_notes else evidence, max_chars=1600 if evidence_notes else 5000)}

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
        "赔付/给付/保险责任类任务应先判断题干事故、费用或责任项目是否属于条款列明的保险责任/费用范围；若已确认不属于范围，直接给出不赔/0，并说明依据，不要再因免赔额、比例或公式缺失而blocked。"
        "禁止根据选项倒推事实，禁止使用选项中的金额修正计算。"
        "计算题必须逐项写出公式、代入和算术过程；百分比要先换算成小数或明确乘法。"
        "如果当前任务是某产品/合同/方案的赔付、给付、退保或收益总额，且题干包含多人、多次、多项费用或多个责任项目，result必须给出该任务口径下的总额，并在calculation中列出分项；不要只把第一个人或第一项的金额作为result。"
        "遇到家庭共享免赔额、共享限额、累计抵扣等规则时，除非题干或条款明确给出按人分摊/按比例分配方法，否则不得自行把总赔付额按个人费用占比分配；任务问产品赔付总额时直接输出产品口径总额。"
        "如果必须计算某个人/某一项金额但缺少分配规则，应status=blocked并写入missing_slots，不能用常见理解、通常理解或比例占比自行补齐。"
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
    option_claims = build_option_claims(state["question"])
    answer_format = str(state["question"].get("answer_format") or "").strip().lower()
    if answer_format == "free":
        answer_instruction = '"answer": "按题干要求填写最终答案；只填数字、日期、排序或文本本身，不要输出选项字母或解释"'
    else:
        answer_instruction = '"answer": "A/B/C/D"'
    system = (
        "你是最终作答Agent。只基于题干和各任务结果计算、排序并匹配选项。"
        "选项只能用于最后匹配，不能用于倒推缺失规则，不能用选项金额修正任务结果。"
        "匹配每个选项时必须逐项核对选项文本中的所有明确金额、赔付/不赔结论、排序关系和对象归属；只要任一明确声明与任务结果冲突，该选项必须is_match=false，不能因为其他部分接近或可排除其他选项而判true。"
        "必须以option_claims.raw_text和option_claims.claims为准逐条核对；不得省略、改写、合并或反向解释选项中的任一claim。"
        "任务结果中的evidence_notes是压缩后的证据依据；当任务result或basis过短时，必须用evidence_notes核对该任务结论，但仍不得用证据改写选项原文。"
        "如果所有选项都与已确认任务结果冲突，应在missing_impact说明无完全匹配或题目/选项可能异常，不得把某个冲突较少的选项改写后判为true。"
        "如果任务结果中有formula/substitution/calculation，优先核对这些结构化计算；匹配选项时以任务result代表的任务口径总额为准，不能把calculation中的分项金额误当作该任务总额。"
        "若任务result同时列出总额和分项，最终匹配应使用总额；只有题干明确问某个人/某一项时才使用对应分项。"
        "如果任务结果内部自相矛盾，要在missing_impact中说明冲突，不要直接用选项补救。"
        "对于status=blocked的任务，不得编造缺失金额、比例、公式或规则；只有在其他已完成任务和明确证据足以排除选项时，才能说明排除逻辑。"
        "只输出JSON。"
    )
    user = f"""
题干：{maybe_fix_mojibake(state["question"]["question"])}

选项原文：
{reasoning_options_text(state["question"])}

选项claim拆解（逐条核对这些claim，不得改写）：
{json.dumps(option_claims, ensure_ascii=False, indent=2)}

任务结果：
{json.dumps(task_results, ensure_ascii=False, indent=2)}

请输出：
{{
  "calculations": [{{"item": "对象", "amount": "金额/结论", "basis": "依据"}}],
  "option_checks": [{{"option": "A/B/C/D", "is_match": true, "claim_checks": [{{"claim_index": 1, "raw_claim": "选项原文片段", "task_result": "对应任务结果", "match": true, "conflict_reason": "无或冲突原因"}}], "reason": "简短原因"}}],
  "missing_impact": "无或说明缺失影响",
  {answer_instruction}
}}
"""
    return system, user


CHINESE_DIGIT_MAP = {
    "〇": "0",
    "零": "0",
    "一": "1",
    "二": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
}


def chinese_number_to_int(text: str) -> int | None:
    text = maybe_fix_mojibake(str(text or "")).strip()
    if not text:
        return None
    if re.fullmatch(r"\d+", text):
        return int(text)
    if all(char in CHINESE_DIGIT_MAP for char in text):
        return int("".join(CHINESE_DIGIT_MAP[char] for char in text))
    if text == "十":
        return 10
    if "十" in text:
        left, _, right = text.partition("十")
        tens = chinese_number_to_int(left) if left else 1
        ones = chinese_number_to_int(right) if right else 0
        if tens is None or ones is None:
            return None
        return tens * 10 + ones
    return None


def parse_date_value(value: str) -> tuple[int, int, int | None] | None:
    text = maybe_fix_mojibake(str(value or ""))
    text = text.replace(" ", "")
    match = re.search(r"((?:19|20)\d{2})年(\d{1,2})月(?:(\d{1,2})日)?", text)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3)) if match.group(3) else None
        return year, month, day
    match = re.search(r"([一二三四五六七八九〇零]{4})年([一二三四五六七八九十]{1,3})月(?:([一二三四五六七八九十]{1,3})日)?", text)
    if match:
        year = chinese_number_to_int(match.group(1))
        month = chinese_number_to_int(match.group(2))
        day = chinese_number_to_int(match.group(3)) if match.group(3) else None
        if year and month:
            return year, month, day
    return None


def compare_date_values(left: tuple[int, int, int | None], right: tuple[int, int, int | None]) -> int | None:
    if left[:2] != right[:2]:
        return 1 if left[:2] > right[:2] else -1
    if left[2] is None or right[2] is None:
        return None
    if left[2] == right[2]:
        return 0
    return 1 if left[2] > right[2] else -1


def task_result_date_by_doc_ordinal(task_results: list[dict]) -> dict[str, tuple[int, int, int | None]]:
    result: dict[str, tuple[int, int, int | None]] = {}
    for item in task_results:
        if not isinstance(item, dict):
            continue
        surface = maybe_fix_mojibake(
            " ".join(
                str(part or "")
                for part in (
                    item.get("task"),
                    item.get("result"),
                    item.get("basis"),
                    item.get("calculation"),
                    item.get("substitution"),
                )
            )
        )
        date_value = None
        for value in (item.get("result"), item.get("amount"), item.get("basis"), surface):
            date_value = parse_date_value(str(value or ""))
            if date_value:
                break
        if not date_value:
            continue
        if "第一份" in surface or "第一" in surface:
            result.setdefault("第一", date_value)
        if "第二份" in surface or "第二" in surface:
            result.setdefault("第二", date_value)
    return result


def deterministic_time_comparison_answer(state: AgentState, task_results: list[dict]) -> str:
    question = maybe_fix_mojibake(str(state.get("question", {}).get("question") or ""))
    if not any(term in question for term in ("晚于", "早于", "不晚于", "不早于")):
        return ""
    if not ("第一" in question and "第二" in question):
        return ""
    dates = task_result_date_by_doc_ordinal(task_results)
    first = dates.get("第一")
    second = dates.get("第二")
    if not first or not second:
        return ""
    cmp_result = compare_date_values(second, first)
    if cmp_result is None:
        return ""
    if "第二" in question and "第一" in question:
        if "不晚于" in question:
            verdict = cmp_result <= 0
        elif "不早于" in question:
            verdict = cmp_result >= 0
        elif "晚于" in question:
            verdict = cmp_result > 0
        elif "早于" in question:
            verdict = cmp_result < 0
        else:
            return ""
        return chr(0x662F) if verdict else chr(0x5426)
    return ""


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


def title_is_avoided(title: str, failed_titles: list[str]) -> bool:
    title_key = normalize_heading_key(title)
    if not title_key:
        return False
    for failed in failed_titles or []:
        failed_key = normalize_heading_key(failed)
        if failed_key and (title_key in failed_key or failed_key in title_key):
            return True
    return False


def pop_next_untried_candidate(
    section_candidate_queue: list[dict],
    start_index: int,
    failed_titles: list[str],
    tried_section_ids: set[str],
) -> tuple[dict | None, int]:
    index = start_index
    while index < len(section_candidate_queue):
        item = section_candidate_queue[index]
        index += 1
        candidate = item.get("candidate") or {}
        section_id = str(candidate.get("section_id") or "")
        title = str(candidate.get("title") or "")
        if section_id and section_id in tried_section_ids:
            continue
        if title_is_avoided(title, failed_titles):
            continue
        if section_id:
            tried_section_ids.add(section_id)
        return item, index
    return None, index


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
    failed_titles = list(dict.fromkeys(failed_titles or []))[:8]
    used_queries = list(dict.fromkeys(used_queries or []))[:8]
    selectable_sections = [
        sec for sec in child_sections if not title_is_avoided(section_title(sec), failed_titles)
    ]
    if selectable_sections:
        child_sections = selectable_sections
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
    combined = merge_evidence(combined, pointer_expanded)
    auto_expanded = expand_auto_anchor_evidence(combined, forward=2)
    for item in auto_expanded:
        item["section_route_steps"] = route_steps
    return merge_evidence(combined, auto_expanded)


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
        recent_evidence: list[dict] = []
        task_evidence_notes: list[dict] = []
        noted_evidence_ids: set[str] = set()
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
        tried_section_ids: set[str] = set()
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
            section_id = str((item.get("candidate") or {}).get("section_id") or "")
            if section_id:
                tried_section_ids.add(section_id)
            for query in item["candidate"].get("search_queries") or []:
                if query not in used_queries:
                    used_queries.append(query)
            recent_evidence = retrieve_section_candidate(
                item["candidate"],
                item["doc_id"],
                "human_toc",
                item["plan"],
                next_candidate_index,
            )
            task_evidence = merge_evidence(task_evidence, recent_evidence)
            task_evidence_notes = update_task_evidence_notes(
                state, task, recent_evidence, task_evidence_notes, noted_evidence_ids
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
                        recent_evidence = evidence
                        task_evidence = merge_evidence(task_evidence, recent_evidence)
                        task_evidence_notes = update_task_evidence_notes(
                            state, task, recent_evidence, task_evidence_notes, noted_evidence_ids
                        )
        audit: dict = {}
        for round_index in range(1, MAX_REASONING_AUDIT_ROUNDS + 1):
            audit_evidence = recent_evidence or task_evidence[-2:]
            task_evidence_notes = update_task_evidence_notes(
                state, task, audit_evidence, task_evidence_notes, noted_evidence_ids
            )
            system, user = build_task_reasoning_audit_prompt(
                state, task, task_memory, audit_evidence, task_evidence_notes
            )
            if DRY_RUN_WITHOUT_LLM:
                audit = dry_json(system, user, "audit")
            else:
                audit, usage = call_qwen_json(system, user)
                add_usage(state, usage)
            audit = enforce_reference_missing_slots(audit, task_evidence, task)
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
                "evidence_notes": task_evidence_notes,
                "route_steps": collect_route_steps_from_evidence(task_evidence),
                "audit_evidence": audit_evidence,
                "evidence": task_evidence,
            }
            task_log["rounds"].append(round_log)
            if audit.get("can_solve") and audit.get("missing_slots"):
                audit["can_solve"] = False
                round_log["audit"] = audit
                round_log["can_solve_overridden"] = "missing_slots_present"
            if audit.get("can_solve"):
                break
            if round_index >= MAX_REASONING_AUDIT_ROUNDS:
                break
            if audit.get("expand_evidence_ids"):
                expanded = expand_forward_evidence_ids(task_evidence, audit.get("expand_evidence_ids") or [], forward=4)
                recent_evidence = expanded
                task_evidence = merge_evidence(task_evidence, recent_evidence)
                task_evidence_notes = update_task_evidence_notes(
                    state, task, recent_evidence, task_evidence_notes, noted_evidence_ids
                )
                round_log["anchor_expanded"] = expanded
            missing = []
            for item in audit.get("missing_slots") or []:
                if not isinstance(item, dict):
                    continue
                reference_targets = extract_reference_targets_from_evidence(task_evidence)
                missing_item = {
                    "slot": maybe_fix_mojibake(str(item.get("slot") or "")).strip(),
                    "target_doc_ids": [doc_id for doc_id in item.get("target_doc_ids", []) if doc_id in state.get("doc_ids", [])] or task.get("target_doc_ids", []),
                    "query": maybe_fix_mojibake(str(item.get("followup_query") or item.get("query") or item.get("slot") or "")).strip(),
                }
                if reference_targets:
                    missing_item["reference_targets"] = reference_targets
                missing.append(missing_item)
            if not missing:
                break
            followup_evidence: list[dict] = []
            for missing_item in missing[:1]:
                reference_evidence = retrieve_reference_followup(
                    state,
                    task,
                    missing_item,
                    task_evidence,
                    "human_toc_followup",
                )
                if reference_evidence:
                    followup_evidence = merge_evidence(followup_evidence, reference_evidence)
                    missing_item["reference_queries"] = reference_followup_queries(state, task, missing_item, task_evidence)
                    break
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
            prefer_existing_candidate = bool(failure_analysis) and any(
                term in maybe_fix_mojibake(str(failure_analysis.get("next_action") or ""))
                for term in ("回目录", "换章节", "换章", "其他章节", "下一")
            )
            if (not followup_evidence) and prefer_existing_candidate:
                item, next_candidate_index = pop_next_untried_candidate(
                    section_candidate_queue,
                    next_candidate_index,
                    failed_titles,
                    tried_section_ids,
                )
                if item:
                    for query in item["candidate"].get("search_queries") or []:
                        if query not in used_queries:
                            used_queries.append(query)
                    tried_candidates.append(
                        {
                            "doc_id": item["doc_id"],
                            "section_id": item["candidate"].get("section_id"),
                            "title": item["candidate"].get("title"),
                            "search_queries": item["candidate"].get("search_queries"),
                            "reason": "next_initial_candidate_after_failed_section",
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
            directed_candidates: list[dict] = []
            if not followup_evidence:
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
                            section_id = str(candidate.get("section_id") or "")
                            if section_id and section_id in tried_section_ids:
                                continue
                            if title_is_avoided(str(candidate.get("title") or ""), failed_titles):
                                continue
                            directed_candidates.append({"doc_id": doc_id, "plan": directed_plan, "candidate": candidate})
            if (not followup_evidence) and directed_candidates:
                item = directed_candidates[0]
                section_id = str(item["candidate"].get("section_id") or "")
                if section_id:
                    tried_section_ids.add(section_id)
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
                item, next_candidate_index = pop_next_untried_candidate(
                    section_candidate_queue,
                    next_candidate_index,
                    failed_titles,
                    tried_section_ids,
                )
                if not item:
                    break
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
                            item, next_candidate_index = pop_next_untried_candidate(
                                section_candidate_queue,
                                next_candidate_index,
                                failed_titles,
                                tried_section_ids,
                            )
                            if not item:
                                continue
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
            recent_evidence = followup_evidence
            task_evidence = merge_evidence(task_evidence, recent_evidence)
            task_evidence_notes = update_task_evidence_notes(
                state, task, recent_evidence, task_evidence_notes, noted_evidence_ids
            )
            round_log["evidence_notes_after_followup"] = task_evidence_notes
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
        if task_result.get("status") == "complete" and result_contains_unsupported_assumption(
            task_result,
            task=task,
            task_memory=task_memory,
            evidence=task_evidence,
            question_text=state["question"]["question"],
        ):
            task_result["status"] = "blocked"
            task_result.setdefault("warnings", []).append("结果包含未经证据支持的假设，已改为blocked")
            missing = task_result.get("missing_slots") or []
            if "具体比例/金额证据" not in missing:
                missing.append("具体比例/金额证据")
            task_result["missing_slots"] = missing
        if not task_result.get("task"):
            task_result["task"] = task.get("task", "")
        task_result["evidence_notes"] = compact_evidence_notes_for_final(task_evidence_notes)
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
    answer_format = str(state["question"].get("answer_format") or "").strip().lower()
    answer = str(output.get("answer") or "").strip()
    if answer_format != "free":
        answer = answer.upper()
        if answer not in "ABCD":
            answer = ""
    else:
        deterministic_answer = deterministic_time_comparison_answer(state, task_results)
        if deterministic_answer:
            output["deterministic_time_comparison_answer"] = deterministic_answer
            if answer != deterministic_answer:
                output["answer_before_time_comparison_fix"] = answer
                add_trace(state, f"human toc time comparison override {answer}->{deterministic_answer}")
            answer = deterministic_answer
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
    graph.add_node("slot_planner", slot_planner_node)
    graph.add_node("route_docs", route_docs_node)
    graph.add_node("retrieve_initial", retrieve_initial_node)
    graph.add_node("evidence_extract", evidence_extract_node)
    graph.add_node("evidence_read", evidence_read_node)
    graph.add_node("audit", audit_node)
    graph.add_node("review_original_after_subsearch", review_original_after_subsearch_node)
    graph.add_node("focus_evidence", focus_evidence_node)
    graph.add_node("retrieve_more", retrieve_more_node)
    graph.add_node("toc_repair", toc_repair_node)
    graph.add_node("judge", judge_node)
    graph.add_node("advance_option", advance_option_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("init")
    graph.add_conditional_edges(
        "init",
        lambda state: "human_toc_reasoning" if use_task_reasoning_path(state["question"]) else "slot_planner",
        {"human_toc_reasoning": "human_toc_reasoning", "slot_planner": "slot_planner"},
    )
    graph.add_edge("human_toc_reasoning", "finalize")
    graph.add_edge("slot_planner", "route_docs")
    graph.add_edge("route_docs", "retrieve_initial")
    graph.add_edge("retrieve_initial", "evidence_extract")
    graph.add_edge("evidence_extract", "evidence_read")
    graph.add_edge("evidence_read", "audit")
    graph.add_conditional_edges(
        "audit",
        route_after_audit,
        {
            "review_original_after_subsearch": "review_original_after_subsearch",
            "focus_evidence": "focus_evidence",
            "retrieve_more": "retrieve_more",
            "toc_repair": "toc_repair",
            "judge": "judge",
        },
    )
    graph.add_edge("review_original_after_subsearch", "evidence_read")
    graph.add_edge("focus_evidence", "evidence_read")
    graph.add_edge("retrieve_more", "evidence_extract")
    graph.add_edge("toc_repair", "evidence_extract")
    graph.add_edge("judge", "advance_option")
    graph.add_conditional_edges("advance_option", route_after_advance, {"route_docs": "slot_planner", "finalize": "finalize"})
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


def select_atomic_record(records: list[dict], atomic_id: str) -> dict:
    for item in records:
        if str(item.get("atomic_id") or item.get("qid") or "") == atomic_id:
            return item
    raise StopIteration(f"Atomic question not found: {atomic_id}")


def all_doc_ids_for_loaded_index() -> list[str]:
    return sorted(PAGES_BY_DOC.keys())


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


def enterprise_doc_route_score(doc_id: str, query: str, atomic: dict) -> tuple[float, list[str]]:
    profile = doc_profile_text(doc_id, max_chars=1400)
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
    tree_doc = TREE_DOC_BY_ID.get(doc_id) or {}
    title_surface = compact_text(" ".join(str(tree_doc.get(key) or "") for key in ("title", "doc_title", "source_path")))
    for entity in atomic.get("entities") or []:
        entity = compact_text(str(entity))
        if entity and entity in title_surface:
            score += 12.0
    if any(term in title_surface for term in query_tokens[:16]):
        score += 2.0
    return score, list(dict.fromkeys(overlap))[:20]


def enterprise_prefilter_doc_ids(atomic: dict, max_docs: int | None = None) -> list[str]:
    source_trace = atomic.get("source_trace") or {}
    explicit = [doc_id for doc_id in source_trace.get("source_doc_ids") or [] if doc_id in PAGES_BY_DOC]
    if explicit:
        return explicit
    if os.environ.get("DISABLE_ENTERPRISE_DOC_ROUTER", "0").lower() in {"1", "true", "yes"}:
        return all_doc_ids_for_loaded_index()
    max_docs = max_docs or int(os.environ.get("ENTERPRISE_DOC_TOP_K", "6"))
    query = enterprise_doc_query_text(atomic)
    scored = []
    for doc_id in all_doc_ids_for_loaded_index():
        score, overlap = enterprise_doc_route_score(doc_id, query, atomic)
        if score > 0:
            scored.append((score, doc_id, overlap))
    scored.sort(key=lambda row: row[0], reverse=True)
    selected = [doc_id for _score, doc_id, _overlap in scored[:max_docs]]
    return selected or all_doc_ids_for_loaded_index()


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


def atomic_to_legacy_question(atomic: dict) -> dict:
    task_type = str(atomic.get("task_type") or "claim_verification")
    source_trace = atomic.get("source_trace") or {}
    doc_ids = enterprise_prefilter_doc_ids(atomic)
    question_text = maybe_fix_mojibake(str(atomic.get("question") or ""))

    base = {
        "qid": str(atomic.get("atomic_id") or atomic.get("source_qid") or TARGET_QID),
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


def enterprise_answer_from_state(state: AgentState) -> str:
    question = state.get("question", {})
    task_type = str(question.get("enterprise_task_type") or "")
    raw = str(state.get("final_answer") or "").strip()
    if task_type == "claim_verification":
        if raw in {"T", "A", "TRUE", "正确"}:
            return "supported"
        if raw in {"F", "B", "FALSE", "错误"}:
            return "contradicted_or_insufficient"
    if task_type == "rule_applicability":
        if raw in {"T", "A", "TRUE", "正确"}:
            return "applicable"
        if raw in {"F", "B", "FALSE", "错误"}:
            return "not_applicable_or_insufficient"
    return raw


def evidence_summary_from_state(state: AgentState, max_items: int = 8) -> list[dict]:
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


def enterprise_reason_from_state(state: AgentState) -> str:
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


def write_enterprise_result(state: AgentState) -> dict:
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
    path = Path(ANSWER_CSV)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"enterprise_result_jsonl={path.resolve()}")
    return result


def main() -> None:
    atomic_records = load_records(ATOMIC_QUESTIONS_PATH)
    atomic = select_atomic_record(atomic_records, TARGET_QID)
    question = atomic_to_legacy_question(atomic)
    print(f"atomic_id={question['qid']}")
    print(f"domain={question.get('enterprise_domain')}")
    print(f"page_index={PAGE_INDEX_PATH}")
    print(f"prefilter_doc_ids={json.dumps(question.get('enterprise_prefiltered_doc_ids', []), ensure_ascii=False)}")
    app = build_graph()
    final_state = app.invoke({"question": question})
    print(f"qid={final_state['qid']}")
    enterprise_result = write_enterprise_result(final_state)
    print(f"answer={enterprise_result.get('answer', '')}")
    print(f"raw_final_answer={final_state.get('final_answer', '')}")
    print(f"status={final_state.get('status')}")
    print(f"token_usage={json.dumps(final_state.get('token_usage', {}), ensure_ascii=False)}")
    if APPEND_ANSWER_CSV:
        upsert_answer_csv(final_state)
    if SAVE_DEBUG_OUTPUTS:
        write_outputs(final_state)


if __name__ == "__main__":
    main()

















