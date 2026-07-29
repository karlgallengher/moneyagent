from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, TypedDict

from enterprise_qa_agent.src.routing.atomic_adapter import (
    atomic_to_legacy_question,
    select_atomic_record,
)
from enterprise_qa_agent.src.routing.doc_routing import (
    configure_doc_routing,
    deterministic_doc_hints,
    doc_aliases,
    doc_hint_scores,
    doc_profile_lines,
    doc_profile_text,
    doc_reroute_scores,
    has_doc_reroute_candidate,
    reroute_docs_for_missing_slot,
)
from enterprise_qa_agent.src.io.formatters import (
    compact_html_text,
    format_evidence,
    format_evidence_notes,
    format_memory_slots,
    print_evidence_preview,
)
from enterprise_qa_agent.src.graph.graph_builder import build_enterprise_graph
from enterprise_qa_agent.src.io.io_utils import load_json, load_jsonl, load_records
from enterprise_qa_agent.src.runtime.llm_client import DRY_RUN_WITHOUT_LLM, VERBOSE_CONSOLE, call_qwen_json, dry_json
from enterprise_qa_agent.src.io.output_utils import (
    upsert_answer_csv,
    write_enterprise_result,
    write_outputs,
)
from enterprise_qa_agent.src.prompts.option_prompts import (
    build_audit_prompt_text,
    build_doc_router_prompt_text,
    build_evidence_read_prompt_text,
    build_evidence_extract_prompt_text,
    build_judge_prompt_text,
    build_slot_planner_prompt_text,
    build_toc_repair_prompt_text,
)
from enterprise_qa_agent.src.core.retrieval_core import BM25, tokenize
from enterprise_qa_agent.src.retrieval.retrieval_aux_tools import (
    configure_retrieval_aux_tools,
    evidence_query_score,
    evidence_slot_score,
    expand_auto_anchor_evidence,
    expand_forward_evidence_ids,
    is_expandable_anchor_evidence,
    rerank_evidence_for_slot,
    rerank_initial_evidence,
    retrieve_by_key_term_coverage,
    summary_candidate_evidence,
)
from enterprise_qa_agent.src.retrieval.retrieval_tools import (
    RetrievalTools,
    configure_basic_retrieval,
    evidence_by_ids_strict,
    merge_evidence,
    page_to_evidence,
    retrieve_multi_doc,
    retrieve_one_doc,
)
from enterprise_qa_agent.src.retrieval.section_retrieval import (
    compact_doc_tree_text,
    compact_toc_for_docs,
    configure_section_retrieval,
    doc_section_catalog,
    expand_title_pointer_evidence,
    find_sections_referred_by_text,
    normalize_heading_key,
    retrieve_by_toc_hint,
    retrieve_one_doc_by_sections,
    section_context_evidence,
    section_pages_as_evidence,
    section_pages_as_evidence_ranked,
    section_title,
    section_title_as_evidence,
    select_relevant_doc_sections,
)
from enterprise_qa_agent.src.state.state_utils import (
    current_evidence_for_read,
    merge_memory_slots,
    normalize_evidence_id_list,
    prioritize_evidence_for_review,
    remembered_evidence_ids,
    select_judge_evidence,
    slot_evidence_ids,
    slot_key,
)
from enterprise_qa_agent.src.prompts.task_reasoning_prompts import (
    build_task_reasoning_audit_prompt_text,
    build_task_reasoning_final_prompt_text,
    build_task_reasoning_plan_prompt_text,
    build_task_reasoning_result_prompt_text,
)
from enterprise_qa_agent.src.core.text_utils import compact_text, maybe_fix_mojibake


DOMAIN_PAGE_INDEX_PATHS = {
    "financial_contracts": "processed/page_index_financial_contracts/page_index.jsonl",
    "financial_reports": "processed/page_index_financial_reports/page_index.jsonl",
    "insurance": "processed/page_index_insurance/page_index.jsonl",
    "regulatory": "processed/page_index_regulatory/page_index.jsonl",
    "research": "processed/page_index_research/page_index.jsonl",
}

RETRIEVAL_TOOLS: RetrievalTools | None = None


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


PAGES = load_jsonl(PAGE_INDEX_PATH)
BM25_INDEX = BM25(PAGES)
configure_basic_retrieval(bm25_index=BM25_INDEX, fix_text=maybe_fix_mojibake)
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


def is_heading_only_page(page: dict) -> bool:
    text = compact_text(page.get("text", ""))
    if len(text) < SECTION_CONTEXT_MIN_CHARS:
        return True
    heading = compact_text(" > ".join(page.get("heading_path") or []))
    if heading and (text == heading or heading.endswith(text)):
        return True
    return False


configure_section_retrieval(
    bm25_index=BM25_INDEX,
    page_by_id=PAGE_BY_ID,
    pages_by_doc=PAGES_BY_DOC,
    sections_by_doc=SECTIONS_BY_DOC,
    doc_sections_by_doc=DOC_SECTIONS_BY_DOC,
    tree_sections_by_doc=TREE_SECTIONS_BY_DOC,
    doc_tree_text_by_id=DOC_TREE_TEXT_BY_ID,
    is_heading_only_page=is_heading_only_page,
    call_json=call_qwen_json,
    add_usage=add_usage,
    dry_run=DRY_RUN_WITHOUT_LLM,
    section_context_per_section=SECTION_CONTEXT_PER_SECTION,
    section_context_max_total=SECTION_CONTEXT_MAX_TOTAL,
)


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



def apply_local_subsearch_to_new_evidence(state: AgentState, source: str) -> None:
    option = state["current_option"]
    option_state = state["option_states"][option]
    original_new_ids = list(option_state.get("new_evidence_ids") or [])
    query = local_subsearch_query(state, option_state)
    focused = get_retrieval_tools().local_subsearch(
        evidence=option_state.get("evidence", []),
        evidence_ids=original_new_ids,
        query=query,
        source=source,
        min_chars=LOCAL_SUBSEARCH_MIN_CHARS,
        max_snippets=LOCAL_SUBSEARCH_MAX_SNIPPETS,
        snippet_chars=LOCAL_SUBSEARCH_SNIPPET_CHARS,
    )["evidence"]
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


configure_retrieval_aux_tools(
    pages=PAGES,
    pages_by_doc=PAGES_BY_DOC,
    is_heading_only_page=is_heading_only_page,
    is_summary_like_page=is_summary_like_page,
    is_low_value_evidence=is_low_value_evidence,
    normalize_key_terms=normalize_key_terms,
    summary_candidates_per_doc=INITIAL_SUMMARY_CANDIDATES_PER_DOC,
    verbose_console=VERBOSE_CONSOLE,
    summary_page_terms=SUMMARY_PAGE_TERMS,
)



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



def use_task_reasoning_path(question: dict) -> bool:
    question_type = str(question.get("type") or "").strip()
    answer_format = str(question.get("answer_format") or "").strip().lower()
    if question_type == "计算题":
        return True
    if question_type == "推理判断" and answer_format != "multi":
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



def get_retrieval_tools() -> RetrievalTools:
    global RETRIEVAL_TOOLS
    if RETRIEVAL_TOOLS is None:
        RETRIEVAL_TOOLS = RetrievalTools(
            section_search_fn=retrieve_one_doc_by_sections,
            toc_hint_fn=retrieve_by_toc_hint,
            toc_text_fn=compact_toc_for_docs,
        )
    return RETRIEVAL_TOOLS



def option_claim(state: AgentState) -> str:
    option = state["current_option"]
    option_state = state.get("option_states", {}).get(option, {})
    if "claim" in option_state:
        return maybe_fix_mojibake(option_state["claim"])
    return maybe_fix_mojibake(state["question"]["options"][option])


UNIVERSAL_CLAIM_TERMS = ("都", "均", "全部", "所有", "每个", "各", "皆")



configure_doc_routing(
    pages_by_doc=PAGES_BY_DOC,
    tree_doc_by_id=TREE_DOC_BY_ID,
    doc_tree_text_by_id=DOC_TREE_TEXT_BY_ID,
    normalize_key_terms=normalize_key_terms,
    option_claim=option_claim,
    option_slot_query_text=option_slot_query_text,
    doc_reroute_top_k=DOC_REROUTE_TOP_K,
    doc_reroute_min_score=DOC_REROUTE_MIN_SCORE,
    doc_reroute_margin=DOC_REROUTE_MARGIN,
)

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
    return build_doc_router_prompt_text(
        maybe_fix_mojibake(state["question"]["question"]),
        option,
        option_claim(state),
        doc_lines,
        profile_lines,
        hint_docs,
    )


def build_slot_planner_prompt(state: AgentState) -> tuple[str, str]:
    option = state["current_option"]
    entities = atomic_entities_for_state(state)
    return build_slot_planner_prompt_text(
        maybe_fix_mojibake(state["question"]["question"]),
        state["question"].get("answer_format"),
        option,
        option_claim(state),
        entities,
    )


def build_audit_prompt(state: AgentState) -> tuple[str, str]:
    option = state["current_option"]
    option_state = state["option_states"][option]
    return build_audit_prompt_text(
        maybe_fix_mojibake(state["question"]["question"]),
        option,
        option_claim(state),
        option_state.get("target_doc_ids", []),
        json.dumps(option_state.get("slots", []), ensure_ascii=False, indent=2),
        format_memory_slots(option_state.get("memory_slots", [])),
        format_memory_slots(option_state.get("evidence_facts", [])),
        format_evidence(prioritize_evidence_for_review(option_state), max_chars=3600),
    )


def build_judge_prompt(state: AgentState) -> tuple[str, str]:
    option = state["current_option"]
    option_state = state["option_states"][option]
    return build_judge_prompt_text(
        maybe_fix_mojibake(state["question"]["question"]),
        state["question"].get("answer_format"),
        option,
        option_claim(state),
        format_memory_slots(option_state.get("memory_slots", [])),
        format_memory_slots(option_state.get("evidence_facts", [])),
        json.dumps(option_state.get("audit", {}), ensure_ascii=False, indent=2),
        format_evidence(select_judge_evidence(option_state), max_chars=6000),
    )


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
                get_retrieval_tools()
                .global_search(
                    query=search["query"],
                    doc_ids=search["target_doc_ids"],
                    top_k_per_doc=min(INITIAL_TOP_K, 2),
                    source="initial",
                )["evidence"]
            )
    elif slot_queries:
        evidence = []
        for query in slot_queries[:4]:
            evidence.extend(
                get_retrieval_tools()
                .global_search(
                    query=query,
                    doc_ids=target_doc_ids,
                    top_k_per_doc=min(INITIAL_TOP_K, 2),
                    source="initial",
                )["evidence"]
            )
    else:
        query = f"{maybe_fix_mojibake(state['question']['question'])}\n{option}. {option_state['claim']}"
        retrieval_query = query
        evidence = get_retrieval_tools().global_search(
            query=query,
            doc_ids=target_doc_ids,
            top_k_per_doc=INITIAL_TOP_K,
            source="initial",
        )["evidence"]
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
    return build_evidence_read_prompt_text(
        maybe_fix_mojibake(state["question"]["question"]),
        option,
        option_claim(state),
        json.dumps(option_state.get("slots", []), ensure_ascii=False, indent=2),
        format_memory_slots(option_state.get("memory_slots", [])),
        format_evidence(current_evidence_for_read(option_state), max_chars=3600),
    )


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
    return build_evidence_extract_prompt_text(
        maybe_fix_mojibake(state["question"]["question"]),
        option,
        option_claim(state),
        json.dumps(option_state.get("audit", {}).get("missing_slots") or [], ensure_ascii=False),
        json.dumps(evidence_items, ensure_ascii=False, indent=2),
    )


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
                get_retrieval_tools().global_search(
                    query=followup_query,
                    doc_ids=target_doc_ids,
                    top_k_per_doc=FOLLOWUP_TOP_K,
                    source=source,
                )["evidence"],
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
        section_evidence = get_retrieval_tools().section_search(
            query=query,
            doc_id=doc_id,
            top_k=3,
            source="toc_repair",
            section_hint=section_hint or query,
        )["evidence"]
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


REASON_SUPPORT_TERMS = (
    "与选项陈述一致",
    "与选项一致",
    "选项成立",
    "陈述成立",
    "说法成立",
    "能够支持",
    "可以支持",
    "支持该选项",
    "支持选项",
    "方向一致",
    "口径一致",
)

REASON_REJECT_TERMS = (
    "与选项陈述不一致",
    "与选项不一致",
    "选项不成立",
    "陈述不成立",
    "说法不成立",
    "选项错误",
    "陈述错误",
    "说法错误",
    "不能支持",
    "无法支持",
    "不支持该选项",
    "不支持选项",
    "方向相反",
    "口径不一致",
    "证据不足",
    "缺少证据",
    "无法判断",
)


def infer_verdict_from_reason(reason: str) -> bool | None:
    text = maybe_fix_mojibake(reason)
    if not text:
        return None
    has_support = any(term in text for term in REASON_SUPPORT_TERMS)
    has_reject = any(term in text for term in REASON_REJECT_TERMS)
    if has_support and not has_reject:
        return True
    if has_reject and not has_support:
        return False
    return None


def set_judgment_answer_from_verdict(judgment: dict, option: str, answer_format: str) -> dict:
    if answer_format == "tf":
        if judgment.get("verdict") is True:
            judgment["answer"] = "T"
        elif judgment.get("verdict") is False:
            judgment["answer"] = "F"
    else:
        if judgment.get("verdict") is True:
            judgment["answer"] = option
        elif judgment.get("verdict") is False:
            judgment["answer"] = ""
    return judgment


def enforce_judgment_reason_consistency(judgment: dict, option: str, answer_format: str) -> dict:
    inferred = infer_verdict_from_reason(str(judgment.get("reason") or ""))
    if inferred is None or judgment.get("verdict") is inferred:
        return judgment
    original = judgment.get("verdict")
    judgment["verdict"] = inferred
    judgment["confidence"] = min(float(judgment.get("confidence") or 0.0), 0.85)
    judgment["reason_consistency_fix"] = {
        "original_verdict": original,
        "fixed_verdict": inferred,
    }
    return set_judgment_answer_from_verdict(judgment, option, answer_format)


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
    answer_format = state["question"].get("answer_format", "")
    judgment = normalize_judgment(judgment, option, answer_format)
    judgment = enforce_judgment_reason_consistency(judgment, option, answer_format)
    if judgment.get("reason_consistency_fix"):
        add_trace(state, f"{option}: judge reason/verdict consistency fixed")
    judgment = enforce_judgment_missing_slot_guard(
        judgment,
        option_state,
        option,
        answer_format,
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
    return build_task_reasoning_plan_prompt_text(
        maybe_fix_mojibake(state["question"]["question"]),
        reasoning_options_text(state["question"]),
        doc_lines,
        valid_doc_ids,
    )


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
    return build_task_reasoning_audit_prompt_text(
        maybe_fix_mojibake(state["question"]["question"]),
        json.dumps(task, ensure_ascii=False, indent=2),
        format_memory_slots(task_memory),
        format_evidence_notes(evidence_notes or []),
        format_evidence(evidence[-2:] if evidence_notes else evidence, max_chars=1600 if evidence_notes else 5000),
    )


def build_task_reasoning_result_prompt(state: AgentState, task: dict, task_memory: list[dict], audit: dict) -> tuple[str, str]:
    return build_task_reasoning_result_prompt_text(
        maybe_fix_mojibake(state["question"]["question"]),
        json.dumps(task, ensure_ascii=False, indent=2),
        format_memory_slots(task_memory),
        json.dumps(audit, ensure_ascii=False),
    )


def build_task_reasoning_final_prompt(state: AgentState, task_results: list[dict]) -> tuple[str, str]:
    option_claims = build_option_claims(state["question"])
    answer_format = str(state["question"].get("answer_format") or "").strip().lower()
    if answer_format == "free":
        answer_instruction = '"answer": "按题干要求填写最终答案；只填数字、日期、排序或文本本身，不要输出选项字母或解释"'
    else:
        answer_instruction = '"answer": "A/B/C/D"'
    return build_task_reasoning_final_prompt_text(
        maybe_fix_mojibake(state["question"]["question"]),
        reasoning_options_text(state["question"]),
        json.dumps(option_claims, ensure_ascii=False, indent=2),
        json.dumps(task_results, ensure_ascii=False, indent=2),
        answer_instruction,
    )


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
    return build_enterprise_graph(
        AgentState,
        nodes={
            "init": init_state,
            "human_toc_reasoning": human_toc_reasoning_node,
            "slot_planner": slot_planner_node,
            "route_docs": route_docs_node,
            "retrieve_initial": retrieve_initial_node,
            "evidence_extract": evidence_extract_node,
            "evidence_read": evidence_read_node,
            "audit": audit_node,
            "review_original_after_subsearch": review_original_after_subsearch_node,
            "focus_evidence": focus_evidence_node,
            "retrieve_more": retrieve_more_node,
            "toc_repair": toc_repair_node,
            "judge": judge_node,
            "advance_option": advance_option_node,
            "finalize": finalize_node,
        },
        route_after_audit=route_after_audit,
        route_after_advance=route_after_advance,
        use_task_reasoning_path=use_task_reasoning_path,
    )


def all_doc_ids_for_loaded_index() -> list[str]:
    return sorted(PAGES_BY_DOC.keys())


def main() -> None:
    atomic_records = load_records(ATOMIC_QUESTIONS_PATH)
    atomic = select_atomic_record(atomic_records, TARGET_QID)
    question = atomic_to_legacy_question(
        atomic,
        TARGET_QID,
        all_doc_ids_for_loaded_index(),
        lambda doc_id: doc_profile_text(doc_id, max_chars=1400),
        TREE_DOC_BY_ID,
    )
    print(f"atomic_id={question['qid']}")
    print(f"domain={question.get('enterprise_domain')}")
    print(f"page_index={PAGE_INDEX_PATH}")
    print(f"prefilter_doc_ids={json.dumps(question.get('enterprise_prefiltered_doc_ids', []), ensure_ascii=False)}")
    app = build_graph()
    final_state = app.invoke({"question": question})
    print(f"qid={final_state['qid']}")
    enterprise_result = write_enterprise_result(final_state, ANSWER_CSV)
    print(f"answer={enterprise_result.get('answer', '')}")
    print(f"raw_final_answer={final_state.get('final_answer', '')}")
    print(f"status={final_state.get('status')}")
    print(f"token_usage={json.dumps(final_state.get('token_usage', {}), ensure_ascii=False)}")
    if APPEND_ANSWER_CSV:
        upsert_answer_csv(final_state, ANSWER_CSV)
    if SAVE_DEBUG_OUTPUTS:
        write_outputs(final_state, OUTPUT_DIR)


if __name__ == "__main__":
    main()




























