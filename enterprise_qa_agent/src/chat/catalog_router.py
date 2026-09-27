from __future__ import annotations

import json
import os
import re
from pathlib import Path

from enterprise_qa_agent.src.chat.domain_registry import domain_index_dirs

REPO_ROOT = Path(__file__).resolve().parents[3]

DOMAIN_PAGE_INDEX_PATHS = {
    "financial_contracts": "processed/page_index_financial_contracts/page_index.jsonl",
    "financial_reports": "processed/page_index_financial_reports/page_index.jsonl",
    "insurance": "processed/page_index_insurance/page_index.jsonl",
    "regulatory": "processed/page_index_regulatory/page_index.jsonl",
    "research": "processed/page_index_research/page_index.jsonl",
}

def refresh_domain_paths() -> None:
    DOMAIN_PAGE_INDEX_PATHS.clear()
    DOMAIN_PAGE_INDEX_PATHS.update(
        {domain: f"{directory}/page_index.jsonl" for domain, directory in domain_index_dirs().items()}
    )


DOMAIN_RULE_TERMS = {
    "insurance": (
        "保险", "条款", "赔付", "理赔", "免责", "退保", "现金价值", "等待期", "保险计划", "保单", "身故", "伤残",
        "投保", "家财险", "财产险", "医疗险", "百万医疗", "意外险", "水管爆裂", "门诊", "医保", "免赔", "应赔", "赔多少",
    ),
    "financial_reports": ("年报", "财报", "财务报表", "年度报告", "营业收入", "营收", "净利润", "研发投入", "资产负债表", "现金流量", "ROE", "毛利率"),
    "financial_contracts": (
        "可转债", "募集说明书", "发行人", "认购", "承诺函", "上市公司", "债券", "保荐", "招股", "公告",
        "发行股份", "支付现金购买资产", "募集配套资金", "关联交易报告书", "重大资产重组", "重组报告书",
        "标的资产", "交易标的", "评估增值率", "资产评估", "评估基准日", "加期评估", "交易作价",
    ),
    "regulatory": ("办法", "规定", "通知", "决定书", "监管", "处罚", "施行", "发布", "文号", "违法", "合规", "反洗钱", "客户尽职调查", "代理行", "空壳银行", "金融机构"),
    "research": ("行业", "趋势", "研报", "渠道", "消费", "集中度", "出海", "格局", "景气", "成本优势", "订单", "新签订单", "GMV", "渗透率", "增长", "市场规模", "结构性降本", "光通信"),
}


def _compact(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _norm_key(value: str) -> str:
    return re.sub(r"[\s、，,。；;：:（）()\[\]【】《》\"'“”‘’/\\_-]+", "", str(value or "").lower())


def load_doc_catalog(legacy_agent) -> dict[str, dict]:
    catalog_path = Path(legacy_agent.PAGE_INDEX_PATH).with_name("doc_catalog.jsonl")
    if not catalog_path.exists():
        return {}
    catalog: dict[str, dict] = {}
    with catalog_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = str(item.get("doc_id") or "")
            if doc_id:
                catalog[doc_id] = item
    return catalog


def load_doc_catalog_path(catalog_path: Path) -> dict[str, dict]:
    if not catalog_path.exists():
        return {}
    catalog: dict[str, dict] = {}
    with catalog_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = str(item.get("doc_id") or "")
            if doc_id:
                catalog[doc_id] = item
    return catalog


def _fixed_text(value, legacy_agent) -> str:
    if isinstance(value, list):
        return "；".join(_fixed_text(item, legacy_agent) for item in value if item)
    return legacy_agent.maybe_fix_mojibake(str(value or ""))


def catalog_profile_text(doc_id: str, catalog: dict[str, dict], legacy_agent, max_chars: int = 900) -> str:
    item = catalog.get(str(doc_id)) or {}
    if not item:
        return ""
    fields = [
        f"title={_fixed_text(item.get('title_fixed') or item.get('title'), legacy_agent)}",
        f"aliases={_fixed_text((item.get('aliases') or [])[:12], legacy_agent)}",
        f"top_headings={_fixed_text((item.get('top_headings') or [])[:18], legacy_agent)}",
        f"first_page={_fixed_text(item.get('first_page_preview'), legacy_agent)[:260]}",
    ]
    return _compact(" ".join(part for part in fields if part))[:max_chars]


def query_doc_terms(query: str) -> list[str]:
    raw = _compact(query)
    terms: list[str] = []
    for quoted in re.findall(r"《([^》]{2,120})》", raw):
        quoted = quoted.strip()
        if quoted and quoted not in terms:
            terms.append(quoted)
    for pattern in (
        r"[\u4e00-\u9fff]{2,12}(?:保险|险|条款|办法|规定|报告|年报|合同|募集说明书)",
        r"[\u4e00-\u9fffA-Za-z0-9.]{2,18}",
    ):
        for match in re.findall(pattern, raw):
            match = match.strip()
            if match and match not in terms:
                terms.append(match)
    for marker in (
        "家财险", "百万医疗", "团体百万医疗", "e生保", "水管爆裂", "门诊", "免赔额", "赔付比例", "保险责任",
        "发行股份及支付现金购买资产", "支付现金购买资产", "募集配套资金", "关联交易报告书", "重大资产重组",
        "重组报告书", "标的资产", "交易标的", "评估增值率", "资产评估报告", "资产评估", "评估基准日",
        "加期评估", "交易作价",
    ):
        if marker in raw and marker not in terms:
            terms.append(marker)
    return terms[:40]


def _is_weak_doc_term(term: str) -> bool:
    normalized = _norm_key(term)
    if not normalized:
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?%?", normalized):
        return True
    if re.fullmatch(r"(?:19|20)\d{2}年?", normalized):
        return True
    if re.fullmatch(r"\d{1,2}(?:月|日)?", normalized):
        return True
    return False


def raw_catalog_profile_text(item: dict, max_chars: int = 2200) -> str:
    fields = [
        f"title={item.get('title_fixed') or item.get('title') or ''}",
        f"aliases={'；'.join(str(value) for value in (item.get('aliases') or [])[:14])}",
        f"top_headings={'；'.join(str(value) for value in (item.get('top_headings') or [])[:24])}",
        f"first_page={str(item.get('first_page_preview') or '')[:260]}",
    ]
    return _compact(" ".join(part for part in fields if part))[:max_chars]


def raw_catalog_route_score(query: str, item: dict) -> tuple[float, list[str]]:
    profile = raw_catalog_profile_text(item)
    profile_key = _norm_key(profile)
    query_key = _norm_key(query)
    terms = query_doc_terms(query)
    hits: list[str] = []
    score = 0.0

    for term in terms:
        term_key = _norm_key(term)
        if not term_key or len(term_key) < 2:
            continue
        if term_key in profile_key:
            hits.append(term)
            score += 0.2 if _is_weak_doc_term(term) else (10.0 if len(term_key) >= 4 else 3.0)

    for size in range(16, 5, -1):
        for start in range(0, max(0, len(query_key) - size + 1)):
            phrase = query_key[start : start + size]
            if len(re.findall(r"[\u4e00-\u9fff]", phrase)) < 3:
                continue
            if phrase in profile_key and phrase not in hits:
                hits.append(phrase)
                score += min(14.0, len(phrase) / 1.3)
        if len(hits) >= 10:
            break

    return score, list(dict.fromkeys(hits))[:14]


def cross_domain_catalog_search(query: str, top_k: int = 12) -> list[dict]:
    refresh_domain_paths()
    results: list[dict] = []
    for domain, page_index_path in DOMAIN_PAGE_INDEX_PATHS.items():
        catalog_path = (REPO_ROOT / page_index_path).with_name("doc_catalog.jsonl")
        catalog = load_doc_catalog_path(catalog_path)
        for doc_id, item in catalog.items():
            score, hits = raw_catalog_route_score(query, item)
            if score <= 0:
                continue
            results.append(
                {
                    "domain": domain,
                    "doc_id": doc_id,
                    "score": round(score, 3),
                    "hits": hits,
                    "title": str(item.get("title_fixed") or item.get("title") or ""),
                }
            )
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def strong_catalog_hit(catalog_hits: list[dict]) -> bool:
    if not catalog_hits:
        return False
    top_score = float(catalog_hits[0].get("score") or 0.0)
    if top_score < float(os.environ.get("ENTERPRISE_CATALOG_STRONG_SCORE", "24")):
        return False
    if len(catalog_hits) == 1:
        return True
    second_score = float(catalog_hits[1].get("score") or 0.0)
    return top_score >= max(second_score + 10.0, second_score * 1.8)


def catalog_route_score(query: str, doc_id: str, catalog: dict[str, dict], legacy_agent) -> tuple[float, list[str]]:
    profile = catalog_profile_text(doc_id, catalog, legacy_agent, max_chars=2200)
    if not profile:
        profile = legacy_agent.doc_profile_text(doc_id, max_chars=1600)
    profile_key = _norm_key(profile)
    query_key = _norm_key(query)
    terms = query_doc_terms(query)
    hits: list[str] = []
    score = 0.0
    for term in terms:
        term_key = _norm_key(term)
        if not term_key or len(term_key) < 2:
            continue
        if term_key in profile_key:
            hits.append(term)
            if _is_weak_doc_term(term):
                score += 0.2
            else:
                score += 8.0 if len(term_key) >= 4 else 3.0
    for size in range(12, 3, -1):
        for start in range(0, max(0, len(query_key) - size + 1)):
            phrase = query_key[start : start + size]
            if len(re.findall(r"[\u4e00-\u9fff]", phrase)) < 2:
                continue
            if phrase in profile_key and phrase not in hits:
                hits.append(phrase)
                score += min(12.0, len(phrase) / 1.4)
        if len(hits) >= 8:
            break
    if any(term in query for term in ("赔", "理赔", "水管", "门诊", "医疗", "家财", "免赔", "责任")):
        clause_terms = ("条款", "保险责任", "赔偿处理", "保险金计算", "免赔额", "赔付比例", "保险计划", "保险标的")
        report_terms = ("行业", "研报", "研究报告", "趋势", "渠道", "景气", "分析")
        if any(term in profile for term in clause_terms):
            score += 10.0
        if any(term in profile for term in report_terms) and not any(term in profile for term in ("条款", "保险合同")):
            score -= 8.0
    if any(term in query for term in ("发行股份", "购买资产", "募集配套", "关联交易报告书", "重大资产重组", "标的资产", "交易标的", "评估增值率", "资产评估", "评估基准日", "加期评估")):
        deal_terms = ("发行股份", "购买资产", "募集配套", "关联交易报告书", "重大资产重组", "重组报告书", "标的资产", "交易标的", "评估", "交易作价")
        research_terms = ("行业", "研报", "研究报告", "趋势", "渠道", "景气", "市场规模")
        if any(term in profile for term in deal_terms):
            score += 14.0
        if any(term in profile for term in research_terms) and not any(term in profile for term in deal_terms):
            score -= 10.0
    return score, list(dict.fromkeys(hits))[:12]


def sort_candidate_docs_by_catalog(query: str, candidate_doc_ids: list[str], catalog: dict[str, dict], legacy_agent) -> tuple[list[str], list[dict]]:
    scored = []
    for index, doc_id in enumerate(candidate_doc_ids):
        score, hits = catalog_route_score(query, doc_id, catalog, legacy_agent)
        scored.append((score, -index, doc_id, hits))
    scored.sort(reverse=True)
    ordered = [doc_id for _score, _neg_index, doc_id, _hits in scored]
    debug = [
        {"doc_id": doc_id, "score": round(score, 3), "hits": hits}
        for score, _neg_index, doc_id, hits in scored[:8]
    ]
    return ordered, debug


def infer_domain(query: str, requested: str) -> str:
    refresh_domain_paths()
    requested = (requested or "").strip()
    if requested in DOMAIN_PAGE_INDEX_PATHS:
        return requested
    if any(term in query for term in ("发行股份", "支付现金购买资产", "募集配套资金", "关联交易报告书", "重大资产重组", "重组报告书", "标的资产", "交易标的", "评估增值率", "资产评估", "评估基准日", "加期评估", "交易作价")):
        return "financial_contracts"
    scores: list[tuple[int, str]] = []
    lowered = query.lower()
    for domain, terms in DOMAIN_RULE_TERMS.items():
        score = sum(1 for term in terms if term.lower() in lowered)
        if score:
            scores.append((score, domain))
    if not scores:
        return "research"
    scores.sort(reverse=True)
    if len(scores) > 1 and scores[0][0] == scores[1][0]:
        return "research"
    return scores[0][1]


def rerank_chat_docs_with_llm(query: str, question: dict, legacy_agent) -> dict:
    if os.environ.get("DISABLE_ENTERPRISE_CHAT_DOC_AGENT", "0").lower() in {"1", "true", "yes"}:
        return question
    if question.get("enterprise_catalog_low_confidence"):
        return question

    candidate_doc_ids = list(dict.fromkeys(question.get("enterprise_prefiltered_doc_ids") or question.get("doc_ids") or []))
    if len(candidate_doc_ids) <= 1:
        return question

    max_candidates = int(os.environ.get("ENTERPRISE_CHAT_DOC_AGENT_CANDIDATES", "14"))
    max_selected = int(os.environ.get("ENTERPRISE_CHAT_DOC_AGENT_TOP_K", "4"))
    catalog = load_doc_catalog(legacy_agent)
    candidate_doc_ids, catalog_debug = sort_candidate_docs_by_catalog(query, candidate_doc_ids, catalog, legacy_agent)
    candidate_doc_ids = candidate_doc_ids[:max_candidates]
    doc_lines = []
    for doc_id in candidate_doc_ids:
        profile = legacy_agent.doc_profile_text(doc_id, max_chars=520)
        catalog_profile = catalog_profile_text(doc_id, catalog, legacy_agent, max_chars=780)
        doc_lines.append(f"- doc_id={doc_id}\n  catalog={catalog_profile}\n  profile={profile}")

    system = (
        "你是企业问答系统的文档路由代理。你的任务是从候选文档中选择最可能包含答案依据的文档。"
        "只能返回 JSON。不要解答问题。"
    )
    user = "\n".join(
        [
            f"用户问题：{query}",
            "",
            "候选文档：",
            "\n".join(doc_lines),
            "",
            "请返回：",
            '{"selected_doc_ids":["doc_id"],"confidence":"high|medium|low","reason":"简短说明"}',
            "规则：",
            "1. 优先选择标题、目录或摘要与问题实体/制度/业务对象直接相关的文档。",
            "2. 不要因为宽泛词如银行、金融、行业就选择主题明显无关的文档。",
            "3. 保险赔付、理赔或赔款计算问题，优先选择具体保险条款、保险合同、保险计划表、保险责任、免赔额、赔付比例、保险金计算相关文档；避免选择行业研报、趋势分析或主题不匹配的责任险/车险文档。",
            "4. 如果问题中出现多个产品、公司、制度或业务对象，应尽量为每个对象保留最匹配的文档。",
            "5. 如果不确定，保留多个可能文档；最多选择 4 个。",
        ]
    )
    try:
        router, usage = legacy_agent.call_qwen_json(system, user)
    except Exception as exc:
        question["enterprise_doc_agent_error"] = str(exc)[:300]
        return question

    selected = [doc_id for doc_id in router.get("selected_doc_ids", []) if doc_id in candidate_doc_ids]
    if not selected:
        return question
    selected = selected[:max_selected]
    question["doc_ids"] = selected
    question["enterprise_doc_agent"] = {
        "candidate_doc_ids": candidate_doc_ids,
        "selected_doc_ids": selected,
        "catalog_rank": catalog_debug,
        "confidence": router.get("confidence"),
        "reason": router.get("reason"),
        "usage": usage,
    }
    question["enterprise_prefiltered_doc_ids"] = selected
    return question


