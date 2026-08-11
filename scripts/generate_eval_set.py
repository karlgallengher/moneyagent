from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enterprise_qa_agent.src.core.text_utils import compact_text, maybe_fix_mojibake


DOMAIN_INDEX_DIRS = {
    "financial_contracts": "processed/page_index_financial_contracts",
    "financial_reports": "processed/page_index_financial_reports",
    "insurance": "processed/page_index_insurance",
    "regulatory": "processed/page_index_regulatory",
    "research": "processed/page_index_research",
}

DEFAULT_OUTPUT = "tests/eval_sets/moneyagent_golden_draft.jsonl"
STOP_TERMS = {
    "公司",
    "报告",
    "年度",
    "本公司",
    "相关",
    "主要",
    "业务",
    "情况",
    "管理",
    "金融",
    "市场",
    "行业",
    "风险",
    "资产",
    "投资",
    "规定",
    "办法",
    "保险",
}

MOJIBAKE_MARKERS = ("鏍", "嵁", "銆", "涓", "鍏", "鐨", "锛", "浠", "绗", "骞", "閫", "鐢")
COMMON_CJK_TERMS = ("根据", "公司", "报告", "发行", "债券", "保险", "年度", "收入", "规定", "管理")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_catalog(index_dir: Path) -> dict[str, dict[str, Any]]:
    catalog = {}
    for item in load_jsonl(index_dir / "doc_catalog.jsonl"):
        doc_id = str(item.get("doc_id") or "")
        if doc_id:
            catalog[doc_id] = item
    return catalog


def force_fix_mojibake(value: str) -> str:
    text = maybe_fix_mojibake(value)
    if not any(marker in text for marker in MOJIBAKE_MARKERS):
        return text
    try:
        fixed = text.encode("gb18030").decode("utf-8")
    except UnicodeError:
        fixed = text.encode("gb18030", errors="ignore").decode("utf-8", errors="ignore")
    fixed_score = sum(fixed.count(term) for term in COMMON_CJK_TERMS) * 4
    text_score = sum(text.count(term) for term in COMMON_CJK_TERMS) * 4
    fixed_score -= sum(fixed.count(marker) for marker in MOJIBAKE_MARKERS)
    text_score -= sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
    return fixed if fixed_score >= text_score else text


def clean(value: Any, max_chars: int = 240) -> str:
    return compact_text(force_fix_mojibake(str(value or "")))[:max_chars]


def title_for_page(page: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> str:
    doc = catalog.get(str(page.get("doc_id") or "")) or {}
    return clean(doc.get("title_fixed") or doc.get("title") or page.get("title") or page.get("doc_id"), 120)


def heading_for_page(page: dict[str, Any]) -> str:
    heading = " > ".join(str(item) for item in page.get("heading_path") or [] if item)
    return clean(heading or page.get("title") or "相关章节", 120)


def extract_numbers(text: str) -> list[str]:
    patterns = [
        r"\d+(?:\.\d+)?\s*%",
        r"\d+(?:\.\d+)?\s*(?:亿元|万元|元|天|日|年|月|倍|个百分点|个基点|万股|股)",
        r"(?:19|20)\d{2}\s*年(?:\s*\d{1,2}\s*月)?(?:\s*\d{1,2}\s*日)?",
    ]
    values: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, text):
            value = clean(match, 40)
            if value and value not in values:
                values.append(value)
    return values[:4]


def keyword_candidates(text: str) -> list[str]:
    words = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,12}", text)
    scored: dict[str, int] = defaultdict(int)
    for word in words:
        if word in STOP_TERMS or word.isdigit():
            continue
        if re.fullmatch(r"\d+(?:\.\d+)?", word):
            continue
        scored[word] += 1 + min(len(word), 6)
    ranked = sorted(scored.items(), key=lambda item: (item[1], len(item[0])), reverse=True)
    return [word for word, _score in ranked[:8]]


def sentence_with_answer(text: str, answer: str) -> str:
    parts = re.split(r"(?<=[。；;.!?？])", text)
    for part in parts:
        if answer in part:
            return clean(part, 320)
    pos = text.find(answer)
    if pos < 0:
        return clean(text, 320)
    return clean(text[max(0, pos - 120): pos + 180], 320)


def make_numeric_case(domain: str, page: dict[str, Any], catalog: dict[str, dict[str, Any]], index: int) -> dict[str, Any] | None:
    text = clean(page.get("text") or page.get("index_text"), 1200)
    numbers = extract_numbers(text)
    if not numbers:
        return None
    answer = numbers[0]
    title = title_for_page(page, catalog)
    heading = heading_for_page(page)
    evidence = sentence_with_answer(text, answer)
    return {
        "id": f"{domain}_num_{index:03d}",
        "domain": domain,
        "question": f"根据《{title}》中“{heading}”部分，文中提到的关键数值是多少？请给出原文中的数值并简要说明对应事项。",
        "answer_type": "numeric_fact",
        "expected_answer": answer,
        "expected_answer_keywords": [answer],
        "expected_doc_ids": [str(page.get("doc_id") or "")],
        "expected_evidence_ids": [str(page.get("page_id") or "")],
        "expected_evidence_keywords": list(dict.fromkeys([answer] + keyword_candidates(evidence)[:4])),
        "source": {
            "doc_id": page.get("doc_id"),
            "page_id": page.get("page_id"),
            "heading": heading,
            "quote": evidence,
        },
        "needs_human_review": True,
    }


def make_summary_case(domain: str, page: dict[str, Any], catalog: dict[str, dict[str, Any]], index: int) -> dict[str, Any] | None:
    text = clean(page.get("text") or page.get("index_text"), 1200)
    keywords = keyword_candidates(text)
    if len(keywords) < 3:
        return None
    title = title_for_page(page, catalog)
    heading = heading_for_page(page)
    return {
        "id": f"{domain}_summary_{index:03d}",
        "domain": domain,
        "question": f"根据《{title}》中“{heading}”部分，概括该部分主要讨论的内容。",
        "answer_type": "summary_keyword",
        "expected_answer": "；".join(keywords[:4]),
        "expected_answer_keywords": keywords[:4],
        "expected_doc_ids": [str(page.get("doc_id") or "")],
        "expected_evidence_ids": [str(page.get("page_id") or "")],
        "expected_evidence_keywords": keywords[:6],
        "source": {
            "doc_id": page.get("doc_id"),
            "page_id": page.get("page_id"),
            "heading": heading,
            "quote": clean(text, 320),
        },
        "needs_human_review": True,
    }


def page_is_useful(page: dict[str, Any]) -> bool:
    text = clean(page.get("text") or page.get("index_text"), 1400)
    if len(text) < 80:
        return False
    if text.count("目录") >= 3 or text.count("免责声明") >= 2:
        return False
    return True


def build_cases(per_domain: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for domain, rel_dir in DOMAIN_INDEX_DIRS.items():
        index_dir = REPO_ROOT / rel_dir
        catalog = load_catalog(index_dir)
        pages = [page for page in load_jsonl(index_dir / "page_index.jsonl") if page_is_useful(page)]
        numeric_count = 0
        summary_count = 0
        used_doc_ids: set[str] = set()

        for page in pages:
            if numeric_count >= max(1, per_domain // 2):
                break
            doc_id = str(page.get("doc_id") or "")
            if doc_id in used_doc_ids:
                continue
            case = make_numeric_case(domain, page, catalog, numeric_count + 1)
            if case:
                cases.append(case)
                used_doc_ids.add(doc_id)
                numeric_count += 1

        for page in pages:
            if numeric_count + summary_count >= per_domain:
                break
            doc_id = str(page.get("doc_id") or "")
            if doc_id in used_doc_ids:
                continue
            case = make_summary_case(domain, page, catalog, summary_count + 1)
            if case:
                cases.append(case)
                used_doc_ids.add(doc_id)
                summary_count += 1
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a draft MoneyAgent eval set from processed indexes.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--per-domain", type=int, default=10)
    args = parser.parse_args()

    cases = build_cases(per_domain=args.per_domain)
    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as fh:
        for case in cases:
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"wrote={output_path}")
    print(f"cases={len(cases)}")


if __name__ == "__main__":
    main()
