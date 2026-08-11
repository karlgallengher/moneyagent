from __future__ import annotations

import json
import math
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

OUTPUT = "tests/eval_sets/moneyagent_multi_task_50_draft.jsonl"
MOJIBAKE_MARKERS = ("鏍", "嵁", "銆", "涓", "鍏", "鐨", "锛", "浠", "绗", "骞", "閫", "鐢")
COMMON_CJK_TERMS = ("根据", "公司", "报告", "发行", "债券", "保险", "年度", "收入", "规定", "管理")
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
    "table",
    "td",
    "tr",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def force_fix_mojibake(value: Any) -> str:
    text = maybe_fix_mojibake(str(value or ""))
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
    return compact_text(force_fix_mojibake(value))[:max_chars]


def load_catalog(index_dir: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("doc_id")): row for row in load_jsonl(index_dir / "doc_catalog.jsonl") if row.get("doc_id")}


def title_for_page(page: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> str:
    doc = catalog.get(str(page.get("doc_id") or "")) or {}
    return clean(doc.get("title_fixed") or doc.get("title") or page.get("title") or page.get("doc_id"), 100)


def heading_for_page(page: dict[str, Any]) -> str:
    heading = " > ".join(str(item) for item in page.get("heading_path") or [] if item)
    return clean(heading or page.get("title") or "相关章节", 100)


def keyword_candidates(text: str, limit: int = 8) -> list[str]:
    words = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,14}", text)
    scored: dict[str, int] = defaultdict(int)
    for word in words:
        if word in STOP_TERMS or word.isdigit() or re.fullmatch(r"\d+(?:\.\d+)?", word):
            continue
        scored[word] += 1 + min(len(word), 6)
    ranked = sorted(scored.items(), key=lambda item: (item[1], len(item[0])), reverse=True)
    return [word for word, _score in ranked[:limit]]


def normalize_unit(unit: str) -> str:
    if unit == "%":
        return "%"
    if unit in {"亿元", "万元", "元"}:
        return unit
    if unit in {"天", "日"}:
        return "天"
    return unit


def parse_numeric(text: str) -> list[dict[str, Any]]:
    pattern = r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*(%|亿元|万元|元|天|日|年|月|倍|个百分点|个基点|万股|股)"
    rows: list[dict[str, Any]] = []
    for match in re.finditer(pattern, text):
        value = float(match.group(1))
        unit = normalize_unit(match.group(2))
        surface = clean(match.group(0), 40)
        if surface and all(item["surface"] != surface for item in rows):
            rows.append({"surface": surface, "value": value, "unit": unit})
    return rows[:8]


def sentence_with(text: str, value: str) -> str:
    parts = re.split(r"(?<=[。；;.!?？])", text)
    for part in parts:
        if value in part:
            return clean(part, 360)
    pos = text.find(value)
    if pos < 0:
        return clean(text, 360)
    return clean(text[max(0, pos - 140): pos + 220], 360)


def useful_pages(index_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for page in load_jsonl(index_dir / "page_index.jsonl"):
        text = clean(page.get("text") or page.get("index_text"), 1400)
        if len(text) < 100:
            continue
        if text.count("目录") >= 3 or text.count("免责声明") >= 2:
            continue
        numbers = parse_numeric(text)
        keywords = keyword_candidates(text)
        if numbers or len(keywords) >= 4:
            page = dict(page)
            page["_clean_text"] = text
            page["_numbers"] = numbers
            page["_keywords"] = keywords
            rows.append(page)
    return rows


def distinct_by_doc(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    rows = []
    for page in pages:
        doc_id = str(page.get("doc_id") or "")
        if doc_id in seen:
            continue
        seen.add(doc_id)
        rows.append(page)
    return rows


def alter_number(number: dict[str, Any]) -> str:
    value = float(number["value"])
    unit = str(number["unit"])
    changed = value + (1.0 if value >= 1 else 0.1)
    if value.is_integer() and unit != "%":
        return f"{int(changed)}{unit}"
    return f"{changed:.2f}{unit}".rstrip("0").rstrip(".") if unit == "%" else f"{changed:.2f}{unit}"


def format_sum(left: dict[str, Any], right: dict[str, Any]) -> str:
    total = float(left["value"]) + float(right["value"])
    unit = str(left["unit"])
    if math.isclose(total, round(total)):
        return f"{int(round(total))}{unit}"
    return f"{total:.2f}{unit}".rstrip("0").rstrip(".")


def base_source(page: dict[str, Any], catalog: dict[str, dict[str, Any]], answer: str = "") -> dict[str, Any]:
    text = page["_clean_text"]
    return {
        "doc_id": str(page.get("doc_id") or ""),
        "page_id": str(page.get("page_id") or ""),
        "title": title_for_page(page, catalog),
        "heading": heading_for_page(page),
        "quote": sentence_with(text, answer) if answer else clean(text, 360),
    }


def make_case(
    case_id: str,
    domain: str,
    task_type: str,
    question: str,
    expected_answer: str,
    expected_doc_ids: list[str],
    expected_evidence_ids: list[str],
    expected_answer_keywords: list[str],
    expected_evidence_keywords: list[str],
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": case_id,
        "domain": domain,
        "task_type": task_type,
        "question": question,
        "expected_answer": expected_answer,
        "expected_answer_keywords": list(dict.fromkeys(expected_answer_keywords)),
        "expected_doc_ids": list(dict.fromkeys(expected_doc_ids)),
        "expected_evidence_ids": list(dict.fromkeys(expected_evidence_ids)),
        "expected_evidence_keywords": list(dict.fromkeys(expected_evidence_keywords)),
        "sources": sources,
        "needs_human_review": True,
    }


def build_domain_cases(domain: str, index_dir: Path) -> list[dict[str, Any]]:
    catalog = load_catalog(index_dir)
    pages = useful_pages(index_dir)
    numeric_pages = distinct_by_doc([page for page in pages if page["_numbers"]])
    keyword_pages = distinct_by_doc([page for page in pages if len(page["_keywords"]) >= 4])
    cases: list[dict[str, Any]] = []

    for idx, page in enumerate(numeric_pages[:2], 1):
        number = page["_numbers"][0]
        title = title_for_page(page, catalog)
        heading = heading_for_page(page)
        cases.append(
            make_case(
                f"{domain}_fact_{idx:02d}",
                domain,
                "fact_extraction",
                f"根据《{title}》中“{heading}”部分，文中提到的关键数值是多少？请给出数值并说明对应事项。",
                number["surface"],
                [str(page.get("doc_id") or "")],
                [str(page.get("page_id") or "")],
                [number["surface"]],
                [number["surface"]] + page["_keywords"][:4],
                [base_source(page, catalog, number["surface"])],
            )
        )

    for idx, page in enumerate(numeric_pages[2:4], 1):
        number = page["_numbers"][0]
        wrong = alter_number(number)
        title = title_for_page(page, catalog)
        heading = heading_for_page(page)
        supported = idx % 2 == 1
        stated = number["surface"] if supported else wrong
        expected = "supported" if supported else "contradicted"
        cases.append(
            make_case(
                f"{domain}_verify_{idx:02d}",
                domain,
                "claim_verification",
                f"请核验：根据《{title}》中“{heading}”部分，相关关键数值为{stated}。该陈述是否成立？请给出依据。",
                expected,
                [str(page.get("doc_id") or "")],
                [str(page.get("page_id") or "")],
                [expected, number["surface"]],
                [number["surface"]] + page["_keywords"][:4],
                [base_source(page, catalog, number["surface"])],
            )
        )

    calc_pairs = []
    for page in numeric_pages:
        by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for number in page["_numbers"]:
            by_unit[number["unit"]].append(number)
        for values in by_unit.values():
            if len(values) >= 2:
                calc_pairs.append((page, values[0], values[1]))
                break
        if len(calc_pairs) >= 2:
            break
    for idx, (page, left, right) in enumerate(calc_pairs[:2], 1):
        answer = format_sum(left, right)
        title = title_for_page(page, catalog)
        heading = heading_for_page(page)
        cases.append(
            make_case(
                f"{domain}_calc_{idx:02d}",
                domain,
                "numeric_calculation",
                f"根据《{title}》中“{heading}”部分，把{left['surface']}和{right['surface']}相加，合计是多少？请写出计算过程。",
                answer,
                [str(page.get("doc_id") or "")],
                [str(page.get("page_id") or "")],
                [answer, left["surface"], right["surface"]],
                [left["surface"], right["surface"]] + page["_keywords"][:4],
                [base_source(page, catalog, left["surface"])],
            )
        )

    comparison_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for page in numeric_pages:
        if page["_numbers"]:
            comparison_candidates[page["_numbers"][0]["unit"]].append(page)
    comparisons = []
    for unit_pages in comparison_candidates.values():
        if len(unit_pages) >= 2:
            comparisons.append((unit_pages[0], unit_pages[1]))
            if len(unit_pages) >= 4:
                comparisons.append((unit_pages[2], unit_pages[3]))
        if len(comparisons) >= 2:
            break
    for idx, (left_page, right_page) in enumerate(comparisons[:2], 1):
        left_num = left_page["_numbers"][0]
        right_num = right_page["_numbers"][0]
        winner = left_page if float(left_num["value"]) >= float(right_num["value"]) else right_page
        winner_num = left_num if winner is left_page else right_num
        left_title = title_for_page(left_page, catalog)
        right_title = title_for_page(right_page, catalog)
        winner_title = title_for_page(winner, catalog)
        cases.append(
            make_case(
                f"{domain}_compare_{idx:02d}",
                domain,
                "multi_doc_comparison",
                f"比较《{left_title}》和《{right_title}》中各自相关章节披露的关键数值，哪一份文档中的数值更高？请分别给出两个数值和依据。",
                winner_title,
                [str(left_page.get("doc_id") or ""), str(right_page.get("doc_id") or "")],
                [str(left_page.get("page_id") or ""), str(right_page.get("page_id") or "")],
                [winner_title, winner_num["surface"]],
                [left_num["surface"], right_num["surface"]] + left_page["_keywords"][:2] + right_page["_keywords"][:2],
                [base_source(left_page, catalog, left_num["surface"]), base_source(right_page, catalog, right_num["surface"])],
            )
        )

    if len(keyword_pages) >= 2:
        first, second = keyword_pages[0], keyword_pages[1]
        first_title = title_for_page(first, catalog)
        second_title = title_for_page(second, catalog)
        keywords = first["_keywords"][:3] + second["_keywords"][:3]
        cases.append(
            make_case(
                f"{domain}_summary_01",
                domain,
                "multi_doc_summary",
                f"分别概括《{first_title}》和《{second_title}》中相关章节讨论的重点，并总结二者共同涉及的主题。",
                "；".join(keywords[:6]),
                [str(first.get("doc_id") or ""), str(second.get("doc_id") or "")],
                [str(first.get("page_id") or ""), str(second.get("page_id") or "")],
                keywords,
                keywords,
                [base_source(first, catalog), base_source(second, catalog)],
            )
        )

    if keyword_pages:
        page = keyword_pages[-1]
        title = title_for_page(page, catalog)
        cases.append(
            make_case(
                f"{domain}_insufficient_01",
                domain,
                "insufficient_evidence",
                f"根据《{title}》，请回答该文档是否披露了火星矿业机器人业务在2099年的营业收入、净利润和客户数量？如果没有充分证据，请明确说明证据不足。",
                "证据不足",
                [str(page.get("doc_id") or "")],
                [str(page.get("page_id") or "")],
                ["证据不足", "未披露", "没有充分证据"],
                page["_keywords"][:4],
                [base_source(page, catalog)],
            )
        )

    return cases[:10]


def main() -> None:
    all_cases: list[dict[str, Any]] = []
    for domain, rel_dir in DOMAIN_INDEX_DIRS.items():
        all_cases.extend(build_domain_cases(domain, REPO_ROOT / rel_dir))
    output_path = REPO_ROOT / OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as fh:
        for case in all_cases[:50]:
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"wrote={output_path}")
    print(f"cases={len(all_cases[:50])}")
    counts = defaultdict(int)
    for case in all_cases[:50]:
        counts[case["task_type"]] += 1
    print(json.dumps(dict(counts), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
