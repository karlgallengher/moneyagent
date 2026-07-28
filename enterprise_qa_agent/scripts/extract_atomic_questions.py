from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "enterprise_qa_agent" / "data" / "atomic_questions"

A_QUESTION_DIR = PROJECT_ROOT / "public_dataset_upload" / "questions" / "group_a"
B_QUESTION_DIR = PROJECT_ROOT / "upload_b" / "question_b"


A_FILES = [
    A_QUESTION_DIR / "financial_contracts_questions.json",
    A_QUESTION_DIR / "financial_reports_questions.json",
    A_QUESTION_DIR / "insurance_questions.json",
    A_QUESTION_DIR / "regulatory_questions.json",
    A_QUESTION_DIR / "research_questions.json",
]

B_FILES = [
    B_QUESTION_DIR / "financial_contracts_b_question.json",
    B_QUESTION_DIR / "financial_reports_b_questions.jsonl",
    B_QUESTION_DIR / "insurance_b_questions.json",
    B_QUESTION_DIR / "regulatory_b_questions.jsonl",
    B_QUESTION_DIR / "research_b_question.jsonl",
]


DOMAIN_ALIASES = {
    "financial_contracts": "financial_contracts",
    "financial_reports": "financial_reports",
    "insurance": "insurance",
    "regulatory": "regulatory",
    "research": "research",
}


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("questions", "data", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        return [data]
    return []


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def infer_split(item: dict[str, Any], fallback: str) -> str:
    split = str(item.get("split") or fallback or "").strip().upper()
    return split if split in {"A", "B"} else fallback


def infer_domain(item: dict[str, Any], path: Path) -> str:
    domain = str(item.get("domain") or "").strip()
    if domain:
        return DOMAIN_ALIASES.get(domain, domain)
    name = path.name.lower()
    for key, value in DOMAIN_ALIASES.items():
        if key in name:
            return value
    if "contract" in name:
        return "financial_contracts"
    if "report" in name:
        return "financial_reports"
    return "unknown"


def infer_answer_format(item: dict[str, Any]) -> str:
    answer_format = str(item.get("answer_format") or "").strip().lower()
    if answer_format:
        return answer_format
    question_type = str(item.get("type") or "").strip()
    if question_type == "多选题":
        return "multi"
    if question_type == "单选题":
        return "mcq"
    if question_type == "判断题":
        return "tf"
    if question_type == "计算题":
        return "free"
    if question_type == "抽取题":
        return "free"
    return "unknown"


def infer_task_type(item: dict[str, Any], option_text: str | None = None) -> str:
    question = normalize_text(item.get("question", ""))
    source_type = str(item.get("type") or "")
    combined = normalize_text(" ".join(part for part in (question, option_text or "") if part))
    if source_type == "计算题" or any(term in combined for term in ("计算", "多少", "合计", "比例", "占比", "增速", "收益率")):
        return "calculation"
    if source_type == "抽取题" or any(term in combined for term in ("抽取", "填写", "日期", "名称", "文号", "施行", "发布")):
        return "fact_extraction"
    if any(term in combined for term in ("排序", "排名", "从高到低", "从低到高", "大于号")):
        return "ranking"
    if any(term in combined for term in ("比较", "高于", "低于", "大于", "小于", "多于", "少于", "最准确", "哪两项")):
        return "comparison"
    if any(term in combined for term in ("适用", "责任免除", "责任范围", "规定", "规则", "监管", "处罚", "违法", "应当", "不得")):
        return "rule_applicability"
    return "claim_verification"


GENERIC_TRUE_FALSE = {
    "正确",
    "错误",
    "对",
    "错",
    "是",
    "否",
    "正确。",
    "错误。",
}


def is_generic_tf_options(options: dict[str, Any]) -> bool:
    if not options:
        return False
    values = {normalize_text(value) for value in options.values()}
    return values.issubset(GENERIC_TRUE_FALSE) and len(values) <= 2


def extract_entities(text: str) -> list[str]:
    text = normalize_text(text)
    entities: list[str] = []
    quoted = re.findall(r"[“\"《]([^”\"》]{2,30})[”\"》]", text)
    entities.extend(quoted)
    candidates = re.findall(
        r"[\u4e00-\u9fffA-Za-z0-9]{2,30}(?:公司|集团|银行|保险|基金|证券|股份|科技|发展|仪器|激光|光模块|产品|合同|规则|办法|通知|条例)",
        text,
    )
    entities.extend(candidates)
    for marker in ("领域", "行业", "产品", "公司", "机构"):
        pattern = rf"([\u4e00-\u9fffA-Za-z0-9、/和及与]{2,80}){marker}"
        for match in re.findall(pattern, text):
            for part in re.split(r"[、/和及与]", match):
                part = re.sub(r"^(?:在|及|与|和|都|各|以下|关于|某|一家|一个|几款|四款|三类|两类)+", "", part.strip())
                part = re.sub(r"(?:中|内|上|下|里)$", "", part)
                if 2 <= len(part) <= 20:
                    entities.append(part)
    for pattern in (
        r"在([\u4e00-\u9fffA-Za-z0-9、/和及与]{2,60})(?:领域|行业)",
        r"([\u4e00-\u9fffA-Za-z0-9]{2,20})、([\u4e00-\u9fffA-Za-z0-9]{2,20})和([\u4e00-\u9fffA-Za-z0-9]{2,20})(?:领域|行业)",
    ):
        for match in re.findall(pattern, text):
            parts = match if isinstance(match, tuple) else re.split(r"[、/和及与]", match)
            for part in parts:
                part = re.sub(r"^(?:在|及|与|和|都|各)+", "", str(part).strip())
                if 2 <= len(part) <= 20:
                    entities.append(part)
    return list(dict.fromkeys(entities))[:12]


METRIC_TERMS = (
    "营业收入",
    "净利润",
    "归母净利润",
    "研发投入",
    "研发费用",
    "现金流",
    "保费",
    "现金价值",
    "保险金",
    "赔付比例",
    "市场规模",
    "增速",
    "收益率",
    "成本",
    "费用率",
    "毛利率",
    "市占率",
)


def extract_metrics(text: str) -> list[str]:
    text = normalize_text(text)
    metrics = [term for term in METRIC_TERMS if term in text]
    metrics.extend(re.findall(r"\d{4}年[^，。；;]{0,16}(?:收入|利润|规模|增速|比例|成本|费用|现金价值)", text))
    return list(dict.fromkeys(metrics))[:12]


def extract_constraints(text: str) -> list[str]:
    text = normalize_text(text)
    constraints: list[str] = []
    if any(term in text for term in ("都", "均", "全部", "各自", "分别", "四款", "三类", "两类", "两者")):
        constraints.append("需要覆盖题干涉及的所有对象，不能只验证其中一部分。")
    if any(term in text for term in ("同时", "并且", "既", "又", "双重")):
        constraints.append("需要同时满足复合条件，不能只满足其中一个条件。")
    if any(term in text for term in ("高于", "低于", "大于", "小于", "超过", "不超过")):
        constraints.append("需要核对比较方向、单位和统计口径。")
    if any(term in text for term in ("第", "年度", "周岁", "日期", "施行", "发布")):
        constraints.append("需要核对时间、年度、年龄或生效日期条件。")
    if any(term in text for term in ("只填写", "保留两位", "%", "大于号", "排序")):
        constraints.append("需要按问题要求输出格式化答案。")
    return constraints


def atomic_question_for_option(question: str, label: str, option_text: str, task_type: str) -> str:
    if task_type in {"comparison", "rule_applicability"}:
        prefix = "请核验并说明依据"
    elif task_type == "calculation":
        prefix = "请根据文档核验或计算"
    else:
        prefix = "请核验"
    return f"{prefix}：在“{question}”这个问题背景下，选项{label}的陈述是否成立：{option_text}"


def atomic_question_for_free(question: str, task_type: str) -> str:
    if task_type == "calculation":
        return f"请根据企业文档计算并回答：{question}"
    if task_type == "ranking":
        return f"请根据企业文档完成排序并回答：{question}"
    return f"请根据企业文档抽取或回答：{question}"


def build_expected_schema(task_type: str) -> dict[str, Any]:
    if task_type == "claim_verification":
        answer = "supported|contradicted|insufficient"
    elif task_type == "rule_applicability":
        answer = "applicable|not_applicable|insufficient"
    elif task_type == "calculation":
        answer = "calculated_value"
    elif task_type == "ranking":
        answer = "ordered_result"
    else:
        answer = "extracted_value"
    return {"answer": answer, "evidence": [], "reason": ""}


def build_atomic_from_question(item: dict[str, Any], split: str, domain: str) -> list[dict[str, Any]]:
    qid = str(item.get("qid") or "").strip()
    question = normalize_text(item.get("question", ""))
    source_type = str(item.get("type") or "")
    answer_format = infer_answer_format(item)
    options = item.get("options") or {}
    doc_ids = item.get("doc_ids") or []
    rows: list[dict[str, Any]] = []

    if isinstance(options, dict) and options and not is_generic_tf_options(options):
        for label in sorted(options):
            option_text = normalize_text(options[label])
            combined = f"{question} {option_text}"
            task_type = infer_task_type(item, option_text)
            rows.append(
                {
                    "atomic_id": f"{qid}_{label}",
                    "source_qid": qid,
                    "source_split": split,
                    "domain": domain,
                    "source_type": source_type,
                    "source_answer_format": answer_format,
                    "task_type": task_type,
                    "question": atomic_question_for_option(question, label, option_text, task_type),
                    "entities": extract_entities(combined),
                    "metrics": extract_metrics(combined),
                    "constraints": extract_constraints(combined),
                    "expected_answer_schema": build_expected_schema(task_type),
                    "source_trace": {
                        "original_question": question,
                        "option_label": label,
                        "option_text": option_text,
                        "source_doc_ids": doc_ids,
                    },
                    "metadata": {
                        "has_options": True,
                        "original_options_count": len(options),
                    },
                }
            )
        return rows

    task_type = infer_task_type(item)
    rows.append(
        {
            "atomic_id": qid,
            "source_qid": qid,
            "source_split": split,
            "domain": domain,
            "source_type": source_type,
            "source_answer_format": answer_format,
            "task_type": task_type,
            "question": atomic_question_for_free(question, task_type),
            "entities": extract_entities(question),
            "metrics": extract_metrics(question),
            "constraints": extract_constraints(question),
            "expected_answer_schema": build_expected_schema(task_type),
            "source_trace": {
                "original_question": question,
                "source_doc_ids": doc_ids,
            },
            "metadata": {
                "has_options": bool(options),
                "generic_true_false_options": is_generic_tf_options(options) if isinstance(options, dict) else False,
            },
        }
    )
    return rows


def load_group(files: list[Path], split: str) -> list[dict[str, Any]]:
    atomic_rows: list[dict[str, Any]] = []
    for path in files:
        rows = read_json_or_jsonl(path)
        for item in rows:
            if not isinstance(item, dict):
                continue
            domain = infer_domain(item, path)
            item_split = infer_split(item, split)
            atomic_rows.extend(build_atomic_from_question(item, item_split, domain))
    return atomic_rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split = Counter(row.get("source_split") for row in rows)
    by_domain = Counter(row.get("domain") for row in rows)
    by_task_type = Counter(row.get("task_type") for row in rows)
    source_qids = {row.get("source_qid") for row in rows}
    return {
        "atomic_questions": len(rows),
        "source_questions": len(source_qids),
        "by_split": dict(sorted(by_split.items())),
        "by_domain": dict(sorted(by_domain.items())),
        "by_task_type": dict(sorted(by_task_type.items())),
    }


def main() -> None:
    a_rows = load_group(A_FILES, "A")
    b_rows = load_group(B_FILES, "B")
    all_rows = a_rows + b_rows

    write_jsonl(OUT_DIR / "atomic_questions_a.jsonl", a_rows)
    write_jsonl(OUT_DIR / "atomic_questions_b.jsonl", b_rows)
    write_jsonl(OUT_DIR / "atomic_questions_all.jsonl", all_rows)

    summary = {
        "A": summarize(a_rows),
        "B": summarize(b_rows),
        "all": summarize(all_rows),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
