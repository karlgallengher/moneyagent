from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


TARGET_QID = "fc_a_001"
QUESTIONS_PATH = "public_dataset_upload/questions/group_a/financial_contracts_questions.json"
PAGE_INDEX_PATH = "processed/page_index_financial_contracts/page_index.jsonl"
OUTPUT_MD = "processed/agent_debug/fc_a_001.md"
OUTPUT_JSON = "processed/agent_debug/fc_a_001_state.json"
INITIAL_TOP_K = 4
FOLLOWUP_TOP_K = 3
MAX_AUDIT_ROUNDS = 2
USE_MOCK_LLM = True


def maybe_fix_mojibake(text: str) -> str:
    try:
        fixed = text.encode("gb18030").decode("utf-8")
    except UnicodeError:
        return text
    common = ("的", "第", "公司", "发行", "债券", "信息", "报告")
    mojibake = tuple(chr(code) for code in (0x951B, 0x7ED7, 0x93C9, 0x95B2, 0x9429, 0x7039, 0x5F42, 0x20AC))
    fixed_score = sum(fixed.count(word) for word in common) * 3 - sum(fixed.count(word) for word in mojibake)
    text_score = sum(text.count(word) for word in common) * 3 - sum(text.count(word) for word in mojibake)
    return fixed if fixed_score > text_score else text


def load_json(path: str) -> Any:
    text = maybe_fix_mojibake(Path(path).read_text(encoding="utf-8"))
    return json.loads(text)


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

    def score(self, query: str, allowed_doc_ids: set[str] | None = None) -> list[tuple[float, dict]]:
        query_terms = Counter(tokenize(query))
        scored: list[tuple[float, dict]] = []
        for idx, doc in enumerate(self.docs):
            if allowed_doc_ids and doc["doc_id"] not in allowed_doc_ids:
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


@dataclass
class Evidence:
    evidence_id: str
    doc_id: str
    page_id: str
    heading_path: list[str]
    text: str
    score: float
    source: str


@dataclass
class OptionState:
    option: str
    claim: str
    status: str = "pending"
    evidence: list[Evidence] = field(default_factory=list)
    audit: dict[str, Any] = field(default_factory=dict)
    judgment: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuestionState:
    qid: str
    question: str
    answer_format: str
    doc_ids: list[str]
    status: str
    options: dict[str, OptionState]
    final_answer: str = ""


def evidence_from_hit(score: float, page: dict, source: str) -> Evidence:
    return Evidence(
        evidence_id=page["page_id"],
        doc_id=page["doc_id"],
        page_id=page["page_id"],
        heading_path=page.get("heading_path") or [],
        text=maybe_fix_mojibake(page.get("text", "")),
        score=score,
        source=source,
    )


def add_unique_evidence(option_state: OptionState, new_items: list[Evidence]) -> None:
    seen = {item.evidence_id for item in option_state.evidence}
    for item in new_items:
        if item.evidence_id in seen:
            continue
        option_state.evidence.append(item)
        seen.add(item.evidence_id)


def retrieve(bm25: BM25, query: str, allowed_doc_ids: set[str], top_k: int, source: str) -> list[Evidence]:
    hits = bm25.score(query, allowed_doc_ids=allowed_doc_ids)[:top_k]
    return [evidence_from_hit(score, page, source) for score, page in hits]


def format_evidence(evidence: list[Evidence], max_chars: int = 5000) -> str:
    blocks: list[str] = []
    used = 0
    for item in evidence:
        text = re.sub(r"\s+", " ", item.text).strip()
        remain = max_chars - used
        if remain <= 0:
            break
        if len(text) > remain:
            text = text[:remain]
        heading = " > ".join(item.heading_path)
        blocks.append(f"[{item.evidence_id}] doc={item.doc_id} heading={heading}\n{text}")
        used += len(text)
    return "\n\n".join(blocks)


def mock_auditor(question: dict, option: str, option_state: OptionState) -> dict:
    claim = maybe_fix_mojibake(option_state.claim)
    evidence_text = "\n".join(item.text for item in option_state.evidence)
    missing: list[str] = []
    required: list[str] = []
    filled: list[dict] = []

    if "发行人" in claim or "发行主体" in claim:
        required.append("发行人/发行主体")
    if "发行金额" in claim or "发行规模" in claim or "上限" in claim:
        required.append("发行金额或发行规模")
    if "评级" in claim or "AAA" in claim or "AA+" in claim:
        required.append("主体信用评级")
    if "受托管理人" in claim or "国信证券" in claim:
        required.append("受托管理人")

    for slot in required:
        if any(term in evidence_text for term in slot.split("或")) or ("主体信用评级" == slot and ("AAA" in evidence_text or "AA+" in evidence_text)):
            filled.append({"slot": slot, "value": "see evidence", "evidence_id": option_state.evidence[0].evidence_id if option_state.evidence else ""})
        else:
            missing.append(slot)

    followup = [f"{maybe_fix_mojibake(question['question'])} {claim} {slot}" for slot in missing]
    return {
        "required_slots": required,
        "filled_slots": filled,
        "missing_slots": missing,
        "can_judge": not missing,
        "followup_queries": followup,
        "mode": "mock",
    }


def mock_judge(option: str, option_state: OptionState) -> dict:
    claim = maybe_fix_mojibake(option_state.claim)
    evidence_text = "\n".join(item.text for item in option_state.evidence)
    verdict = None
    reason = "mock judge could not determine"

    if option == "A":
        verdict = "广东省广晟控股集团有限公司" in evidence_text
        reason = "evidence contains doc1 issuer name" if verdict else "issuer name not found"
    elif option == "B":
        verdict = ("不超过10亿元" in evidence_text or "10 亿元" in evidence_text) and ("不超过5亿元" in evidence_text or "5亿元" in evidence_text)
        reason = "evidence shows doc1 10亿元 and doc2 5亿元" if verdict else "amount comparison evidence incomplete"
    elif option == "C":
        verdict = False if "AA+" in evidence_text else None
        reason = "doc2 evidence contains AA+, so not both AAA" if verdict is False else "rating evidence incomplete"
    elif option == "D":
        verdict = "国信证券股份有限公司" in evidence_text and "受托管理人" in evidence_text
        reason = "evidence contains 国信证券股份有限公司 as 受托管理人" if verdict else "trustee evidence incomplete"

    return {
        "option": option,
        "status": "done" if verdict is not None else "failed",
        "verdict": verdict,
        "confidence": 0.9 if verdict is not None else 0.0,
        "reason": reason,
        "evidence_ids": [item.evidence_id for item in option_state.evidence],
        "mode": "mock",
    }


def call_qwen_json(system_prompt: str, user_prompt: str) -> dict:
    raise NotImplementedError("Qwen API is not configured yet. Set USE_MOCK_LLM=True or implement this function.")


def audit_evidence(question: dict, option: str, option_state: OptionState) -> dict:
    if USE_MOCK_LLM or not os.getenv("DASHSCOPE_API_KEY"):
        return mock_auditor(question, option, option_state)
    system = "You are an evidence auditor. Return strict JSON only."
    user = f"""
Question:
{maybe_fix_mojibake(question['question'])}

Option {option}:
{maybe_fix_mojibake(option_state.claim)}

Evidence:
{format_evidence(option_state.evidence)}

Return JSON with keys: required_slots, filled_slots, missing_slots, can_judge, followup_queries.
"""
    return call_qwen_json(system, user)


def judge_option(question: dict, option: str, option_state: OptionState) -> dict:
    if USE_MOCK_LLM or not os.getenv("DASHSCOPE_API_KEY"):
        return mock_judge(option, option_state)
    system = "You judge whether one option is correct based only on evidence. Return strict JSON only."
    user = f"""
Question:
{maybe_fix_mojibake(question['question'])}

Option {option}:
{maybe_fix_mojibake(option_state.claim)}

Audit:
{json.dumps(option_state.audit, ensure_ascii=False)}

Evidence:
{format_evidence(option_state.evidence)}

Return JSON with keys: option, status, verdict, confidence, reason, evidence_ids.
"""
    return call_qwen_json(system, user)


def finalize_answer(state: QuestionState) -> str:
    true_options = [key for key in "ABCD" if key in state.options and state.options[key].judgment.get("verdict") is True]
    if state.answer_format == "multi":
        return "".join(true_options)
    if state.answer_format in {"mcq", "tf"}:
        return true_options[0] if true_options else ""
    return "".join(true_options)


def run_agent(question: dict, bm25: BM25) -> QuestionState:
    options = {
        key: OptionState(option=key, claim=maybe_fix_mojibake(value))
        for key, value in question.get("options", {}).items()
    }
    state = QuestionState(
        qid=question["qid"],
        question=maybe_fix_mojibake(question["question"]),
        answer_format=question["answer_format"],
        doc_ids=question.get("doc_ids") or [],
        status="running",
        options=options,
    )
    allowed = set(state.doc_ids)

    for option, option_state in state.options.items():
        option_state.status = "retrieving"
        query = f"{state.question}\n{option}. {option_state.claim}"
        add_unique_evidence(option_state, retrieve(bm25, query, allowed, INITIAL_TOP_K, "initial"))

        for _round in range(MAX_AUDIT_ROUNDS):
            option_state.status = "auditing"
            option_state.audit = audit_evidence(question, option, option_state)
            if option_state.audit.get("can_judge"):
                break
            queries = option_state.audit.get("followup_queries") or []
            if not queries:
                break
            option_state.status = "retrieving_more"
            for followup_query in queries:
                add_unique_evidence(option_state, retrieve(bm25, followup_query, allowed, FOLLOWUP_TOP_K, "followup"))

        option_state.status = "judging"
        option_state.judgment = judge_option(question, option, option_state)
        option_state.status = option_state.judgment.get("status", "done")

    state.final_answer = finalize_answer(state)
    state.status = "done" if all(item.status == "done" for item in state.options.values()) else "partial"
    return state


def state_to_dict(state: QuestionState) -> dict:
    return {
        "qid": state.qid,
        "question": state.question,
        "answer_format": state.answer_format,
        "doc_ids": state.doc_ids,
        "status": state.status,
        "final_answer": state.final_answer,
        "options": {
            key: {
                "claim": value.claim,
                "status": value.status,
                "audit": value.audit,
                "judgment": value.judgment,
                "evidence": [item.__dict__ for item in value.evidence],
            }
            for key, value in state.options.items()
        },
    }


def write_report(state: QuestionState) -> None:
    lines: list[str] = [f"# Agent Debug: {state.qid}", ""]
    lines.append(f"status: `{state.status}`")
    lines.append(f"final_answer: `{state.final_answer}`")
    lines.append("")
    lines.append(f"question: {state.question}")
    lines.append("")
    for option, option_state in state.options.items():
        lines.append(f"## Option {option}")
        lines.append("")
        lines.append(f"claim: {option_state.claim}")
        lines.append(f"status: `{option_state.status}`")
        lines.append("")
        lines.append("audit:")
        lines.append("```json")
        lines.append(json.dumps(option_state.audit, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
        lines.append("judgment:")
        lines.append("```json")
        lines.append(json.dumps(option_state.judgment, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
        lines.append("evidence:")
        for item in option_state.evidence:
            heading = " > ".join(item.heading_path)
            lines.append(f"- `{item.doc_id}` `{item.page_id}` score={item.score:.3f} source={item.source}")
            lines.append(f"  - heading: {heading}")
            lines.append(f"  - text: {re.sub(r'\\s+', ' ', item.text).strip()[:1200]}")
        lines.append("")

    output_md = Path(OUTPUT_MD)
    output_json = Path(OUTPUT_JSON)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    output_json.write_text(json.dumps(state_to_dict(state), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"answer={state.final_answer}")
    print(f"state={state.status}")
    print(f"report={output_md.resolve()}")
    print(f"json={output_json.resolve()}")


def main() -> None:
    questions = load_json(QUESTIONS_PATH)
    question = next(item for item in questions if item["qid"] == TARGET_QID)
    pages = load_jsonl(PAGE_INDEX_PATH)
    bm25 = BM25(pages)
    state = run_agent(question, bm25)
    write_report(state)


if __name__ == "__main__":
    main()
