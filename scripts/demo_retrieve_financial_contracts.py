from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


QUESTIONS_PATH = "public_dataset_upload/questions/group_a/financial_contracts_questions.json"
PAGE_INDEX_PATH = "processed/page_index_financial_contracts/page_index.jsonl"
OUTPUT_PATH = "processed/retrieval_demo/financial_contracts_retrieval.md"
TOP_K = 8
OPTION_TOP_K = 5
PREVIEW_CHARS = 20000


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


def load_json(path: str):
    text = Path(path).read_text(encoding="utf-8")
    text = maybe_fix_mojibake(text)
    return json.loads(text)


def load_jsonl(path: str) -> list[dict]:
    rows = []
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


def detect_target_doc_ids(text: str, doc_ids: list[str]) -> list[str]:
    text = maybe_fix_mojibake(text)
    if not doc_ids:
        return []
    first_markers = ("第一份", "第一个", "文档一", "第一份文档", "首份")
    second_markers = ("第二份", "第二个", "文档二", "第二份文档")
    third_markers = ("第三份", "第三个", "文档三", "第三份文档")
    all_markers = ("两份", "二者", "均", "都", "分别", "对比", "相比", "低于", "高于", "两者")

    targets: list[str] = []
    if any(marker in text for marker in first_markers) and len(doc_ids) >= 1:
        targets.append(doc_ids[0])
    if any(marker in text for marker in second_markers) and len(doc_ids) >= 2:
        targets.append(doc_ids[1])
    if any(marker in text for marker in third_markers) and len(doc_ids) >= 3:
        targets.append(doc_ids[2])
    if any(marker in text for marker in all_markers):
        targets.extend(doc_ids)
    if not targets:
        targets = list(doc_ids)

    deduped: list[str] = []
    seen: set[str] = set()
    for doc_id in targets:
        if doc_id not in seen:
            deduped.append(doc_id)
            seen.add(doc_id)
    return deduped


def score_by_doc_targets(bm25: BM25, query: str, doc_ids: list[str], top_k_per_doc: int) -> list[tuple[float, dict]]:
    hits: list[tuple[float, dict]] = []
    for doc_id in detect_target_doc_ids(query, doc_ids):
        hits.extend(bm25.score(query, allowed_doc_ids={doc_id})[:top_k_per_doc])
    return sorted(hits, key=lambda item: item[0], reverse=True)


def build_query(question: dict) -> str:
    option_text = "\n".join(f"{key}. {value}" for key, value in question.get("options", {}).items())
    return "\n".join([question.get("question", ""), option_text])


def build_option_query(question: dict, option_key: str, option_value: str) -> str:
    return "\n".join([question.get("question", ""), f"{option_key}. {option_value}"])


def short(text: str, limit: int = PREVIEW_CHARS) -> str:
    text = re.sub(r"\s+", " ", maybe_fix_mojibake(text)).strip()
    return text if len(text) <= limit else text[:limit] + "..."


def append_hits(lines: list[str], hits: list[tuple[float, dict]]) -> None:
    if not hits:
        lines.append("_No hits._")
        lines.append("")
        return
    for rank, (score, page) in enumerate(hits, start=1):
        heading = " > ".join(page.get("heading_path") or [])
        lines.append(f"{rank}. `{page['doc_id']}` `{page['page_id']}` score={score:.3f}")
        lines.append(f"   - heading: {heading}")
        lines.append(f"   - text: {short(page.get('text', ''))}")
    lines.append("")


def main() -> None:
    questions = load_json(QUESTIONS_PATH)
    pages = load_jsonl(PAGE_INDEX_PATH)
    bm25 = BM25(pages)

    lines: list[str] = ["# Financial Contracts Retrieval Demo", ""]
    for question in questions:
        qid = question["qid"]
        allowed = set(question.get("doc_ids") or [])
        query = build_query(question)
        hits = score_by_doc_targets(bm25, query, list(allowed), TOP_K)

        lines.append(f"## {qid}")
        lines.append("")
        lines.append(f"**doc_ids:** {', '.join(sorted(allowed))}")
        lines.append("")
        lines.append(f"**question:** {maybe_fix_mojibake(question.get('question', ''))}")
        lines.append("")
        for key, value in question.get("options", {}).items():
            lines.append(f"- {key}. {maybe_fix_mojibake(value)}")
        lines.append("")
        lines.append("### Common retrieval")
        lines.append("")
        append_hits(lines, hits)

        lines.append("### Option retrieval")
        lines.append("")
        for key, value in question.get("options", {}).items():
            option_query = build_option_query(question, key, value)
            option_hits = score_by_doc_targets(bm25, option_query, list(allowed), OPTION_TOP_K)
            lines.append(f"#### Option {key}")
            lines.append("")
            lines.append(f"query: {maybe_fix_mojibake(option_query)}")
            lines.append("")
            append_hits(lines, option_hits)

    output = Path(OUTPUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"questions={len(questions)}")
    print(f"pages={len(pages)}")
    print(f"output={output.resolve()}")


if __name__ == "__main__":
    main()
