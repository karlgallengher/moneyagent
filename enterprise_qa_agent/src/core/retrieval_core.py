from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from enterprise_qa_agent.src.core.text_utils import maybe_fix_mojibake


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

