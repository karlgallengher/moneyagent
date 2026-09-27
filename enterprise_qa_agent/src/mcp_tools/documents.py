from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from enterprise_qa_agent.src.core.retrieval_core import BM25, tokenize
from enterprise_qa_agent.src.core.text_utils import maybe_fix_mojibake
from enterprise_qa_agent.src.chat.domain_registry import domain_index_dirs, load_domains, UPLOAD_ROOT


REPO_ROOT = Path(__file__).resolve().parents[3]

DOMAIN_INDEX_DIRS = {
    "financial_contracts": "processed/page_index_financial_contracts",
    "financial_reports": "processed/page_index_financial_reports",
    "insurance": "processed/page_index_insurance",
    "regulatory": "processed/page_index_regulatory",
    "research": "processed/page_index_research",
}


def _resolve_domain(domain: str) -> str:
    domain = str(domain or "").strip()
    DOMAIN_INDEX_DIRS.clear()
    DOMAIN_INDEX_DIRS.update(domain_index_dirs())
    if domain not in DOMAIN_INDEX_DIRS:
        raise ValueError(f"domain must be one of {sorted(DOMAIN_INDEX_DIRS)}")
    return domain


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _domain_dir(domain: str) -> Path:
    return REPO_ROOT / DOMAIN_INDEX_DIRS[_resolve_domain(domain)]


def _text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(_text(item) for item in value)
    return maybe_fix_mojibake(str(value or ""))


def _clip_text(value: str, max_chars: int) -> str:
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 3)].rstrip() + "..."


def _catalog_text(row: dict[str, Any]) -> str:
    return " ".join(
        [
            _text(row.get("title_fixed") or row.get("title")),
            _text(row.get("aliases") or []),
            _text(row.get("top_headings") or []),
            _text(row.get("first_page_preview")),
        ]
    )


def _doc_title(row: dict[str, Any]) -> str:
    return _text(row.get("title_fixed") or row.get("title"))


def _term_hits(query: str, text: str, max_hits: int = 12) -> list[str]:
    terms = list(dict.fromkeys(tokenize(query)))
    lowered = text.lower()
    hits = []
    for term in terms:
        if not term:
            continue
        if term.lower() in lowered:
            hits.append(term)
        if len(hits) >= max_hits:
            break
    return hits


def _simple_score(query: str, text: str) -> float:
    query_terms = tokenize(query)
    if not query_terms:
        return 0.0
    counts = {}
    lowered = text.lower()
    for term in query_terms:
        counts[term] = lowered.count(term.lower())
    score = sum(min(count, 3) for count in counts.values())
    phrase_bonus = 4 if str(query).strip() and str(query).strip() in text else 0
    return float(score + phrase_bonus)


@lru_cache(maxsize=16)
def _catalog(domain: str) -> tuple[dict[str, Any], ...]:
    rows = _load_jsonl(_domain_dir(domain) / "doc_catalog.jsonl")
    return tuple(rows)


@lru_cache(maxsize=16)
def _pages(domain: str) -> tuple[dict[str, Any], ...]:
    rows = _load_jsonl(_domain_dir(domain) / "page_index.jsonl")
    return tuple(rows)


@lru_cache(maxsize=16)
def _page_bm25(domain: str) -> BM25:
    return BM25(list(_pages(domain)), text_key="index_text")


def search_docs(query: str, domain: str, top_k: int = 5) -> dict[str, Any]:
    domain = _resolve_domain(domain)
    top_k = max(1, min(int(top_k or 5), 20))
    page_doc_scores: dict[str, float] = {}
    page_doc_hits: dict[str, list[str]] = {}
    all_doc_ids = {str(row.get("doc_id") or "") for row in _pages(domain)}
    for score, page in _page_bm25(domain).score(query, all_doc_ids)[:50]:
        doc_id = str(page.get("doc_id") or "")
        page_doc_scores[doc_id] = page_doc_scores.get(doc_id, 0.0) + float(score)
        if doc_id not in page_doc_hits:
            text = _text(page.get("text") or page.get("index_text"))
            page_doc_hits[doc_id] = _term_hits(query, text)

    scored: list[dict[str, Any]] = []
    for row in _catalog(domain):
        doc_id = str(row.get("doc_id") or "")
        profile = _catalog_text(row)
        catalog_score = _simple_score(query, profile)
        page_score = page_doc_scores.get(doc_id, 0.0)
        score = catalog_score + page_score
        if score <= 0:
            continue
        matched_terms = list(dict.fromkeys(_term_hits(query, profile) + page_doc_hits.get(doc_id, [])))
        scored.append(
            {
                "domain": domain,
                "doc_id": doc_id,
                "title": _doc_title(row),
                "score": round(score, 3),
                "catalog_score": round(catalog_score, 3),
                "page_score": round(page_score, 3),
                "matched_terms": matched_terms[:12],
                "page_count": row.get("page_count"),
                "first_page_id": row.get("first_page_id"),
                "first_page_preview": _clip_text(_text(row.get("first_page_preview")), 240),
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return {"query": query, "domain": domain, "results": scored[:top_k]}


def list_domains() -> dict[str, Any]:
    DOMAIN_INDEX_DIRS.clear()
    DOMAIN_INDEX_DIRS.update(domain_index_dirs())
    custom = load_domains()
    results = []
    for domain, relative_dir in DOMAIN_INDEX_DIRS.items():
        domain_dir = REPO_ROOT / relative_dir
        catalog_rows = _catalog(domain)
        page_rows = _pages(domain)
        upload_path = UPLOAD_ROOT / domain
        results.append(
            {
                "domain": domain,
                "name": custom.get(domain, {}).get("name", domain),
                "index_dir": relative_dir,
                "catalog_path": str(domain_dir / "doc_catalog.jsonl"),
                "page_index_path": str(domain_dir / "page_index.jsonl"),
                "catalog_exists": (domain_dir / "doc_catalog.jsonl").exists(),
                "page_index_exists": (domain_dir / "page_index.jsonl").exists(),
                "doc_count": len(catalog_rows),
                "page_count": len(page_rows),
                "upload_count": len(list(upload_path.glob("*.md"))) + len(list(upload_path.glob("*.txt"))) if domain in custom else 0,
                "status": custom.get(domain, {}).get("status", "ready"),
                "last_built_at": custom.get(domain, {}).get("last_built_at", ""),
            }
        )
    for domain, item in custom.items():
        if domain not in DOMAIN_INDEX_DIRS:
            upload_path = UPLOAD_ROOT / domain
            results.append({
                "domain": domain,
                "name": item.get("name", domain),
                "index_dir": "",
                "catalog_exists": False,
                "page_index_exists": False,
                "doc_count": 0,
                "page_count": 0,
                "upload_count": len(list(upload_path.glob("*.md"))) + len(list(upload_path.glob("*.txt"))),
                "status": item.get("status", "needs_build"),
                "last_built_at": item.get("last_built_at", ""),
            })
    return {"domains": results}


def get_doc_outline(domain: str, doc_id: str, max_headings: int = 80) -> dict[str, Any]:
    domain = _resolve_domain(domain)
    doc_id = str(doc_id or "").strip()
    max_headings = max(1, min(int(max_headings or 80), 300))
    if not doc_id:
        raise ValueError("doc_id is required")

    catalog_row = None
    for row in _catalog(domain):
        if str(row.get("doc_id") or "") == doc_id:
            catalog_row = row
            break
    if catalog_row is None:
        raise ValueError(f"doc not found: domain={domain} doc_id={doc_id}")

    headings: list[dict[str, Any]] = []
    seen = set()
    doc_pages = []
    for page in _pages(domain):
        if str(page.get("doc_id") or "") != doc_id:
            continue
        doc_pages.append(page)
        heading_path = page.get("heading_path") or []
        key = json.dumps(heading_path, ensure_ascii=False)
        if heading_path and key not in seen:
            seen.add(key)
            headings.append(
                {
                    "page_id": str(page.get("page_id") or ""),
                    "heading_path": heading_path,
                }
            )
        if len(headings) >= max_headings:
            break

    aliases = catalog_row.get("aliases") or []
    top_headings = catalog_row.get("top_headings") or []
    return {
        "domain": domain,
        "doc_id": doc_id,
        "title": _doc_title(catalog_row),
        "aliases": [_text(item) for item in aliases],
        "page_count": catalog_row.get("page_count") or len(doc_pages),
        "first_page_id": catalog_row.get("first_page_id"),
        "first_page_preview": _clip_text(_text(catalog_row.get("first_page_preview")), 500),
        "top_headings": [_text(item) for item in top_headings],
        "headings": headings,
        "headings_truncated": len(headings) >= max_headings,
    }


def search_pages(
    query: str,
    domain: str,
    doc_ids: list[str] | None = None,
    top_k: int = 5,
    max_quote_chars: int = 700,
) -> dict[str, Any]:
    domain = _resolve_domain(domain)
    top_k = max(1, min(int(top_k or 5), 20))
    max_quote_chars = max(80, min(int(max_quote_chars or 700), 3000))
    all_pages = list(_pages(domain))
    allowed_doc_ids = {str(item) for item in doc_ids or [] if str(item)}
    if not allowed_doc_ids:
        allowed_doc_ids = {str(row.get("doc_id") or "") for row in all_pages}
    hits = _page_bm25(domain).score(query, allowed_doc_ids)[:top_k]
    results = []
    for score, page in hits:
        text = _text(page.get("text") or page.get("index_text"))
        results.append(
            {
                "domain": domain,
                "doc_id": str(page.get("doc_id") or ""),
                "page_id": str(page.get("page_id") or ""),
                "score": round(float(score), 3),
                "heading_path": page.get("heading_path") or [],
                "quote": _clip_text(text, max_quote_chars),
            }
        )
    return {"query": query, "domain": domain, "doc_ids": sorted(allowed_doc_ids), "results": results}


def get_page(domain: str, page_id: str = "", doc_id: str = "", max_chars: int = 5000) -> dict[str, Any]:
    domain = _resolve_domain(domain)
    page_id = str(page_id or "").strip()
    doc_id = str(doc_id or "").strip()
    max_chars = max(100, min(int(max_chars or 5000), 20000))
    for page in _pages(domain):
        if page_id and str(page.get("page_id") or "") != page_id:
            continue
        if doc_id and str(page.get("doc_id") or "") != doc_id:
            continue
        text = _text(page.get("text") or page.get("index_text"))
        return {
            "domain": domain,
            "doc_id": str(page.get("doc_id") or ""),
            "page_id": str(page.get("page_id") or ""),
            "heading_path": page.get("heading_path") or [],
            "text": _clip_text(text, max_chars),
        }
    raise ValueError(f"page not found: domain={domain} doc_id={doc_id} page_id={page_id}")
