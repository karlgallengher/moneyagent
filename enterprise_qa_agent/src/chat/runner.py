from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path


# Edit these two values when you want to ask a different question from code.
CHAT_QUERY = "东方甄选相关研报中，GMV、渠道结构或自营产品表现有哪些变化？请概括主要原因。"
CHAT_DOMAIN = ""  # empty/auto, or one of: financial_contracts, financial_reports, insurance, regulatory, research


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)


from enterprise_qa_agent.src.chat.catalog_router import (
    DOMAIN_PAGE_INDEX_PATHS,
    cross_domain_catalog_search,
    infer_domain,
    rerank_chat_docs_with_llm,
    strong_catalog_hit,
)

def chat_qid(query: str) -> str:
    digest = hashlib.md5(query.encode("utf-8")).hexdigest()[:8]
    surface = re.sub(r"\W+", "_", query, flags=re.UNICODE).strip("_")[:24]
    return f"chat_{surface}_{digest}" if surface else f"chat_{digest}"


def run_chat_query(
    query: str,
    requested_domain: str = "",
    preferred_doc_ids: list[str] | None = None,
    output_dir: str | None = None,
) -> dict:
    query = query.strip()
    requested_domain = (requested_domain or "").strip()
    preferred_doc_ids = list(dict.fromkeys(preferred_doc_ids or []))
    catalog_hits: list[dict] = []
    if requested_domain in DOMAIN_PAGE_INDEX_PATHS:
        domain = requested_domain
    else:
        catalog_hits = cross_domain_catalog_search(query, top_k=int(os.environ.get("ENTERPRISE_CATALOG_SEARCH_TOP_K", "12")))
        domain = str(catalog_hits[0]["domain"]) if catalog_hits else infer_domain(query, requested_domain)

    os.environ["PAGE_INDEX_PATH"] = DOMAIN_PAGE_INDEX_PATHS[domain]
    if output_dir:
        os.environ["OUTPUT_DIR"] = output_dir
    else:
        os.environ.setdefault("OUTPUT_DIR", "enterprise_qa_agent/outputs/chat_debug")
    os.environ.setdefault("ENTERPRISE_DOC_TOP_K", "12")

    from enterprise_qa_agent.src import legacy_agent

    qid = os.environ.get("CHAT_QID", chat_qid(query))
    question = legacy_agent.chat_to_legacy_question(
        query,
        domain,
        qid,
        legacy_agent.all_doc_ids_for_loaded_index(),
        lambda doc_id: legacy_agent.doc_profile_text(doc_id, max_chars=1400),
        legacy_agent.TREE_DOC_BY_ID,
    )
    all_domain_doc_ids = legacy_agent.all_doc_ids_for_loaded_index()
    valid_preferred_doc_ids = [doc_id for doc_id in preferred_doc_ids if doc_id in legacy_agent.TREE_DOC_BY_ID]
    domain_catalog_doc_ids = [
        str(item["doc_id"])
        for item in catalog_hits
        if item.get("domain") == domain and str(item.get("doc_id") or "") in legacy_agent.TREE_DOC_BY_ID
    ]
    if valid_preferred_doc_ids:
        question["doc_ids"] = valid_preferred_doc_ids
        question["enterprise_prefiltered_doc_ids"] = valid_preferred_doc_ids
        question["enterprise_session_preferred_doc_ids"] = valid_preferred_doc_ids
    elif domain_catalog_doc_ids and strong_catalog_hit(catalog_hits):
        domain_catalog_doc_ids = list(dict.fromkeys(domain_catalog_doc_ids))[: int(os.environ.get("ENTERPRISE_DOC_TOP_K", "12"))]
        question["doc_ids"] = domain_catalog_doc_ids
        question["enterprise_prefiltered_doc_ids"] = domain_catalog_doc_ids
        question["enterprise_catalog_search"] = catalog_hits
    elif question.get("enterprise_chat") and len(question.get("enterprise_prefiltered_doc_ids") or []) <= 2:
        question["doc_ids"] = all_domain_doc_ids
        question["enterprise_prefiltered_doc_ids"] = all_domain_doc_ids
        question["enterprise_catalog_search"] = catalog_hits
        question["enterprise_catalog_low_confidence"] = True
    question = rerank_chat_docs_with_llm(query, question, legacy_agent)
    print(f"chat_qid={question['qid']}")
    print(f"domain={domain}")
    print(f"page_index={legacy_agent.PAGE_INDEX_PATH}")
    if catalog_hits:
        print(f"catalog_search={json.dumps(catalog_hits[:5], ensure_ascii=False)}")
    print(f"prefilter_doc_ids={json.dumps(question.get('enterprise_prefiltered_doc_ids', []), ensure_ascii=False)}")
    if question.get("enterprise_doc_agent"):
        print(f"doc_agent={json.dumps(question['enterprise_doc_agent'], ensure_ascii=False)}")

    app = legacy_agent.build_graph()
    final_state = app.invoke({"question": question})
    print(f"answer={final_state.get('final_answer', '')}")
    print(f"status={final_state.get('status')}")
    print(f"token_usage={json.dumps(final_state.get('token_usage', {}), ensure_ascii=False)}")
    legacy_agent.write_outputs(final_state, os.environ["OUTPUT_DIR"])
    return final_state


def main() -> None:
    query = os.environ.get("CHAT_QUERY", CHAT_QUERY).strip()
    requested_domain = os.environ.get("CHAT_DOMAIN", CHAT_DOMAIN).strip()
    run_chat_query(query, requested_domain)


if __name__ == "__main__":
    main()


