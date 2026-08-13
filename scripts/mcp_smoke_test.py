from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enterprise_qa_agent.src.mcp_tools import calculate_finance, get_doc_outline, list_domains, search_docs, search_pages
from scripts.mcp_server import build_server


def main() -> None:
    domains = list_domains()
    docs = search_docs("平安e生保 家庭共享免赔额", "insurance", top_k=1)
    pages = search_pages("家庭共享免赔额1万元", "insurance", doc_ids=["5"], top_k=1)
    outline = get_doc_outline("insurance", "5", max_headings=3)
    calc = calculate_finance("bond_interest", principal=10000, rate=0.015, days=120)
    server = build_server()

    result = {
        "domain_count": len(domains["domains"]),
        "top_doc_id": docs["results"][0]["doc_id"] if docs["results"] else "",
        "top_page_doc_id": pages["results"][0]["doc_id"] if pages["results"] else "",
        "outline_doc_id": outline["doc_id"],
        "bond_interest": calc["result"],
        "server_type": type(server).__name__,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
