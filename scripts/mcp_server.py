from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enterprise_qa_agent.src.mcp_tools import calculate_finance, get_page, search_docs, search_pages


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def build_server() -> Any:
    try:
        from mcp.server import MCPServer
    except ImportError as exc:
        try:
            from mcp.server.fastmcp import FastMCP as MCPServer
        except ImportError:
            raise SystemExit(
                "The 'mcp' package is not installed or is incompatible. Install dependencies with:\n"
                "  pip install -r requirements.txt"
            ) from exc

    mcp = MCPServer("moneyagent")

    @mcp.tool()
    def search_docs_tool(query: str, domain: str, top_k: int = 5) -> str:
        """Search MoneyAgent document catalogs by domain."""
        return _json(search_docs(query=query, domain=domain, top_k=top_k))

    @mcp.tool()
    def search_pages_tool(
        query: str,
        domain: str,
        doc_ids: list[str] | None = None,
        top_k: int = 5,
        max_quote_chars: int = 700,
    ) -> str:
        """Search page-level evidence in a domain, optionally limited to doc_ids."""
        return _json(
            search_pages(
                query=query,
                domain=domain,
                doc_ids=doc_ids or [],
                top_k=top_k,
                max_quote_chars=max_quote_chars,
            )
        )

    @mcp.tool()
    def get_page_tool(domain: str, page_id: str = "", doc_id: str = "", max_chars: int = 5000) -> str:
        """Get one indexed page by page_id or doc_id."""
        return _json(get_page(domain=domain, page_id=page_id, doc_id=doc_id, max_chars=max_chars))

    @mcp.tool()
    def calculate_finance_tool(operation: str, params: dict[str, Any]) -> str:
        """Run deterministic finance calculations such as ratio, yoy_reverse, bond_interest, or deductible_claim."""
        return _json(calculate_finance(operation, **(params or {})))

    return mcp


def main() -> None:
    mcp = build_server()
    mcp.run()


if __name__ == "__main__":
    main()
