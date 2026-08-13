from __future__ import annotations

from typing import Any


def ask_moneyagent(
    query: str,
    domain: str = "",
    preferred_doc_ids: list[str] | None = None,
    include_raw_state: bool = False,
) -> dict[str, Any]:
    from enterprise_qa_agent.src.chat.api import MoneyAgentRequest, answer_question

    query = str(query or "").strip()
    if not query:
        raise ValueError("query is required")

    response = answer_question(
        MoneyAgentRequest(
            query=query,
            domain=str(domain or "").strip(),
            preferred_doc_ids=tuple(str(item) for item in preferred_doc_ids or [] if str(item)),
            output_dir="enterprise_qa_agent/outputs/mcp_chat_debug",
        ),
        include_raw_state=include_raw_state,
    )
    return response.to_dict(include_raw_state=include_raw_state)
