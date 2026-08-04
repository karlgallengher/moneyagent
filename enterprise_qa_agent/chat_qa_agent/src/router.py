from __future__ import annotations

import re

from .config import MAX_FACTS_IN_CONTEXT, MAX_SUMMARY_CHARS
from .utils import clean_str_list, json_dumps, json_loads, safe_text


def build_memory_context(state: dict) -> str:
    parts: list[str] = []
    summary = safe_text(state.get("global_summary"), MAX_SUMMARY_CHARS)
    if summary:
        parts.append("Long-term summary: " + summary)

    facts = []
    for row in state.get("facts") or []:
        subject = safe_text(row.get("subject"), 80)
        fact = safe_text(row.get("fact"), 260)
        if fact:
            facts.append(f"{subject}: {fact}" if subject else fact)
    if facts:
        parts.append("Known facts: " + "; ".join(facts[-MAX_FACTS_IN_CONTEXT:]))

    turns = []
    for row in state.get("turn_window") or []:
        user_query = safe_text(row.get("user_query"), 220)
        answer = safe_text(row.get("assistant_answer"), 420)
        doc_ids = json_loads(row.get("doc_ids"), [])
        doc_text = ", docs=" + ",".join(str(item) for item in doc_ids[:8]) if doc_ids else ""
        if user_query or answer:
            turns.append(f"User: {user_query}{doc_text}\nAssistant: {answer}")
    if turns:
        parts.append("Recent turns:\n" + "\n".join(turns))
    return "\n".join(parts)


def normalize_intent_text(query: str) -> str:
    return re.sub(r"[\s,.;:!?\u3001\uff0c\u3002\uff1b\uff1a\uff01\uff1f]+", "", safe_text(query).lower())


def local_intent_answer(intent: str) -> str:
    if intent == "help":
        return (
            "你可以直接问需要查文档的问题，也可以连续追问。"
            "例如：比较两篇研报的增长驱动因素；再问它们的成本优势区别。"
            "输入 /reset 清空当前会话，输入 exit 退出。"
        )
    return (
        "你好，我是这个企业知识库问答 Agent。"
        "我会基于本地文档索引进行检索、证据分析和多轮追问记忆；"
        "适合问保险条款、监管规则、研报、财报、合同公告等文档问题。"
    )


def make_route(
    route: str,
    standalone_query: str,
    reason: str = "",
    *,
    direct_answer: str = "",
    query_type: str = "standalone",
    retrieval_scope: str = "fresh_search",
    memory_subjects: list[str] | None = None,
    new_subjects: list[str] | None = None,
    use_active_docs: bool = False,
    usage: dict | None = None,
    router_error: str = "",
) -> dict:
    return {
        "route": route if route in {"direct_answer", "knowledge_qa"} else "knowledge_qa",
        "query_type": query_type if query_type in {"standalone", "followup_same_scope", "followup_expand_scope", "followup_switch_scope"} else "standalone",
        "retrieval_scope": retrieval_scope if retrieval_scope in {"active_docs_only", "active_docs_plus_new_search", "fresh_search"} else "fresh_search",
        "direct_answer": safe_text(direct_answer, 1200),
        "standalone_query": safe_text(standalone_query, 1800),
        "memory_subjects": memory_subjects or [],
        "new_subjects": new_subjects or [],
        "use_active_docs": bool(use_active_docs),
        "reason": safe_text(reason, 300),
        "token_usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "router_error": router_error,
    }


def fallback_rule_route(query: str) -> dict:
    compact = normalize_intent_text(query)
    is_short = len(compact) <= 24
    greeting_hit = any(marker in compact for marker in ("你好", "您好", "hello", "hi", "嗨", "在吗"))
    identity_hit = any(marker in compact for marker in ("你是谁", "你是誰", "介绍一下你自己", "自我介绍"))
    help_hit = compact in {"帮助", "怎么用", "你能做什么", "help"}
    if help_hit:
        return make_route("direct_answer", query, "fallback help request", direct_answer=local_intent_answer("help"))
    if is_short and (greeting_hit or identity_hit):
        return make_route("direct_answer", query, "fallback greeting or identity request", direct_answer=local_intent_answer("local_chat"))
    return make_route("knowledge_qa", query, "fallback requires document retrieval")


def route_input_intent(query: str, state: dict | None = None) -> dict:
    state = state or {}
    memory_context = build_memory_context(state)
    active_doc_ids = [str(item) for item in state.get("active_doc_ids") or [] if item]
    active_subjects = [str(item) for item in state.get("active_subjects") or [] if item]
    system = (
        "You are the entry router for an enterprise document QA assistant. Return JSON only. "
        "First decide whether the user message can be answered directly without retrieval. "
        "Direct answers are only for greetings, identity/meta/help, or casual conversation that needs no enterprise documents. "
        "Questions about insurance clauses, regulations, research reports, financial reports, contracts, numbers, dates, comparisons, or evidence must use knowledge_qa. "
        "For knowledge_qa, classify the relationship to conversation memory: standalone, followup_same_scope, followup_expand_scope, or followup_switch_scope. "
        "followup_same_scope means the user asks about the same remembered subjects/documents; use active_docs_only. "
        "followup_expand_scope means the user keeps remembered subjects but adds new subjects for comparison; use active_docs_plus_new_search and rewrite standalone_query to include both old and new subjects. "
        "followup_switch_scope means the user changes to a new topic/domain; use fresh_search. "
        "Never use active_docs_only when the user introduces a new comparison target. If unsure, use fresh_search."
    )
    schema = {
        "route": "direct_answer|knowledge_qa",
        "query_type": "standalone|followup_same_scope|followup_expand_scope|followup_switch_scope",
        "retrieval_scope": "active_docs_only|active_docs_plus_new_search|fresh_search",
        "direct_answer": "",
        "standalone_query": "",
        "memory_subjects": [],
        "new_subjects": [],
        "reason": "",
    }
    user = "\n".join(
        [
            "Output schema:",
            json_dumps(schema),
            "",
            "Examples:",
            "User: hello -> route=direct_answer.",
            "User: What is the cost advantage difference between them? -> query_type=followup_same_scope, retrieval_scope=active_docs_only.",
            "User: Compare them with 3D printing cost reduction in consumer electronics. -> query_type=followup_expand_scope, retrieval_scope=active_docs_plus_new_search, standalone_query must include remembered subjects and the new 3D printing subject.",
            "User: Now check outpatient injury insurance compensation. -> query_type=followup_switch_scope, retrieval_scope=fresh_search.",
            "",
            "Conversation memory:",
            safe_text(memory_context, 1800) or "(none)",
            "",
            "Active subjects: " + json_dumps(active_subjects[:12]),
            "Active doc ids: " + json_dumps(active_doc_ids[:12]),
            "User message: " + safe_text(query, 1000),
        ]
    )
    try:
        from enterprise_qa_agent.src.runtime.llm_client import call_qwen_json

        output, usage = call_qwen_json(system, user)
    except Exception as exc:
        route = fallback_rule_route(query)
        route["router_error"] = str(exc)[:300]
        return route

    route_name = str(output.get("route") or "").strip()
    direct_answer = safe_text(output.get("direct_answer"), 1200)
    if route_name == "direct_answer" and direct_answer:
        return make_route(
            "direct_answer",
            safe_text(output.get("standalone_query"), 1800) or query,
            safe_text(output.get("reason"), 300),
            direct_answer=direct_answer,
            query_type=str(output.get("query_type") or "standalone"),
            retrieval_scope="fresh_search",
            memory_subjects=clean_str_list(output.get("memory_subjects")),
            new_subjects=clean_str_list(output.get("new_subjects")),
            usage=usage,
        )

    query_type = str(output.get("query_type") or "standalone").strip()
    retrieval_scope = str(output.get("retrieval_scope") or "fresh_search").strip()
    if query_type == "followup_same_scope":
        retrieval_scope = "active_docs_only"
    elif query_type == "followup_expand_scope":
        retrieval_scope = "active_docs_plus_new_search"
    elif query_type in {"standalone", "followup_switch_scope"}:
        retrieval_scope = "fresh_search"
    standalone_query = safe_text(output.get("standalone_query"), 1800) or safe_text(query, 1800)
    memory_subjects = clean_str_list(output.get("memory_subjects"))
    new_subjects = clean_str_list(output.get("new_subjects"))
    if retrieval_scope == "active_docs_plus_new_search" and (memory_subjects or new_subjects):
        subject_text = "; ".join(memory_subjects + new_subjects)
        if subject_text and subject_text not in standalone_query:
            standalone_query = f"{standalone_query}\nComparison subjects to cover: {subject_text}"
    return make_route(
        "knowledge_qa",
        standalone_query,
        safe_text(output.get("reason"), 300),
        query_type=query_type,
        retrieval_scope=retrieval_scope,
        memory_subjects=memory_subjects,
        new_subjects=new_subjects,
        use_active_docs=bool(active_doc_ids) and retrieval_scope == "active_docs_only",
        usage=usage,
    )
