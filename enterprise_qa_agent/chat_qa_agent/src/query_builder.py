from __future__ import annotations

from .memory_policy import format_evidence_memory_for_prompt, select_evidence_memory_for_route
from .router import build_memory_context
from .utils import safe_text


def looks_contextual(query: str) -> bool:
    text = safe_text(query)
    if len(text) <= 24:
        return True
    markers = (
        "它", "他们", "它们", "这个", "这些", "上述", "上面", "刚才", "继续", "再说", "那",
        "第二个", "前一个", "后一个", "那篇", "该文档", "这篇", "和它比", "它们的",
    )
    return any(marker in text for marker in markers) and len(text) <= 120


def looks_like_expansion(query: str) -> bool:
    text = safe_text(query)
    markers = ("再和", "相比", "对比", "另外", "新增", "换成", "加入", "扩展到", "再比较")
    return any(marker in text for marker in markers)


def resolve_query(query: str, state: dict) -> dict:
    query = safe_text(query)
    has_history = bool(state.get("turn_window") or state.get("global_summary"))
    is_followup = has_history and looks_contextual(query)
    memory_context = build_memory_context(state) if has_history else ""

    if not has_history:
        resolved_query = query
    else:
        resolved_query = "\n".join(
            part
            for part in (
                "Use the following conversation memory to interpret the current question. If the current question introduces a new subject or document, do not restrict retrieval to old documents. If it is a follow-up, prefer the prior topic.",
                memory_context,
                "Current question: " + query,
            )
            if part
        )

    active_doc_ids = [str(item) for item in state.get("active_doc_ids") or [] if item]
    use_active_docs = bool(active_doc_ids) and is_followup and not looks_like_expansion(query)
    return {
        "resolved_query": resolved_query,
        "is_followup": is_followup,
        "use_active_docs": use_active_docs,
        "preferred_doc_ids": active_doc_ids if use_active_docs else [],
        "reason": "contextual follow-up" if is_followup else "standalone or expanded query",
    }


def build_rag_query_from_route(user_query: str, route: dict, state: dict | None = None) -> str:
    query_type = str(route.get("query_type") or "standalone")
    base_query = safe_text(route.get("standalone_query"), 1800) or safe_text(user_query, 1800)
    route = dict(route)
    route["current_query"] = user_query
    memory_subjects = [safe_text(item, 100) for item in route.get("memory_subjects") or [] if safe_text(item)]
    new_subjects = [safe_text(item, 100) for item in route.get("new_subjects") or [] if safe_text(item)]
    evidence_prompt = format_evidence_memory_for_prompt(select_evidence_memory_for_route(state or {}, route))

    if query_type == "followup_same_scope" and memory_subjects:
        return "\n".join(
            part
            for part in [
                "会话消解后的企业问答问题。",
                "问题类型：同范围追问。",
                "记忆主体：" + "；".join(memory_subjects),
                "用户追问：" + safe_text(user_query, 1000),
                "任务：只在上述记忆主体范围内回答；先明确代词指代，再分别检索和比较各主体证据。若证据记忆已覆盖当前槽位可以优先参考，不足时继续检索。",
                evidence_prompt,
                "独立问题：" + base_query,
            ]
            if part
        )

    if query_type == "followup_expand_scope" and (memory_subjects or new_subjects):
        lines = [
            "会话消解后的企业问答问题。",
            "问题类型：扩展范围追问。",
        ]
        if memory_subjects:
            lines.append("保留上文主体：" + "；".join(memory_subjects))
        if new_subjects:
            lines.append("新增需检索主体：" + "；".join(new_subjects))
        lines.append("用户追问：" + safe_text(user_query, 1000))
        lines.append("任务：同时覆盖上文主体和新增主体；证据记忆只能用于上文主体，新增主体必须重新检索；不要把答案限制在旧文档内。")
        if evidence_prompt:
            lines.append(evidence_prompt)
        lines.append("独立问题：" + base_query)
        return "\n".join(lines)

    return base_query
