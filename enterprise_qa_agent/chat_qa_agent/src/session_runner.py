from __future__ import annotations

import json
import os
import sqlite3

from .config import CHAT_DOMAIN, CHAT_QUERY, DB_PATH, RESET_SESSION, SESSION_ID
from .memory_policy import select_evidence_memory_for_route
from .memory_store import (
    cleanup_cold_evidence_memory,
    connect_db,
    ensure_session,
    init_db,
    load_session_state,
    mark_evidence_memory_used,
    maybe_update_global_summary,
    reset_session,
    save_local_turn,
    save_turn_and_memory,
)
from .query_builder import build_rag_query_from_route, resolve_query
from .router import route_input_intent
from .utils import safe_text


def handle_direct_answer(conn: sqlite3.Connection, session_id: str, query: str, route: dict) -> bool:
    if str(route.get("route") or "") != "direct_answer":
        return False
    answer = safe_text(route.get("direct_answer"), 1200)
    if not answer:
        return False
    turn_index = save_local_turn(conn, session_id, query, answer, route.get("token_usage") or {})
    print(f"\n[router] route=direct_answer reason={route.get('reason', '')}")
    print(f"assistant> {answer}")
    print(f"[session] saved_turn={turn_index} direct_answer=True token_usage={json.dumps(route.get('token_usage') or {}, ensure_ascii=False)}")
    return True


def print_startup(session_id: str, requested_domain: str) -> None:
    print(f"Enterprise chat session started. session_id={session_id}")
    print(f"db_path={DB_PATH}")
    print(f"domain={requested_domain or 'auto'}")
    print("Type a question and press Enter. Type exit, quit, or q to stop. Type /reset to clear this session.")


def run_one_turn(conn: sqlite3.Connection, session_id: str, requested_domain: str, query: str) -> None:
    state = load_session_state(conn, session_id)
    deleted_cold = cleanup_cold_evidence_memory(conn, session_id, int(state.get("turn_count") or 0))
    route = route_input_intent(query, state)
    if handle_direct_answer(conn, session_id, query, route):
        return

    from enterprise_qa_agent.scripts.enterprise_chat_agent import run_chat_query

    route_for_memory = dict(route)
    route_for_memory["current_query"] = query
    selected_evidence_memory = select_evidence_memory_for_route(state, route_for_memory)
    rag_query = build_rag_query_from_route(query, route, state)
    resolved = resolve_query(rag_query, state)
    resolved["resolved_query"] = rag_query
    retrieval_scope = str(route.get("retrieval_scope") or "fresh_search")
    if retrieval_scope == "active_docs_only" and state.get("active_doc_ids"):
        resolved["use_active_docs"] = True
        resolved["preferred_doc_ids"] = [str(item) for item in state.get("active_doc_ids") or [] if item]
    elif retrieval_scope in {"active_docs_plus_new_search", "fresh_search"}:
        resolved["use_active_docs"] = False
        resolved["preferred_doc_ids"] = []

    print(f"\n[router] route=knowledge_qa query_type={route.get('query_type', '')} scope={retrieval_scope} reason={route.get('reason', '')}")
    if route.get("memory_subjects") or route.get("new_subjects"):
        print(f"[router] memory_subjects={json.dumps(route.get('memory_subjects') or [], ensure_ascii=False)} new_subjects={json.dumps(route.get('new_subjects') or [], ensure_ascii=False)}")
    if route.get("router_error"):
        print(f"[router] fallback_error={route.get('router_error')}")
    print(f"[session] turn_count={state['turn_count']} followup={resolved['is_followup']} active_docs={resolved['use_active_docs']}")
    if deleted_cold:
        print(f"[session] evidence_memory_deleted_cold={deleted_cold}")
    if resolved.get("preferred_doc_ids"):
        print(f"[session] preferred_doc_ids={json.dumps(resolved['preferred_doc_ids'], ensure_ascii=False)}")
    if selected_evidence_memory:
        used_slots = list(
            dict.fromkeys(safe_text(row.get("slot"), 80) for row in selected_evidence_memory if safe_text(row.get("slot")))
        )
        print(f"[session] evidence_memory_used={len(selected_evidence_memory)} slots={json.dumps(used_slots[:5], ensure_ascii=False)}")

    domain_for_query = requested_domain
    if not domain_for_query and retrieval_scope == "active_docs_only":
        domain_for_query = str(state.get("active_domain") or "")

    final_state = run_chat_query(
        str(resolved["resolved_query"]),
        domain_for_query,
        preferred_doc_ids=list(resolved.get("preferred_doc_ids") or []),
        output_dir="enterprise_qa_agent/outputs/chat_debug",
    )
    router_usage = route.get("token_usage") or {}
    if router_usage:
        usage = final_state.setdefault("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage[key] = int(usage.get(key) or 0) + int(router_usage.get(key) or 0)
    turn_index = save_turn_and_memory(conn, session_id, query, resolved, final_state)
    mark_evidence_memory_used(conn, session_id, selected_evidence_memory)
    maybe_update_global_summary(conn, session_id, turn_index)
    print(f"\nassistant> {final_state.get('final_answer', '')}")
    print(f"[session] saved_turn={turn_index} token_usage={json.dumps(final_state.get('token_usage', {}), ensure_ascii=False)}")


def main() -> None:
    session_id = os.environ.get("SESSION_ID", SESSION_ID).strip() or "default"
    requested_domain = os.environ.get("CHAT_DOMAIN", CHAT_DOMAIN).strip()
    reset = os.environ.get("RESET_SESSION", str(RESET_SESSION)).lower() in {"1", "true", "yes", "y"}
    one_shot_query = os.environ.get("CHAT_QUERY", CHAT_QUERY).strip()

    with connect_db() as conn:
        init_db(conn)
        if reset:
            reset_session(conn, session_id)
            ensure_session(conn, session_id)

        if one_shot_query:
            run_one_turn(conn, session_id, requested_domain, one_shot_query)
            return

        print_startup(session_id, requested_domain)
        while True:
            try:
                query = input("\nuser> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye")
                break
            if not query:
                continue
            if query.lower() in {"exit", "quit", "q"}:
                print("bye")
                break
            if query == "/reset":
                reset_session(conn, session_id)
                ensure_session(conn, session_id)
                print(f"[session] reset session_id={session_id}")
                continue
            run_one_turn(conn, session_id, requested_domain, query)
