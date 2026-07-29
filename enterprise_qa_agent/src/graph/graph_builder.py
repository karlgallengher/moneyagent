from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, StateGraph


def build_enterprise_graph(
    state_type: type,
    nodes: dict[str, Callable[..., Any]],
    route_after_audit: Callable[..., str],
    route_after_advance: Callable[..., str],
    use_task_reasoning_path: Callable[[dict], bool],
):
    graph = StateGraph(state_type)
    graph.add_node("init", nodes["init"])
    graph.add_node("human_toc_reasoning", nodes["human_toc_reasoning"])
    graph.add_node("slot_planner", nodes["slot_planner"])
    graph.add_node("route_docs", nodes["route_docs"])
    graph.add_node("retrieve_initial", nodes["retrieve_initial"])
    graph.add_node("evidence_extract", nodes["evidence_extract"])
    graph.add_node("evidence_read", nodes["evidence_read"])
    graph.add_node("audit", nodes["audit"])
    graph.add_node("review_original_after_subsearch", nodes["review_original_after_subsearch"])
    graph.add_node("focus_evidence", nodes["focus_evidence"])
    graph.add_node("retrieve_more", nodes["retrieve_more"])
    graph.add_node("toc_repair", nodes["toc_repair"])
    graph.add_node("judge", nodes["judge"])
    graph.add_node("advance_option", nodes["advance_option"])
    graph.add_node("finalize", nodes["finalize"])

    graph.set_entry_point("init")
    graph.add_conditional_edges(
        "init",
        lambda state: "human_toc_reasoning" if use_task_reasoning_path(state["question"]) else "slot_planner",
        {"human_toc_reasoning": "human_toc_reasoning", "slot_planner": "slot_planner"},
    )
    graph.add_edge("human_toc_reasoning", "finalize")
    graph.add_edge("slot_planner", "route_docs")
    graph.add_edge("route_docs", "retrieve_initial")
    graph.add_edge("retrieve_initial", "evidence_extract")
    graph.add_edge("evidence_extract", "evidence_read")
    graph.add_edge("evidence_read", "audit")
    graph.add_conditional_edges(
        "audit",
        route_after_audit,
        {
            "review_original_after_subsearch": "review_original_after_subsearch",
            "focus_evidence": "focus_evidence",
            "retrieve_more": "retrieve_more",
            "toc_repair": "toc_repair",
            "judge": "judge",
        },
    )
    graph.add_edge("review_original_after_subsearch", "evidence_read")
    graph.add_edge("focus_evidence", "evidence_read")
    graph.add_edge("retrieve_more", "evidence_extract")
    graph.add_edge("toc_repair", "evidence_extract")
    graph.add_edge("judge", "advance_option")
    graph.add_conditional_edges("advance_option", route_after_advance, {"route_docs": "slot_planner", "finalize": "finalize"})
    graph.add_edge("finalize", END)
    return graph.compile()
