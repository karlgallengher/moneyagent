from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from enterprise_qa_agent.src.io.output_utils import (
    enterprise_reason_from_state,
    evidence_summary_from_state,
    facts_summary_from_state,
    question_info_summary_from_state,
)


@dataclass(frozen=True)
class MoneyAgentRequest:
    query: str
    domain: str = ""
    session_id: str = ""
    preferred_doc_ids: tuple[str, ...] = ()
    output_dir: str = "enterprise_qa_agent/outputs/chat_debug"


@dataclass(frozen=True)
class MoneyAgentResponse:
    query: str
    qid: str
    status: str
    answer: str
    domain: str
    doc_ids: list[str]
    evidence: list[dict[str, Any]]
    facts: list[dict[str, Any]]
    question_info: list[dict[str, Any]]
    reason: str
    confidence: float | None
    token_usage: dict[str, int]
    debug_files: dict[str, str]
    error_type: str = ""
    error_message: str = ""
    raw_state: dict[str, Any] | None = None

    def to_dict(self, include_raw_state: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if not include_raw_state:
            data.pop("raw_state", None)
        return data


def _debug_files(qid: str, output_dir: str) -> dict[str, str]:
    output_path = Path(output_dir)
    return {
        "json": str(output_path / f"langgraph_{qid}_state.json"),
        "markdown": str(output_path / f"langgraph_{qid}.md"),
    }


def _confidence_from_state(state: dict[str, Any]) -> float | None:
    judgment = state.get("reasoning_judgment") or {}
    value = judgment.get("confidence")
    if value is None:
        for option_state in (state.get("option_states") or {}).values():
            option_judgment = option_state.get("judgment") or {}
            value = option_judgment.get("confidence")
            if value is not None:
                break
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def state_to_response(
    state: dict[str, Any],
    query: str,
    output_dir: str,
    include_raw_state: bool = False,
) -> MoneyAgentResponse:
    question = state.get("question") or {}
    qid = str(state.get("qid") or question.get("qid") or "")
    return MoneyAgentResponse(
        query=query,
        qid=qid,
        status=str(state.get("status") or ""),
        answer=str(state.get("final_answer") or ""),
        domain=str(question.get("enterprise_domain") or question.get("domain") or ""),
        doc_ids=[str(item) for item in question.get("enterprise_prefiltered_doc_ids") or question.get("doc_ids") or []],
        evidence=evidence_summary_from_state(state),
        facts=facts_summary_from_state(state),
        question_info=question_info_summary_from_state(state),
        reason=enterprise_reason_from_state(state),
        confidence=_confidence_from_state(state),
        token_usage={
            "prompt_tokens": int((state.get("token_usage") or {}).get("prompt_tokens") or 0),
            "completion_tokens": int((state.get("token_usage") or {}).get("completion_tokens") or 0),
            "total_tokens": int((state.get("token_usage") or {}).get("total_tokens") or 0),
        },
        debug_files=_debug_files(qid, output_dir),
        raw_state=state if include_raw_state else None,
    )


def answer_question(
    request: MoneyAgentRequest,
    include_raw_state: bool = False,
) -> MoneyAgentResponse:
    from enterprise_qa_agent.src.chat.runner import run_chat_query

    try:
        final_state = run_chat_query(
            request.query,
            requested_domain=request.domain,
            preferred_doc_ids=list(request.preferred_doc_ids),
            output_dir=request.output_dir,
        )
        return state_to_response(
            final_state,
            query=request.query,
            output_dir=request.output_dir,
            include_raw_state=include_raw_state,
        )
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        qid = str(request.query or "").strip()
        return MoneyAgentResponse(
            query=request.query,
            qid=qid,
            status="error",
            answer="",
            domain=str(request.domain or ""),
            doc_ids=[],
            evidence=[],
            facts=[],
            question_info=[],
            reason="",
            confidence=None,
            token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            debug_files=_debug_files(qid, request.output_dir),
            error_type=type(exc).__name__,
            error_message=str(exc),
            raw_state=None,
        )
