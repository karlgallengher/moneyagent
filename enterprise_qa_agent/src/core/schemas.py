from __future__ import annotations

from typing import Any, Literal, TypedDict


TaskType = Literal[
    "claim_verification",
    "fact_extraction",
    "calculation",
    "comparison",
    "ranking",
    "rule_applicability",
]


class SourceTrace(TypedDict, total=False):
    original_question: str
    option_label: str
    option_text: str
    source_doc_ids: list[str]


class ExpectedAnswerSchema(TypedDict, total=False):
    answer: str
    evidence: list[dict[str, Any]]
    reason: str


class AtomicQuestion(TypedDict, total=False):
    atomic_id: str
    source_qid: str
    source_split: str
    domain: str
    source_type: str
    source_answer_format: str
    task_type: TaskType
    question: str
    entities: list[str]
    metrics: list[str]
    constraints: list[str]
    expected_answer_schema: ExpectedAnswerSchema
    source_trace: SourceTrace
    metadata: dict[str, Any]

