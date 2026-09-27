from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enterprise_qa_agent.src.chat import MoneyAgentRequest, answer_question


DEFAULT_EVAL_SET = "tests/eval_sets/moneyagent_research_20_cn.jsonl"
DEFAULT_OUTPUT = "enterprise_qa_agent/outputs/eval/moneyagent_eval_results.jsonl"
DEFAULT_AGGREGATE_OUTPUT = "enterprise_qa_agent/outputs/eval/moneyagent_eval_results_all_cases.jsonl"

# Change this number when you click "Run" in VSCode and want to inspect one case.
# It is 1-based: 1 means the first row in DEFAULT_EVAL_SET, 2 means the second row.
DEFAULT_CASE_INDEX = 10


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def hit_any(text: str, keywords: list[str]) -> bool:
    text = text or ""
    return any(str(keyword) and str(keyword) in text for keyword in keywords)


def score_case(case: dict[str, Any], response: dict[str, Any], elapsed_seconds: float) -> dict[str, Any]:
    expected_doc_ids = {str(item) for item in case.get("expected_doc_ids") or [] if item}
    actual_doc_ids = {str(item) for item in response.get("doc_ids") or [] if item}
    evidence_rows = response.get("evidence") or []
    evidence_doc_ids = {str(item.get("doc_id") or "") for item in evidence_rows if isinstance(item, dict)}
    evidence_text = "\n".join(
        str(item.get("quote") or "") + " " + str(item.get("heading") or "")
        for item in evidence_rows
        if isinstance(item, dict)
    )
    answer_text = str(response.get("answer") or "")
    reason_text = str(response.get("reason") or "")
    answer_keywords = [str(item) for item in case.get("expected_answer_keywords") or [] if item]
    evidence_keywords = [str(item) for item in case.get("expected_evidence_keywords") or [] if item]
    matched_doc_ids = expected_doc_ids & (actual_doc_ids | evidence_doc_ids)
    doc_hit = bool(matched_doc_ids) if expected_doc_ids else False
    multi_doc_coverage = expected_doc_ids <= (actual_doc_ids | evidence_doc_ids) if expected_doc_ids else False
    answer_keyword_hit = hit_any(answer_text + "\n" + reason_text, answer_keywords)
    evidence_keyword_hit = hit_any(evidence_text + "\n" + reason_text + "\n" + answer_text, evidence_keywords)
    task_type = str(case.get("task_type") or case.get("answer_type") or "")
    if task_type == "insufficient_evidence":
        passed = answer_keyword_hit
    elif len(expected_doc_ids) > 1:
        passed = multi_doc_coverage and (answer_keyword_hit or evidence_keyword_hit)
    else:
        passed = doc_hit and (answer_keyword_hit or evidence_keyword_hit)
    return {
        "id": case.get("id"),
        "domain": case.get("domain"),
        "task_type": task_type,
        "doc_hit": doc_hit,
        "multi_doc_coverage": multi_doc_coverage,
        "answer_keyword_hit": answer_keyword_hit,
        "evidence_keyword_hit": evidence_keyword_hit,
        "passed": passed,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "question": case.get("question"),
        "expected_answer": case.get("expected_answer"),
        "expected_doc_ids": sorted(expected_doc_ids),
        "response": response,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if not total:
        return {"total": 0}
    summary = {
        "total": total,
        "pass_rate": round(sum(1 for row in rows if row["passed"]) / total, 4),
        "doc_hit_rate": round(sum(1 for row in rows if row["doc_hit"]) / total, 4),
        "multi_doc_coverage_rate": round(sum(1 for row in rows if row["multi_doc_coverage"]) / total, 4),
        "answer_keyword_hit_rate": round(sum(1 for row in rows if row["answer_keyword_hit"]) / total, 4),
        "evidence_keyword_hit_rate": round(sum(1 for row in rows if row["evidence_keyword_hit"]) / total, 4),
        "avg_latency_seconds": round(sum(float(row["elapsed_seconds"]) for row in rows) / total, 3),
    }
    by_type: dict[str, dict[str, Any]] = {}
    for task_type in sorted({str(row.get("task_type") or "") for row in rows}):
        typed = [row for row in rows if str(row.get("task_type") or "") == task_type]
        by_type[task_type] = {
            "total": len(typed),
            "pass_rate": round(sum(1 for row in typed if row["passed"]) / len(typed), 4),
            "doc_hit_rate": round(sum(1 for row in typed if row["doc_hit"]) / len(typed), 4),
            "multi_doc_coverage_rate": round(sum(1 for row in typed if row["multi_doc_coverage"]) / len(typed), 4),
        }
    summary["by_task_type"] = by_type
    return summary


def load_existing_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def upsert_aggregate_results(path: Path, new_rows: list[dict[str, Any]], eval_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_id = {
        str(row.get("id") or ""): row
        for row in load_existing_results(path)
        if row.get("id")
    }
    for row in new_rows:
        row_id = str(row.get("id") or "")
        if row_id:
            rows_by_id[row_id] = row

    ordered_ids = [str(case.get("id") or "") for case in eval_cases if case.get("id")]
    ordered_rows = [rows_by_id.pop(row_id) for row_id in ordered_ids if row_id in rows_by_id]
    ordered_rows.extend(rows_by_id.values())
    write_results(path, ordered_rows)
    return ordered_rows


def print_case_detail(scored: dict[str, Any]) -> None:
    response = scored.get("response") or {}
    print("\n" + "=" * 80)
    print(f"id: {scored.get('id')}")
    print(f"domain: {scored.get('domain')} task_type: {scored.get('task_type')}")
    print(f"passed: {scored.get('passed')}")
    print(
        "hits: "
        f"doc={scored.get('doc_hit')} "
        f"answer={scored.get('answer_keyword_hit')} "
        f"evidence={scored.get('evidence_keyword_hit')}"
    )
    print("\n[Question]")
    print(scored.get("question") or "")
    print("\n[Expected]")
    print(scored.get("expected_answer") or "")
    print("\n[Agent Answer]")
    print(response.get("answer") or "")
    if response.get("reason"):
        print("\n[Reason]")
        print(response.get("reason"))
    if response.get("doc_ids"):
        print("\n[Doc IDs]")
        print(", ".join(str(item) for item in response.get("doc_ids") or []))
    if response.get("status") == "error":
        print("\n[Error]")
        print(f"{response.get('error_type', '')}: {response.get('error_message', '')}")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MoneyAgent on a JSONL eval set.")
    parser.add_argument("--eval-set", default=DEFAULT_EVAL_SET)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--aggregate-output", default=DEFAULT_AGGREGATE_OUTPUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--case-index",
        type=int,
        default=DEFAULT_CASE_INDEX,
        help=f"1-based case index to run. Default: {DEFAULT_CASE_INDEX}.",
    )
    parser.add_argument("--case-id", default="", help="Run a single case by id.")
    parser.add_argument("--all", action="store_true", help="Run all cases in the eval set.")
    parser.add_argument("--include-raw-state", action="store_true")
    args = parser.parse_args()

    all_cases = load_jsonl(REPO_ROOT / args.eval_set)
    cases = all_cases
    if args.case_id:
        cases = [case for case in cases if str(case.get("id") or "") == args.case_id]
        if not cases:
            raise SystemExit(f"case id not found: {args.case_id}")
    elif args.all:
        pass
    elif args.limit:
        cases = cases[: args.limit]
    else:
        if args.case_index < 1 or args.case_index > len(cases):
            raise SystemExit(f"--case-index must be between 1 and {len(cases)}")
        cases = [cases[args.case_index - 1]]
    output_path = REPO_ROOT / args.output

    scored_rows: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        try:
            response = answer_question(
                MoneyAgentRequest(query=str(case["question"]), domain=str(case.get("domain") or "")),
                include_raw_state=args.include_raw_state,
            ).to_dict(include_raw_state=args.include_raw_state)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            response = {
                "query": str(case.get("question") or ""),
                "qid": str(case.get("id") or ""),
                "status": "error",
                "answer": "",
                "domain": str(case.get("domain") or ""),
                "doc_ids": [],
                "evidence": [],
                "reason": "",
                "confidence": None,
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "debug_files": {},
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        elapsed = time.perf_counter() - started
        scored = score_case(case, response, elapsed)
        scored_rows.append(scored)
        print(
            f"{scored['id']} passed={scored['passed']} doc_hit={scored['doc_hit']} "
            f"answer_hit={scored['answer_keyword_hit']} evidence_hit={scored['evidence_keyword_hit']}"
        )
        print_case_detail(scored)

    write_results(output_path, scored_rows)

    summary = summarize(scored_rows)
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    aggregate_path = REPO_ROOT / args.aggregate_output
    aggregate_rows = upsert_aggregate_results(aggregate_path, scored_rows, all_cases)
    aggregate_summary = summarize(aggregate_rows)
    aggregate_summary_path = aggregate_path.with_name(aggregate_path.stem + "_summary.json")
    aggregate_summary_path.write_text(json.dumps(aggregate_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"results={output_path}")
    print(f"summary={summary_path}")
    print(f"aggregate_results={aggregate_path}")
    print(f"aggregate_summary={aggregate_summary_path}")


if __name__ == "__main__":
    main()
