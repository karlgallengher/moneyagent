from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)


DEFAULT_QUERY = "请概括这个金融知识库可以回答哪些类型的问题。"
DEFAULT_EVAL_SET = "tests/eval_sets/moneyagent_research_20_cn.jsonl"

# VSCode click-run defaults. Edit these values, then click Run.
# RUN_MODE: "chat", "session", "eval", "mcp", or "web-api"
RUN_MODE = "session"
RUN_QUERY = ""
RUN_DOMAIN = ""
RUN_CASE_INDEX = 1
RUN_CASE_ID = ""
RUN_EVAL_ALL = False
RUN_SESSION_ID = "default"
RUN_RESET_SESSION = False


def run_chat(args: argparse.Namespace) -> None:
    from enterprise_qa_agent.src.chat import MoneyAgentRequest, answer_question

    query = str(args.query or "").strip() or DEFAULT_QUERY
    response = answer_question(
        MoneyAgentRequest(
            query=query,
            domain=str(args.domain or ""),
            preferred_doc_ids=tuple(args.doc_id or []),
            output_dir=str(args.output_dir),
        ),
        include_raw_state=bool(args.include_raw_state),
    )
    data = response.to_dict(include_raw_state=bool(args.include_raw_state))
    print(f"qid={data.get('qid', '')}")
    print(f"domain={data.get('domain', '')}")
    print(f"doc_ids={data.get('doc_ids', [])}")
    print(f"status={data.get('status', '')}")
    print(f"token_usage={data.get('token_usage', {})}")
    print("\nanswer>")
    print(data.get("answer", ""))


def run_session(args: argparse.Namespace) -> None:
    if args.session_id:
        os.environ["SESSION_ID"] = str(args.session_id)
    if args.domain is not None:
        os.environ["CHAT_DOMAIN"] = str(args.domain)
    if args.reset:
        os.environ["RESET_SESSION"] = "1"
    else:
        os.environ.pop("RESET_SESSION", None)
    if args.query:
        os.environ["CHAT_QUERY"] = str(args.query)
    else:
        os.environ["CHAT_QUERY"] = ""

    from enterprise_qa_agent.chat_qa_agent.src.session_runner import main

    main()


def run_eval(args: argparse.Namespace) -> None:
    script = REPO_ROOT / "scripts" / "evaluate_moneyagent.py"
    argv = [
        str(script),
        "--eval-set",
        str(args.eval_set),
        "--output",
        str(args.output),
        "--aggregate-output",
        str(args.aggregate_output),
    ]
    if args.all:
        argv.append("--all")
    elif args.limit:
        argv.extend(["--limit", str(args.limit)])
    elif args.case_id:
        argv.extend(["--case-id", str(args.case_id)])
    else:
        argv.extend(["--case-index", str(args.case_index)])
    if args.include_raw_state:
        argv.append("--include-raw-state")

    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv


def run_mcp(args: argparse.Namespace) -> None:
    script = REPO_ROOT / "scripts" / "mcp_server.py"
    print("Starting MoneyAgent MCP server over stdio...")
    runpy.run_path(str(script), run_name="__main__")


def run_web_api(args: argparse.Namespace) -> None:
    from scripts.web_api import main

    main()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified entrypoint for MoneyAgent.",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    chat = subparsers.add_parser("chat", help="Run one single-turn question.")
    chat.add_argument("-q", "--query", default=DEFAULT_QUERY)
    chat.add_argument("--domain", default="")
    chat.add_argument("--doc-id", action="append", default=[])
    chat.add_argument("--output-dir", default="enterprise_qa_agent/outputs/chat_debug")
    chat.add_argument("--include-raw-state", action="store_true")
    chat.set_defaults(func=run_chat)

    session = subparsers.add_parser("session", help="Run multi-turn chat with SQLite memory.")
    session.add_argument("-q", "--query", default="", help="Optional one-shot turn. Empty starts interactive chat.")
    session.add_argument("--domain", default="")
    session.add_argument("--session-id", default="default")
    session.add_argument("--reset", action="store_true")
    session.set_defaults(func=run_session)

    eval_parser = subparsers.add_parser("eval", help="Run evaluation cases.")
    eval_parser.add_argument("--eval-set", default=DEFAULT_EVAL_SET)
    eval_parser.add_argument("--output", default="enterprise_qa_agent/outputs/eval/moneyagent_eval_results.jsonl")
    eval_parser.add_argument(
        "--aggregate-output",
        default="enterprise_qa_agent/outputs/eval/moneyagent_eval_results_all_cases.jsonl",
    )
    eval_parser.add_argument("--case-index", type=int, default=1)
    eval_parser.add_argument("--case-id", default="")
    eval_parser.add_argument("--limit", type=int, default=0)
    eval_parser.add_argument("--all", action="store_true")
    eval_parser.add_argument("--include-raw-state", action="store_true")
    eval_parser.set_defaults(func=run_eval)

    mcp_parser = subparsers.add_parser("mcp", help="Run the MoneyAgent MCP stdio server.")
    mcp_parser.set_defaults(func=run_mcp)

    web_api_parser = subparsers.add_parser("web-api", help="Run the MoneyAgent FastAPI backend for the Web UI.")
    web_api_parser.set_defaults(func=run_web_api)

    return parser


def default_args_for_click_run() -> list[str]:
    if RUN_MODE == "chat":
        args = ["chat", "--query", RUN_QUERY]
        if RUN_DOMAIN:
            args.extend(["--domain", RUN_DOMAIN])
        return args
    if RUN_MODE == "session":
        args = ["session", "--session-id", RUN_SESSION_ID]
        if RUN_QUERY:
            args.extend(["--query", RUN_QUERY])
        if RUN_DOMAIN:
            args.extend(["--domain", RUN_DOMAIN])
        if RUN_RESET_SESSION:
            args.append("--reset")
        return args
    if RUN_MODE == "eval":
        args = ["eval", "--eval-set", DEFAULT_EVAL_SET]
        if RUN_EVAL_ALL:
            args.append("--all")
        elif RUN_CASE_ID:
            args.extend(["--case-id", RUN_CASE_ID])
        else:
            args.extend(["--case-index", str(RUN_CASE_INDEX)])
        return args
    if RUN_MODE == "mcp":
        return ["mcp"]
    if RUN_MODE == "web-api":
        return ["web-api"]
    raise SystemExit(f"Unsupported RUN_MODE: {RUN_MODE}")


def main() -> None:
    try:
        parser = build_parser()
        cli_args = sys.argv[1:] or default_args_for_click_run()
        args = parser.parse_args(cli_args)
        args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted")
    except SystemExit:
        raise
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
