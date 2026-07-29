from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from enterprise_qa_agent.src.core.text_utils import maybe_fix_mojibake


REQUEST_TIMEOUT_SECONDS = 180
REQUEST_RETRY_TIMES = 3
VERBOSE_CONSOLE = True
DRY_RUN_WITHOUT_LLM = os.environ.get("DRY_RUN_WITHOUT_LLM", "0").lower() in {"1", "true", "yes"}

DASHSCOPE_API_KEY_ENV = "DEEPSEEK_API_KEY"
DASHSCOPE_API_KEY_FILE = "api_ds"
DASHSCOPE_BASE_URL = "https://api.deepseek.com"
# DASHSCOPE_API_KEY_ENV = "DASHSCOPE_API_KEY"
# DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# QWEN_MODEL = "qwen3.6-plus"
# QWEN_MODEL = "qwen3.5-plus-2026-04-20"

SPARKAI_API_KEY_ENV = DASHSCOPE_API_KEY_ENV
SPARKAI_API_KEY_FILE = DASHSCOPE_API_KEY_FILE
SPARKAI_BASE_URL = DASHSCOPE_BASE_URL
SPARKAI_CHAT_URL = f"{SPARKAI_BASE_URL.rstrip('/')}/chat/completions"
QWEN_MODEL = os.environ.get("QWEN_MODEL", "deepseek-chat")


def load_sparkai_api_key() -> str:
    api_key = os.getenv(SPARKAI_API_KEY_ENV)
    if api_key:
        return api_key.strip()
    key_file = Path(SPARKAI_API_KEY_FILE)
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()
    raise RuntimeError(f"Set {SPARKAI_API_KEY_ENV} or create {SPARKAI_API_KEY_FILE}.")


def extract_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"No JSON object found: {text[:300]}")
    return json.loads(match.group(0))


def dry_json(system_prompt: str, user_prompt: str, purpose: str) -> dict:
    if purpose == "slot_plan":
        return {
            "slots": [
                {
                    "slot": "dry run slot",
                    "queries": [maybe_fix_mojibake(user_prompt)[:180]],
                    "key_terms": [],
                }
            ]
        }
    if purpose == "router":
        return {"target_doc_ids": [], "doc_reasons": {}, "reason": "dry run"}
    if purpose == "audit":
        return {"can_judge": True, "filled_slots": [], "missing_slots": []}
    if purpose == "judge":
        return {"verdict": False, "confidence": 0.0, "reason": "dry run", "answer": "F"}
    return {}


def call_qwen_json(system_prompt: str, user_prompt: str) -> tuple[dict, dict[str, int]]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Missing dependency: openai. Run `pip install openai`.") from exc

    api_key = load_sparkai_api_key()
    if VERBOSE_CONSOLE:
        print(f"[spark request] model={QWEN_MODEL} chars={len(system_prompt) + len(user_prompt)}", flush=True)
    client = OpenAI(api_key=api_key, base_url=SPARKAI_BASE_URL)
    last_error: Exception | None = None
    for attempt in range(1, REQUEST_RETRY_TIMES + 1):
        try:
            response = client.chat.completions.create(
                model=QWEN_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            break
        except Exception as exc:
            last_error = exc
            if VERBOSE_CONSOLE:
                print(f"[spark request error] attempt={attempt}/{REQUEST_RETRY_TIMES} error={exc}", flush=True)
            if attempt < REQUEST_RETRY_TIMES:
                time.sleep(2 * attempt)
    else:
        raise RuntimeError(f"SparkAI request failed after {REQUEST_RETRY_TIMES} attempts: {last_error}") from last_error

    usage = getattr(response, "usage", None) or {}
    if VERBOSE_CONSOLE:
        total_tokens = getattr(usage, "total_tokens", 0) if not isinstance(usage, dict) else usage.get("total_tokens", 0)
        print(f"[spark response] total_tokens={total_tokens}", flush=True)
    content = response.choices[0].message.content or ""
    usage_dict = (
        usage
        if isinstance(usage, dict)
        else {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }
    )
    return extract_json_object(content), {
        "prompt_tokens": int(usage_dict.get("prompt_tokens") or 0),
        "completion_tokens": int(usage_dict.get("completion_tokens") or 0),
        "total_tokens": int(usage_dict.get("total_tokens") or 0),
    }

