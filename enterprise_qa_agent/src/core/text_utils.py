from __future__ import annotations

import re
from typing import Any


def maybe_fix_mojibake(text: str) -> str:
    mojibake = tuple(chr(code) for code in (0x951B, 0x7ED7, 0x93C9, 0x95B2, 0x9429, 0x7039, 0x5F42, 0x20AC))
    mojibake_markers = mojibake + ("锛", "鈥", "鐨", "骞", "鏈", "鍙", "鎵", "涓", "瀹", "绗", "妯")
    if not any(marker in text for marker in mojibake_markers):
        return text
    try:
        fixed = text.encode("gb18030").decode("utf-8")
    except UnicodeError:
        return text
    common = ("\u7684", "\u7b2c", "\u516c\u53f8", "\u53d1\u884c", "\u503a\u5238", "\u4fe1\u606f", "\u62a5\u544a")
    fixed_score = sum(fixed.count(word) for word in common) * 3 - sum(fixed.count(word) for word in mojibake)
    text_score = sum(text.count(word) for word in common) * 3 - sum(text.count(word) for word in mojibake)
    return fixed if fixed_score > text_score else text


def fix_mojibake_deep(value: Any) -> Any:
    if isinstance(value, str):
        return maybe_fix_mojibake(value)
    if isinstance(value, list):
        return [fix_mojibake_deep(item) for item in value]
    if isinstance(value, dict):
        return {key: fix_mojibake_deep(item) for key, item in value.items()}
    return value


def compact_text(value: str) -> str:
    return re.sub(r"\s+", " ", maybe_fix_mojibake(value or "")).strip()
