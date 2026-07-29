from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser

from enterprise_qa_agent.src.core.text_utils import compact_text


class CompactHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"tr", "p", "div", "br", "h1", "h2", "h3", "h4", "li"}:
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"tr", "p", "div", "h1", "h2", "h3", "h4", "li"}:
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append(" | ")

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.parts.append(text)

    def text(self) -> str:
        text = "".join(self.parts)
        text = re.sub(r"[ \t]*\|[ \t]*", " | ", text)
        text = re.sub(r"(?:[ \t]*\|[ \t]*){2,}", " | ", text)
        lines = [re.sub(r"\s+", " ", line).strip(" |") for line in text.splitlines()]
        return "\n".join(line for line in lines if line)


def compact_html_text(text: str, compact_html: bool = True) -> str:
    if not compact_html or "<" not in text or ">" not in text:
        return text
    parser = CompactHTMLParser()
    try:
        parser.feed(text)
        compacted = parser.text()
    except Exception:
        compacted = re.sub(r"<[^>]+>", " ", text)
    compacted = unescape(compacted)
    compacted = re.sub(r"\n{3,}", "\n\n", compacted).strip()
    return compacted or text


def format_evidence(evidence: list[dict], max_chars: int = 7000) -> str:
    chunks: list[str] = []
    used = 0
    for item in evidence:
        text = compact_html_text(item.get("text", ""))
        text = re.sub(r"[ \t]+", " ", text).strip()
        if used >= max_chars:
            break
        remaining = max_chars - used
        if len(text) > remaining:
            text = text[:remaining]
        heading = " > ".join(item.get("heading_path") or [])
        chunks.append(f"[{item['evidence_id']}] doc={item['doc_id']} heading={heading}\n{text}")
        used += len(text)
    return "\n\n".join(chunks)


def format_evidence_notes(notes: list[dict], max_chars: int = 1800) -> str:
    if not notes:
        return "[]"
    lines: list[str] = []
    used = 0
    for note in notes:
        text = compact_text(str(note.get("note") or ""))
        if not text:
            continue
        line = (
            f"- [{note.get('source_evidence_id')}] doc={note.get('doc_id')} "
            f"heading={note.get('heading')}: {text}"
        )
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines) if lines else "[]"


def print_evidence_preview(
    evidence: list[dict],
    print_text: bool = True,
    preview_chars: int = 700,
) -> None:
    for ev in evidence:
        heading = " > ".join(ev.get("heading_path") or [])
        print(f"- {ev['doc_id']} {ev['page_id']} score={ev['score']:.3f} heading={heading}", flush=True)
        if print_text:
            text = re.sub(r"\s+", " ", ev.get("text", "")).strip()
            preview = text[:preview_chars].encode("gbk", errors="replace").decode("gbk")
            print(f"  text: {preview}", flush=True)


def format_memory_slots(memory_slots: list[dict]) -> str:
    if not memory_slots:
        return "[]"
    lines = []
    for item in memory_slots:
        evidence_ids = ", ".join(item.get("evidence_ids") or [])
        lines.append(f"- {item.get('slot')}: {item.get('value')} [{evidence_ids}]")
    return "\n".join(lines)

