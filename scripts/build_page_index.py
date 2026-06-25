from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# Edit this config, then click VSCode's "Run Python File" button.
# DEFAULT_INPUT can be a single .md file or a directory.
DEFAULT_INPUT = "public_dataset_upload/raw_md/research"
DEFAULT_OUTPUT = "processed/page_index_research"
DEFAULT_MAX_CHARS = 1800
DEFAULT_OVERLAP_CHARS = 160
DEFAULT_LIMIT = 0
DEFAULT_FIX_MOJIBAKE = True
DEFAULT_SPLIT_MODE = "heading"  # "heading", "heading_plus_numbered", or "block".


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"\d+(?:\.\d+)+[\.、]?\s+\S+|"
    r"[一二三四五六七八九十]+[、.]\s*\S+|"
    r"[（(][一二三四五六七八九十]+[）)]\s*\S+|"
    r"[（(]\d+[）)]\s*\S+|"
    r"\d+[）)]\s*\S+|"
    r"\d+[、.]\s*\S+"
    r")"
)
CHINESE_NUM = "\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u5343\u4e07\u96f6\u3007"
CLAUSE_RE = re.compile(rf"(\u7b2c[{CHINESE_NUM}\d]+[\u7ae0\u8282\u6761\u6b3e\u9879])")
DATE_RE = re.compile(r"\d{4}\u5e74(?:\d{1,2}\u6708(?:\d{1,2}\u65e5)?)?")
NUMBER_RE = re.compile(
    rf"(?:\d+(?:\.\d+)?%|\d+(?:\.\d+)?(?:\u4e07|\u4ebf)?"
    rf"(?:\u5143|\u80a1|\u5428|\u5bb6|\u4e2a|\u9879|\u500d|\u5e74|\u4e2a\u6708|\u65e5)|"
    rf"[{CHINESE_NUM}]+\u5206\u4e4b[{CHINESE_NUM}]+)"
)
IMAGE_RE = re.compile(r"!\[[^\]]*]\([^)]+\)")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
DETAILS_RE = re.compile(r"<details>\s*<summary>\s*(.*?)\s*</summary>\s*(.*?)</details>", re.S | re.I)


@dataclass
class Block:
    kind: str
    text: str
    char_start: int
    char_end: int
    heading_path: list[str] = field(default_factory=list)
    level: int = 0


def maybe_fix_mojibake(text: str) -> str:
    # Common case: UTF-8 Chinese text was decoded as GBK/CP936 before being saved.
    try:
        fixed = text.encode("gb18030").decode("utf-8")
    except UnicodeError:
        return text
    common = ("\u7684", "\u7b2c", "\u516c\u53f8", "\u53d1\u884c", "\u503a\u5238", "\u4fe1\u606f", "\u62a5\u544a")
    mojibake = tuple(chr(code) for code in (0x951B, 0x7ED7, 0x93C9, 0x95B2, 0x9429, 0x7039, 0x5F42, 0x20AC))
    fixed_score = sum(fixed.count(word) for word in common) * 3 - sum(fixed.count(word) for word in mojibake)
    text_score = sum(text.count(word) for word in common) * 3 - sum(text.count(word) for word in mojibake)
    return fixed if fixed_score > text_score else text


def stable_id(text: str, length: int = 10) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]


def clean_line(line: str) -> str:
    return IMAGE_RE.sub("", line).rstrip()


def unwrap_details(text: str) -> str:
    def replace(match: re.Match) -> str:
        summary = re.sub(r"\s+", " ", match.group(1)).strip()
        body = match.group(2).strip()
        if summary:
            return f"\n\n[details: {summary}]\n{body}\n"
        return f"\n\n{body}\n"

    return DETAILS_RE.sub(replace, text)


def normalize_text(text: str, fix_mojibake: bool) -> str:
    if fix_mojibake:
        text = maybe_fix_mojibake(text)
    text = HTML_COMMENT_RE.sub("", text).replace("\ufeff", "")
    text = unwrap_details(text)
    lines = [clean_line(line) for line in text.splitlines()]
    compact: list[str] = []
    blank = False
    for line in lines:
        if not line.strip():
            if not blank:
                compact.append("")
            blank = True
            continue
        compact.append(line)
        blank = False
    return "\n".join(compact).strip()


def is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def current_path(stack: list[tuple[int, str]]) -> list[str]:
    return [title for _, title in stack]


NUMBERED_MARKER_RE = re.compile(
    rf"^\s*(?P<marker>"
    rf"\d+(?:\.\d+)+(?=\s)|"
    rf"\d+[\.、．]|"
    rf"[{CHINESE_NUM}]+[、．\.]|"
    rf"[（(][{CHINESE_NUM}]+[）)]|"
    rf"[（(]\d+[）)]|"
    rf"\d+[）)]"
    rf")\s*(?P<body>\S.*)$"
)


def numbered_marker_level(marker: str, numbered_stack: list[tuple[int, str]]) -> int:
    marker = marker.strip()
    if re.match(r"^\d+(?:\.\d+)+", marker):
        return max(1, marker.rstrip(".、．").count(".") + 1)
    if re.match(r"^\d+[\.、．]", marker):
        return 1
    if re.match(rf"^[{CHINESE_NUM}]+[、．\.]", marker):
        return 1
    if re.match(rf"^[（(][{CHINESE_NUM}]+[）)]", marker):
        return 1
    if re.match(r"^(?:[（(]\d+[）)]|\d+[）)])", marker):
        if numbered_stack and re.search(r"[：:]$", numbered_stack[-1][1].strip()):
            return numbered_stack[-1][0] + 1
        return 1
    return 1


def numbered_heading_match(line: str) -> tuple[str, int] | None:
    match = NUMBERED_MARKER_RE.match(line)
    if not match:
        return None
    marker = match.group("marker")
    body = match.group("body")
    if not body.strip():
        return None
    return line.strip(), 0


def parse_markdown_by_heading(text: str, fix_mojibake: bool) -> tuple[list[dict], list[Block]]:
    text = normalize_text(text, fix_mojibake=fix_mojibake)
    lines = text.splitlines()
    offset = 0
    heading_stack: list[tuple[int, str]] = []
    toc: list[dict] = []
    blocks: list[Block] = []
    buf: list[str] = []
    buf_start = 0
    buf_heading_path: list[str] = []
    buf_level = 0

    def flush(end_pos: int) -> None:
        nonlocal buf, buf_start, buf_heading_path, buf_level
        raw = "\n".join(buf).strip()
        if raw:
            blocks.append(
                Block(
                    kind="section",
                    text=raw,
                    char_start=buf_start,
                    char_end=end_pos,
                    heading_path=buf_heading_path,
                    level=buf_level,
                )
            )
        buf = []
        buf_heading_path = []
        buf_level = 0

    for line in lines:
        line_start = offset
        line_end = offset + len(line)
        offset = line_end + 1

        heading = HEADING_RE.match(line)
        if heading:
            flush(line_start)
            level = len(heading.group(1))
            title = heading.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            path = current_path(heading_stack)
            toc.append({"level": level, "title": title, "heading_path": path})
            buf_start = line_start
            buf_heading_path = path
            buf_level = level
            continue

        if not buf and heading_stack:
            buf_start = line_start
            buf_heading_path = current_path(heading_stack)
            buf_level = heading_stack[-1][0]
        buf.append(line)

    flush(len(text))
    return toc, blocks


def parse_markdown_by_heading_and_numbered(text: str, fix_mojibake: bool) -> tuple[list[dict], list[Block]]:
    text = normalize_text(text, fix_mojibake=fix_mojibake)
    lines = text.splitlines()
    offset = 0
    heading_stack: list[tuple[int, str]] = []
    numbered_stack: list[tuple[int, str]] = []
    toc: list[dict] = []
    blocks: list[Block] = []
    buf: list[str] = []
    buf_start = 0
    buf_heading_path: list[str] = []
    buf_level = 0

    def full_path() -> list[str]:
        return current_path(heading_stack) + current_path(numbered_stack)

    def flush(end_pos: int) -> None:
        nonlocal buf, buf_start, buf_heading_path, buf_level
        raw = "\n".join(buf).strip()
        if raw:
            blocks.append(
                Block(
                    kind="section",
                    text=raw,
                    char_start=buf_start,
                    char_end=end_pos,
                    heading_path=buf_heading_path,
                    level=buf_level,
                )
            )
        buf = []
        buf_heading_path = []
        buf_level = 0

    for line in lines:
        line_start = offset
        line_end = offset + len(line)
        offset = line_end + 1

        heading = HEADING_RE.match(line)
        if heading:
            flush(line_start)
            numbered_stack = []
            level = len(heading.group(1))
            title = heading.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            path = current_path(heading_stack)
            toc.append({"level": level, "title": title, "heading_path": path})
            buf_start = line_start
            buf_heading_path = path
            buf_level = level
            continue

        numbered_match = NUMBERED_MARKER_RE.match(line)
        if numbered_match and heading_stack:
            flush(line_start)
            title = line.strip()
            numbered_level = numbered_marker_level(numbered_match.group("marker"), numbered_stack)
            while numbered_stack and numbered_stack[-1][0] >= numbered_level:
                numbered_stack.pop()
            numbered_stack.append((numbered_level, title))
            path = full_path()
            level = heading_stack[-1][0] + numbered_level
            toc.append({"level": level, "title": title, "heading_path": path})
            buf_start = line_start
            buf_heading_path = path
            buf_level = level
            buf.append(line)
            continue

        if not buf and heading_stack:
            buf_start = line_start
            buf_heading_path = full_path()
            buf_level = heading_stack[-1][0] + (numbered_stack[-1][0] if numbered_stack else 0)
        buf.append(line)

    flush(len(text))
    return toc, blocks


def parse_markdown_by_block(text: str, fix_mojibake: bool) -> tuple[list[dict], list[Block]]:
    text = normalize_text(text, fix_mojibake=fix_mojibake)
    lines = text.splitlines()
    offset = 0
    heading_stack: list[tuple[int, str]] = []
    toc: list[dict] = []
    blocks: list[Block] = []
    buf: list[str] = []
    buf_start = 0
    buf_kind = "text"

    def flush(end_pos: int) -> None:
        nonlocal buf, buf_start, buf_kind
        raw = "\n".join(buf).strip()
        if raw:
            blocks.append(
                Block(
                    kind=buf_kind,
                    text=raw,
                    char_start=buf_start,
                    char_end=end_pos,
                    heading_path=current_path(heading_stack),
                    level=heading_stack[-1][0] if heading_stack else 0,
                )
            )
        buf = []
        buf_kind = "text"

    for line in lines:
        line_start = offset
        line_end = offset + len(line)
        offset = line_end + 1

        heading = HEADING_RE.match(line)
        if heading:
            flush(line_start)
            level = len(heading.group(1))
            title = heading.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            toc.append({"level": level, "title": title, "heading_path": current_path(heading_stack)})
            continue

        kind = "table" if is_table_line(line) else "text"
        if buf and kind != buf_kind:
            flush(line_start)
        if not buf:
            buf_start = line_start
            buf_kind = kind
        buf.append(line)

    flush(len(text))
    return toc, blocks


def split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}".strip()
            continue
        if current:
            parts.append(current)
        if len(para) <= max_chars:
            current = para
            continue
        start = 0
        while start < len(para):
            end = min(len(para), start + max_chars)
            parts.append(para[start:end])
            if end == len(para):
                break
            start = max(0, end - overlap_chars)
        current = ""
    if current:
        parts.append(current)
    return parts


def extract_table_headers(text: str) -> list[str]:
    for line in text.splitlines():
        if is_table_line(line):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            return [cell for cell in cells if cell and not set(cell) <= {"-", ":"}]
    return []


def extract_terms(text: str) -> dict:
    return {
        "clause_ids": sorted(set(CLAUSE_RE.findall(text))),
        "dates": sorted(set(DATE_RE.findall(text))),
        "numbers": sorted(set(NUMBER_RE.findall(text))),
        "table_headers": extract_table_headers(text),
    }


def infer_doc_id(path: Path) -> str:
    name = path.name
    for suffix in ("_no_images.md", ".md"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def infer_domain(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    parts = rel.parts
    if "raw_md" in parts:
        idx = parts.index("raw_md")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    abs_parts = path.parts
    if "raw_md" in abs_parts:
        idx = abs_parts.index("raw_md")
        if idx + 1 < len(abs_parts):
            return abs_parts[idx + 1]
    return parts[0] if len(parts) > 1 else "unknown"


def build_pages(
    path: Path,
    root: Path,
    max_chars: int,
    overlap_chars: int,
    fix_mojibake: bool,
    split_mode: str,
) -> tuple[dict, dict, list[dict]]:
    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    text = normalize_text(raw_text, fix_mojibake=fix_mojibake)
    if split_mode == "heading":
        toc, blocks = parse_markdown_by_heading(raw_text, fix_mojibake=fix_mojibake)
    elif split_mode == "heading_plus_numbered":
        toc, blocks = parse_markdown_by_heading_and_numbered(raw_text, fix_mojibake=fix_mojibake)
    else:
        toc, blocks = parse_markdown_by_block(raw_text, fix_mojibake=fix_mojibake)

    doc_id = infer_doc_id(path)
    domain = infer_domain(path, root)
    title = toc[0]["title"] if toc else path.stem
    pages: list[dict] = []
    page_no = 1

    for block in blocks:
        parts = [block.text] if split_mode == "heading" else split_long_text(block.text, max_chars, overlap_chars)
        for part_idx, part in enumerate(parts):
            terms = extract_terms(part)
            heading_text = " ".join(block.heading_path)
            index_text = "\n".join(
                item
                for item in [
                    title,
                    heading_text,
                    " ".join(terms["clause_ids"]),
                    " ".join(terms["table_headers"]),
                    " ".join(terms["dates"]),
                    " ".join(terms["numbers"]),
                    part,
                ]
                if item
            )
            pages.append(
                {
                    "doc_id": doc_id,
                    "domain": domain,
                    "source_path": str(path),
                    "page_id": f"{doc_id}_lp_{page_no:04d}",
                    "page_no": page_no,
                    "logical_page": True,
                    "kind": block.kind,
                    "title": title,
                    "heading_path": block.heading_path,
                    "level": block.level,
                    "text": part,
                    "index_text": index_text,
                    "char_start": block.char_start,
                    "char_end": block.char_end,
                    "part_index": part_idx,
                    **terms,
                }
            )
            page_no += 1

    doc = {
        "doc_id": doc_id,
        "domain": domain,
        "title": title,
        "source_path": str(path),
        "page_count": len(pages),
        "content_hash": stable_id(text, 16),
    }
    toc_doc = {"doc_id": doc_id, "domain": domain, "title": title, "source_path": str(path), "toc": toc}
    return doc, toc_doc, pages


def iter_markdown_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
    else:
        yield from sorted(input_path.rglob("*.md"))


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def shorten(text: str, limit: int = 220) -> str:
    one_line = re.sub(r"\s+", " ", text).strip()
    return one_line if len(one_line) <= limit else one_line[:limit] + "..."


def write_toc_tree(path: Path, toc_docs: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for doc in toc_docs:
        lines.append(f"# {doc['doc_id']} | {doc['title']}")
        lines.append("")
        for item in doc["toc"]:
            indent = "  " * max(0, item["level"] - 1)
            lines.append(f"{indent}- {item['title']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def heading_startswith(path: list[str], prefix: list[str]) -> bool:
    return len(path) >= len(prefix) and path[: len(prefix)] == prefix


def build_tree_docs(toc_docs: Iterable[dict], pages: Iterable[dict]) -> list[dict]:
    pages_by_doc: dict[str, list[dict]] = {}
    for page in pages:
        pages_by_doc.setdefault(page["doc_id"], []).append(page)

    tree_docs: list[dict] = []
    for doc in toc_docs:
        doc_pages = pages_by_doc.get(doc["doc_id"], [])
        sections: list[dict] = []
        seen_paths: set[tuple[str, ...]] = set()
        for idx, item in enumerate(doc["toc"], start=1):
            heading_path = item.get("heading_path") or []
            path_key = tuple(heading_path)
            if not heading_path or path_key in seen_paths:
                continue
            seen_paths.add(path_key)

            exact_pages = [
                page for page in doc_pages
                if tuple(page.get("heading_path") or []) == path_key
            ]
            descendant_pages = [
                page for page in doc_pages
                if heading_startswith(page.get("heading_path") or [], heading_path)
            ]
            preview_pages = exact_pages or descendant_pages
            preview = shorten(" ".join(page.get("text", "") for page in preview_pages[:2]), 260)
            section_id = f"{doc['doc_id']}_sec_{idx:04d}"
            sections.append(
                {
                    "section_id": section_id,
                    "doc_id": doc["doc_id"],
                    "level": item.get("level", len(heading_path)),
                    "title": item.get("title", heading_path[-1]),
                    "heading_path": heading_path,
                    "page_ids": [page["page_id"] for page in exact_pages],
                    "descendant_page_ids": [page["page_id"] for page in descendant_pages],
                    "preview": preview,
                }
            )
        path_to_section_id = {
            tuple(section.get("heading_path") or []): section["section_id"]
            for section in sections
        }
        children_by_parent: dict[str, list[str]] = {section["section_id"]: [] for section in sections}
        for section in sections:
            heading_path = section.get("heading_path") or []
            parent_id = ""
            for parent_len in range(len(heading_path) - 1, 0, -1):
                parent_id = path_to_section_id.get(tuple(heading_path[:parent_len]), "")
                if parent_id:
                    break
            section["parent_id"] = parent_id
            if parent_id:
                children_by_parent.setdefault(parent_id, []).append(section["section_id"])
        for section in sections:
            section["children_ids"] = children_by_parent.get(section["section_id"], [])

        tree_docs.append(
            {
                "doc_id": doc["doc_id"],
                "domain": doc["domain"],
                "title": doc["title"],
                "source_path": doc["source_path"],
                "sections": sections,
            }
        )
    return tree_docs


def write_doc_files(output_dir: Path, tree_docs: Iterable[dict], pages: Iterable[dict]) -> None:
    pages_by_doc: dict[str, list[dict]] = {}
    for page in pages:
        pages_by_doc.setdefault(page["doc_id"], []).append(page)

    docs_dir = output_dir / "docs"
    for doc in tree_docs:
        doc_id = doc["doc_id"]
        doc_dir = docs_dir / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        tree_lines = [f"# doc {doc_id} | {doc['title']}", ""]
        section_rows: list[dict] = []
        for section in doc.get("sections", []):
            heading = " > ".join(section.get("heading_path") or [])
            tree_lines.append(f"{section['section_id']} | L{section.get('level', '')} | {heading}")
            section_rows.append(
                {
                    "section_id": section["section_id"],
                    "doc_id": doc_id,
                    "level": section.get("level"),
                    "title": section.get("title"),
                    "heading_path": section.get("heading_path") or [],
                    "parent_id": section.get("parent_id", ""),
                    "children_ids": section.get("children_ids") or [],
                    "page_ids": section.get("page_ids") or [],
                    "descendant_page_ids": section.get("descendant_page_ids") or [],
                    "preview": section.get("preview", ""),
                }
            )

        (doc_dir / "tree.md").write_text("\n".join(tree_lines), encoding="utf-8", newline="\n")
        write_jsonl(doc_dir / "sections.jsonl", section_rows)
        write_jsonl(doc_dir / "pages.jsonl", pages_by_doc.get(doc_id, []))


def write_page_preview(path: Path, pages: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    current_doc = None
    for page in pages:
        if page["doc_id"] != current_doc:
            current_doc = page["doc_id"]
            lines.append(f"# {page['doc_id']} | {page['title']}")
            lines.append("")
        heading = " > ".join(page["heading_path"]) if page["heading_path"] else "(no heading)"
        lines.append(f"## {page['page_id']} [{page['kind']}]")
        lines.append(f"- heading: {heading}")
        if page["clause_ids"]:
            lines.append(f"- clauses: {', '.join(page['clause_ids'][:8])}")
        if page["dates"]:
            lines.append(f"- dates: {', '.join(page['dates'][:8])}")
        if page["numbers"]:
            lines.append(f"- numbers: {', '.join(page['numbers'][:12])}")
        if page["table_headers"]:
            lines.append(f"- table_headers: {', '.join(page['table_headers'][:12])}")
        lines.append("")
        lines.append(shorten(page["text"]))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    input_path = Path(DEFAULT_INPUT).resolve()
    output_dir = Path(DEFAULT_OUTPUT).resolve()
    files = list(iter_markdown_files(input_path))
    if DEFAULT_LIMIT:
        files = files[:DEFAULT_LIMIT]

    root = input_path if input_path.is_dir() else input_path.parent
    documents: list[dict] = []
    tocs: list[dict] = []
    pages: list[dict] = []
    for file_path in files:
        doc, toc_doc, page_rows = build_pages(
            file_path,
            root=root,
            max_chars=DEFAULT_MAX_CHARS,
            overlap_chars=DEFAULT_OVERLAP_CHARS,
            fix_mojibake=DEFAULT_FIX_MOJIBAKE,
            split_mode=DEFAULT_SPLIT_MODE,
        )
        documents.append(doc)
        tocs.append(toc_doc)
        pages.extend(page_rows)

    write_jsonl(output_dir / "documents.jsonl", documents)
    write_jsonl(output_dir / "toc.jsonl", tocs)
    write_jsonl(output_dir / "page_index.jsonl", pages)
    tree_docs = build_tree_docs(tocs, pages)
    write_jsonl(output_dir / "tree_doc.jsonl", tree_docs)
    write_doc_files(output_dir, tree_docs, pages)
    write_toc_tree(output_dir / "toc_tree.md", tocs)
    write_page_preview(output_dir / "page_preview.md", pages)
    print(f"processed_files={len(files)}")
    print(f"documents={len(documents)}")
    print(f"pages={len(pages)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
