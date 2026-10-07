from __future__ import annotations

import json
import os
import re
import shutil
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("MONEYAGENT_DATA_ROOT", str(REPO_ROOT)))
REGISTRY_PATH = DATA_ROOT / "processed" / "user_domains.json"
UPLOAD_ROOT = DATA_ROOT / "public_dataset_upload" / "raw_md"
INDEX_ROOT = DATA_ROOT / "processed" / "user_page_indexes"
INDEX_LOCK = threading.RLock()
_VISIBLE_DOMAINS: ContextVar[frozenset[str] | None] = ContextVar("visible_domains", default=None)

BUILTIN_DOMAINS = {
    "financial_contracts": "processed/page_index_financial_contracts",
    "financial_reports": "processed/page_index_financial_reports",
    "insurance": "processed/page_index_insurance",
    "regulatory": "processed/page_index_regulatory",
    "research": "processed/page_index_research",
}
DOMAIN_RE = re.compile(r"^[a-z][a-z0-9_-]{2,39}$")
SPLIT_MODES = {"heading", "heading_plus_numbered", "block"}


def load_domains() -> dict[str, dict]:
    if not REGISTRY_PATH.exists():
        return {}
    data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Invalid domain registry")
    return data


def save_domains(domains: dict[str, dict]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = REGISTRY_PATH.with_name(f".user_domains_{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(domains, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, REGISTRY_PATH)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def visible_domains(domains: set[str]):
    token = _VISIBLE_DOMAINS.set(frozenset(domains))
    try:
        yield
    finally:
        _VISIBLE_DOMAINS.reset(token)


def owned_domain_keys(user_id: str) -> set[str]:
    return {
        domain for domain, item in load_domains().items()
        if item.get("owner_id") == user_id
    }


def claim_legacy_domains(user_id: str) -> None:
    with INDEX_LOCK:
        domains = load_domains()
        for item in domains.values():
            if not item.get("owner_id"):
                item["owner_id"] = user_id
        save_domains(domains)


def create_domain(domain: str, name: str, split_mode: str = "heading", *, owner_id: str | None = None) -> dict:
    if not DOMAIN_RE.fullmatch(domain) or domain in BUILTIN_DOMAINS:
        raise ValueError("领域标识需为 3-40 位小写字母、数字、下划线或连字符，且不能与内置领域重名")
    name = name.strip()
    if not name or len(name) > 60:
        raise ValueError("领域名称长度须为 1-60 字符")
    if split_mode not in SPLIT_MODES:
        raise ValueError("无效的切分方式")
    with INDEX_LOCK:
        domains = load_domains()
        if domain in domains:
            raise ValueError("领域标识已存在")
        domains[domain] = {
            "name": name,
            "split_mode": split_mode,
            "index_dir": "",
            "status": "needs_build",
            "owner_id": owner_id,
        }
        save_domains(domains)
        return domains[domain]


def user_domain(domain: str) -> dict:
    item = load_domains().get(domain)
    if item is None:
        raise ValueError("用户领域不存在")
    return item


def domain_index_dirs() -> dict[str, str]:
    visible = _VISIBLE_DOMAINS.get()
    result = dict(BUILTIN_DOMAINS)
    for domain, item in load_domains().items():
        if visible is not None and domain not in visible:
            continue
        index_dir = str(item.get("index_dir") or "")
        if DOMAIN_RE.fullmatch(domain) and index_dir:
            path = (REPO_ROOT / index_dir).resolve()
            if not Path(index_dir).is_absolute() and not path.exists():
                path = (DATA_ROOT / index_dir).resolve()
            if path.is_relative_to(INDEX_ROOT.resolve()) and (path / "page_index.jsonl").exists():
                result[domain] = str(path) if not path.is_relative_to(REPO_ROOT) else index_dir
    return result


def upload_dir(domain: str) -> Path:
    user_domain(domain)
    return UPLOAD_ROOT / domain


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def mark_domain_dirty(domain: str) -> None:
    with INDEX_LOCK:
        domains = load_domains()
        item = domains.get(domain)
        if item is None:
            raise ValueError("用户领域不存在")
        item["status"] = "needs_build"
        item["updated_at"] = _now_iso()
        save_domains(domains)


def list_domain_files(domain: str) -> list[dict]:
    folder = upload_dir(domain)
    if not folder.exists():
        return []
    from scripts.build_page_index import infer_doc_id

    files = []
    for path in sorted(folder.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        stat = path.stat()
        files.append(
            {
                "filename": path.name,
                "doc_id": infer_doc_id(path),
                "size_bytes": stat.st_size,
                "updated_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            }
        )
    return files


def delete_domain_file(domain: str, filename: str) -> None:
    with INDEX_LOCK:
        filename = validate_filename(filename)
        path = upload_dir(domain) / filename
        if not path.exists() or not path.is_file():
            raise FileNotFoundError("文件不存在")
        path.unlink()
        mark_domain_dirty(domain)


def delete_user_domain(domain: str) -> None:
    if domain in BUILTIN_DOMAINS:
        raise ValueError("内置领域不能删除")
    with INDEX_LOCK:
        domains = load_domains()
        if domain not in domains:
            raise ValueError("用户领域不存在")
        source = upload_dir(domain).resolve()
        index = (INDEX_ROOT / domain).resolve()
        upload_root = UPLOAD_ROOT.resolve()
        index_root = INDEX_ROOT.resolve()
        if (
            not DOMAIN_RE.fullmatch(domain)
            or source != upload_root / domain
            or index != index_root / domain
            or source.parent != upload_root
            or index.parent != index_root
        ):
            raise ValueError("领域路径不安全")
        if source.exists():
            shutil.rmtree(source)
        if index.exists():
            shutil.rmtree(index)
        del domains[domain]
        save_domains(domains)
        from enterprise_qa_agent.src.mcp_tools.documents import _catalog, _pages, _page_bm25

        _catalog.cache_clear()
        _pages.cache_clear()
        _page_bm25.cache_clear()


def replace_domain_file(domain: str, filename: str, content: bytes) -> None:
    with INDEX_LOCK:
        filename = validate_filename(filename)
        folder = upload_dir(domain)
        path = folder / filename
        if not path.is_file():
            raise FileNotFoundError("文件不存在，请先上传")
        temporary = folder / f".{filename}.{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_bytes(content)
            os.replace(temporary, path)
            mark_domain_dirty(domain)
        finally:
            temporary.unlink(missing_ok=True)


def validate_filename(filename: str) -> str:
    if (
        not filename
        or filename != Path(filename).name
        or "\\" in filename
        or "/" in filename
        or filename.startswith(".")
        or filename != filename.strip()
        or filename.endswith(".")
        or len(filename) > 120
        or any(ord(char) < 32 or char in '<>:"|?*' for char in filename)
    ):
        raise ValueError("文件名不合法")
    stem = filename.rsplit(".", 1)[0]
    if filename.lower().endswith((".md", ".txt")) and stem and stem.rstrip(" .").upper() not in {
        "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))
    } and not stem.endswith((" ", ".")):
        return filename
    raise ValueError("只支持 .md 或 .txt 文件")


def build_domain(domain: str) -> dict:
    from scripts.build_page_index import build_index, infer_doc_id

    with INDEX_LOCK:
        item = user_domain(domain)
        source = upload_dir(domain)
        files = sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in {".md", ".txt"}) if source.exists() else []
        if not files:
            raise ValueError("请先上传 .md 或 .txt 文件")
        ids = [infer_doc_id(path) for path in files]
        if len(ids) != len(set(ids)):
            raise ValueError("文件名对应的文档 ID 重复，请重命名后再构建")
        output = INDEX_ROOT / domain / f"build_{uuid.uuid4().hex}"
        result = build_index(source, output, str(item["split_mode"]))
        if not result["page_count"]:
            raise ValueError("文件未产生可检索内容，请检查文件格式和内容")
        domains = load_domains()
        domains[domain]["index_dir"] = (
            output.relative_to(REPO_ROOT).as_posix()
            if output.is_relative_to(REPO_ROOT) else str(output.resolve())
        )
        domains[domain]["status"] = "ready"
        domains[domain]["last_built_at"] = _now_iso()
        save_domains(domains)
        from enterprise_qa_agent.src.mcp_tools.documents import _catalog, _pages, _page_bm25

        _catalog.cache_clear()
        _pages.cache_clear()
        _page_bm25.cache_clear()
        return result
