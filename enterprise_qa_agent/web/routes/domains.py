import os

from fastapi import APIRouter, Depends, File, UploadFile

from enterprise_qa_agent.src.chat.domain_registry import (
    BUILTIN_DOMAINS, INDEX_LOCK, build_domain, create_domain, delete_domain_file,
    delete_user_domain, list_domain_files, load_domains, mark_domain_dirty,
    owned_domain_keys, replace_domain_file, upload_dir, validate_filename,
    visible_domains,
)
from enterprise_qa_agent.src.mcp_tools import list_domains
from enterprise_qa_agent.web.authorization import require_owned_domain
from enterprise_qa_agent.web.response import fail, ok
from enterprise_qa_agent.web.schemas import DomainCreateRequest
from enterprise_qa_agent.chat_qa_agent.src.memory_store import connect_db
from scripts.web_auth import consume_build_quota, csrf_user, current_user

router = APIRouter()


@router.get("/api/domains")
def domains(user: dict = Depends(current_user)):
    try:
        owned = owned_domain_keys(user["user_id"])
        with INDEX_LOCK, visible_domains(owned):
            result = list_domains()
        result["domains"] = [
            item for item in result["domains"]
            if item["domain"] in BUILTIN_DOMAINS or item["domain"] in owned
        ]
        return ok(result)
    except Exception as exc:
        return fail(exc)


@router.post("/api/domains")
def add_domain(request: DomainCreateRequest, user: dict = Depends(csrf_user)):
    try:
        slug = request.domain.strip()
        if len(slug) > 30:
            raise ValueError("领域标识不能超过 30 个字符")
        if len(owned_domain_keys(user["user_id"])) >= max(1, int(os.environ.get("MONEYAGENT_DOMAINS_PER_USER", "8"))):
            from fastapi import HTTPException
            raise HTTPException(status_code=429, detail="自建领域数量已达上限")
        domain = f"u{user['user_id'][:8]}_{slug}"
        result = create_domain(domain, request.name, request.split_mode, owner_id=user["user_id"])
        return ok({"domain": domain, **result})
    except Exception as exc:
        return fail(exc)


@router.get("/api/domains/{domain}/files")
def domain_files(domain: str, user: dict = Depends(current_user)):
    try:
        require_owned_domain(domain, user["user_id"])
        return ok({"domain": domain, "files": list_domain_files(domain)})
    except Exception as exc:
        return fail(exc)


async def _read_upload(file: UploadFile) -> bytes:
    content = await file.read(8 * 1024 * 1024 + 1)
    if not content or len(content) > 8 * 1024 * 1024:
        raise ValueError("文件不能为空，且不能超过 8 MB")
    content.decode("utf-8-sig")
    return content


@router.post("/api/domains/{domain}/files")
async def add_domain_file(domain: str, file: UploadFile = File(...), user: dict = Depends(csrf_user)):
    try:
        require_owned_domain(domain, user["user_id"])
        filename = validate_filename(file.filename or "")
        content = await _read_upload(file)
        with INDEX_LOCK:
            folder = upload_dir(domain)
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / filename
            if path.exists():
                raise ValueError("该文件名已存在，请先重命名文件")
            existing = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".md", ".txt"}]
            if len(existing) >= max(1, int(os.environ.get("MONEYAGENT_FILES_PER_DOMAIN", "50"))):
                from fastapi import HTTPException
                raise HTTPException(status_code=429, detail="该领域的文件数量已达上限")
            if sum(p.stat().st_size for p in existing) + len(content) > max(
                1, int(os.environ.get("MONEYAGENT_UPLOAD_BYTES_PER_DOMAIN", str(64 * 1024 * 1024)))
            ):
                from fastapi import HTTPException
                raise HTTPException(status_code=413, detail="该领域上传内容超过容量上限")
            from scripts.build_page_index import infer_doc_id
            if infer_doc_id(path) in {infer_doc_id(p) for p in existing}:
                raise ValueError("文档 ID 与已有文件重复，请重命名文件")
            with path.open("xb") as output:
                output.write(content)
            mark_domain_dirty(domain)
        return ok({"domain": domain, "filename": filename})
    except Exception as exc:
        return fail(exc)
    finally:
        await file.close()


@router.put("/api/domains/{domain}/files/{filename}")
async def replace_domain_upload(domain: str, filename: str, file: UploadFile = File(...), user: dict = Depends(csrf_user)):
    try:
        require_owned_domain(domain, user["user_id"])
        validate_filename(filename)
        content = await _read_upload(file)
        with INDEX_LOCK:
            folder = upload_dir(domain)
            path = folder / filename
            if not path.is_file():
                raise FileNotFoundError("文件不存在，请先上传")
            used = sum(p.stat().st_size for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() in {".md", ".txt"} and p.name != filename)
            if used + len(content) > max(1, int(os.environ.get(
                "MONEYAGENT_UPLOAD_BYTES_PER_DOMAIN", str(64 * 1024 * 1024)
            ))):
                from fastapi import HTTPException
                raise HTTPException(status_code=413, detail="该领域上传内容超过容量上限")
            replace_domain_file(domain, filename, content)
        return ok({"domain": domain, "filename": filename, "replaced": True})
    except Exception as exc:
        return fail(exc)
    finally:
        await file.close()


@router.delete("/api/domains/{domain}/files/{filename}")
def remove_domain_file(domain: str, filename: str, user: dict = Depends(csrf_user)):
    try:
        require_owned_domain(domain, user["user_id"])
        delete_domain_file(domain, filename)
        return ok({"domain": domain, "filename": filename, "deleted": True})
    except Exception as exc:
        return fail(exc)


@router.post("/api/domains/{domain}/build")
def build_domain_index(domain: str, user: dict = Depends(csrf_user)):
    try:
        require_owned_domain(domain, user["user_id"])
        with connect_db() as conn:
            consume_build_quota(conn, user["user_id"])
        return ok({"domain": domain, **build_domain(domain)})
    except Exception as exc:
        return fail(exc)


@router.delete("/api/domains/{domain}")
def remove_domain(domain: str, user: dict = Depends(csrf_user)):
    try:
        require_owned_domain(domain, user["user_id"])
        delete_user_domain(domain)
        return ok({"domain": domain, "deleted": True})
    except Exception as exc:
        return fail(exc)
