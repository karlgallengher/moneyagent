from __future__ import annotations

import contextlib
import io
import sys
import uuid
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from enterprise_qa_agent.chat_qa_agent.src.memory_store import (
    connect_db,
    ensure_session,
    init_db,
    list_sessions,
    load_session_state,
    reset_session,
    update_session_title,
)
from enterprise_qa_agent.chat_qa_agent.src.session_runner import run_one_turn
from enterprise_qa_agent.src.chat.api import state_to_response
from enterprise_qa_agent.src.chat.domain_registry import (
    INDEX_LOCK, build_domain, create_domain, delete_domain_file, delete_user_domain,
    list_domain_files, mark_domain_dirty, replace_domain_file, upload_dir, validate_filename,
)
from enterprise_qa_agent.src.mcp_tools import list_domains


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    domain: str = ""
    session_id: str = "default"
    preferred_doc_ids: list[str] = Field(default_factory=list)
    include_raw_state: bool = False


class SessionResetRequest(BaseModel):
    session_id: str = "web-default"


class SessionCreateRequest(BaseModel):
    title: str = ""


class SessionRenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=80)


class DomainCreateRequest(BaseModel):
    domain: str
    name: str
    split_mode: str = "heading"


app = FastAPI(title="MoneyAgent Web API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def ok(data: Any) -> dict[str, Any]:
    return {"ok": True, "data": data, "error": None}


def fail(exc: Exception) -> dict[str, Any]:
    return {
        "ok": False,
        "data": None,
        "error": {"type": type(exc).__name__, "message": str(exc)},
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    return ok({"status": "ready"})


@app.get("/api/domains")
def domains() -> dict[str, Any]:
    try:
        return ok(list_domains())
    except Exception as exc:
        return fail(exc)


@app.post("/api/domains")
def add_domain(request: DomainCreateRequest) -> dict[str, Any]:
    try:
        result = create_domain(request.domain.strip(), request.name, request.split_mode)
        return ok({"domain": request.domain.strip(), **result})
    except Exception as exc:
        return fail(exc)


@app.get("/api/domains/{domain}/files")
def domain_files(domain: str) -> dict[str, Any]:
    try:
        return ok({"domain": domain, "files": list_domain_files(domain)})
    except Exception as exc:
        return fail(exc)


@app.post("/api/domains/{domain}/files")
async def add_domain_file(domain: str, file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        filename = validate_filename(file.filename or "")
        content = await file.read(8 * 1024 * 1024 + 1)
        if not content or len(content) > 8 * 1024 * 1024:
            raise ValueError("文件不能为空，且不能超过 8 MB")
        content.decode("utf-8-sig")
        with INDEX_LOCK:
            folder = upload_dir(domain)
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / filename
            if path.exists():
                raise ValueError("该文件名已存在，请先重命名文件")
            from scripts.build_page_index import infer_doc_id

            existing_ids = {
                infer_doc_id(other) for other in folder.iterdir()
                if other.is_file() and other.suffix.lower() in {".md", ".txt"}
            }
            if infer_doc_id(path) in existing_ids:
                raise ValueError("文档 ID 与已有文件重复，请重命名文件")
            with path.open("xb") as output:
                output.write(content)
            mark_domain_dirty(domain)
        return ok({"domain": domain, "filename": filename})
    except Exception as exc:
        return fail(exc)
    finally:
        await file.close()


@app.put("/api/domains/{domain}/files/{filename}")
async def replace_domain_upload(domain: str, filename: str, file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        validate_filename(filename)
        content = await file.read(8 * 1024 * 1024 + 1)
        if not content or len(content) > 8 * 1024 * 1024:
            raise ValueError("文件不能为空，且不能超过 8 MB")
        content.decode("utf-8-sig")
        replace_domain_file(domain, filename, content)
        return ok({"domain": domain, "filename": filename, "replaced": True})
    except Exception as exc:
        return fail(exc)
    finally:
        await file.close()


@app.delete("/api/domains/{domain}/files/{filename}")
def remove_domain_file(domain: str, filename: str) -> dict[str, Any]:
    try:
        delete_domain_file(domain, filename)
        return ok({"domain": domain, "filename": filename, "deleted": True})
    except Exception as exc:
        return fail(exc)


@app.post("/api/domains/{domain}/build")
def build_domain_index(domain: str) -> dict[str, Any]:
    try:
        return ok({"domain": domain, **build_domain(domain)})
    except Exception as exc:
        return fail(exc)


@app.delete("/api/domains/{domain}")
def remove_domain(domain: str) -> dict[str, Any]:
    try:
        delete_user_domain(domain)
        return ok({"domain": domain, "deleted": True})
    except Exception as exc:
        return fail(exc)


@app.get("/api/sessions")
def sessions() -> dict[str, Any]:
    try:
        with connect_db() as conn:
            init_db(conn)
            return ok({"sessions": list_sessions(conn)})
    except Exception as exc:
        return fail(exc)


@app.post("/api/sessions")
def create_session(request: SessionCreateRequest) -> dict[str, Any]:
    try:
        session_id = f"web-{uuid.uuid4().hex[:12]}"
        with connect_db() as conn:
            init_db(conn)
            ensure_session(conn, session_id)
            if request.title.strip():
                update_session_title(conn, session_id, request.title)
            return ok({"session_id": session_id, "title": request.title.strip() or "新会话"})
    except Exception as exc:
        return fail(exc)


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    try:
        session_id = request.session_id.strip() or "web-default"
        with connect_db() as conn:
            init_db(conn)
            ensure_session(conn, session_id)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                final_state = run_one_turn(conn, session_id, request.domain, request.query.strip())
            session_row = conn.execute(
                """
                SELECT s.title, COUNT(t.id) AS turn_count
                FROM sessions s
                LEFT JOIN turns t ON t.session_id = s.session_id
                WHERE s.session_id = ?
                GROUP BY s.session_id
                """,
                (session_id,),
            ).fetchone()
            if (
                session_row
                and int(session_row["turn_count"] or 0) == 1
                and str(session_row["title"] or "") == session_id
            ):
                update_session_title(conn, session_id, request.query.strip())
        response = state_to_response(
            final_state,
            query=request.query,
            output_dir="enterprise_qa_agent/outputs/web_chat_debug",
            include_raw_state=request.include_raw_state,
        ).to_dict(include_raw_state=request.include_raw_state)
        response["session_id"] = session_id
        response["session_trace"] = output.getvalue()
        return ok(response)
    except Exception as exc:
        return fail(exc)


@app.get("/api/sessions/{session_id}")
def session_history(session_id: str) -> dict[str, Any]:
    try:
        clean_session_id = session_id.strip() or "web-default"
        with connect_db() as conn:
            init_db(conn)
            state = load_session_state(conn, clean_session_id)
            session_row = conn.execute(
                "SELECT title FROM sessions WHERE session_id = ?",
                (clean_session_id,),
            ).fetchone()
        turns = [
            {
                "turn_index": row.get("turn_index"),
                "user_query": row.get("user_query"),
                "assistant_answer": row.get("assistant_answer"),
                "domain": row.get("domain"),
                "doc_ids": row.get("doc_ids"),
                "created_at": row.get("created_at"),
            }
            for row in state.get("turn_window") or []
        ]
        return ok(
            {
                "session_id": clean_session_id,
                "title": str(session_row["title"] or clean_session_id) if session_row else clean_session_id,
                "turn_count": state.get("turn_count", 0),
                "active_domain": state.get("active_domain", ""),
                "active_doc_ids": state.get("active_doc_ids", []),
                "turns": turns,
            }
        )
    except Exception as exc:
        return fail(exc)


@app.post("/api/sessions/reset")
def reset_session_api(request: SessionResetRequest) -> dict[str, Any]:
    try:
        session_id = request.session_id.strip() or "web-default"
        with connect_db() as conn:
            init_db(conn)
            reset_session(conn, session_id)
            ensure_session(conn, session_id)
        return ok({"session_id": session_id, "reset": True})
    except Exception as exc:
        return fail(exc)


@app.patch("/api/sessions/{session_id}")
def rename_session(session_id: str, request: SessionRenameRequest) -> dict[str, Any]:
    try:
        clean_session_id = session_id.strip()
        if not clean_session_id:
            raise ValueError("session_id 不能为空")
        with connect_db() as conn:
            init_db(conn)
            update_session_title(conn, clean_session_id, request.title)
        return ok({"session_id": clean_session_id, "title": request.title.strip()})
    except Exception as exc:
        return fail(exc)


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, Any]:
    try:
        clean_session_id = session_id.strip()
        if not clean_session_id:
            raise ValueError("session_id 不能为空")
        with connect_db() as conn:
            init_db(conn)
            reset_session(conn, clean_session_id)
        return ok({"session_id": clean_session_id, "deleted": True})
    except Exception as exc:
        return fail(exc)


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install web API dependencies with: pip install fastapi uvicorn") from exc
    uvicorn.run("scripts.web_api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
