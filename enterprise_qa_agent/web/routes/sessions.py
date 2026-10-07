import uuid

from fastapi import APIRouter, Depends

from enterprise_qa_agent.chat_qa_agent.src.memory_store import (
    connect_db, init_db, list_sessions, load_session_state, reset_session,
    update_session_title,
)
from enterprise_qa_agent.web.authorization import require_session
from enterprise_qa_agent.web.response import fail, ok
from enterprise_qa_agent.web.schemas import (
    SessionCreateRequest, SessionRenameRequest, SessionResetRequest,
)
from scripts.web_auth import csrf_user, current_user

router = APIRouter()


@router.get("/api/sessions")
def sessions(user: dict = Depends(current_user)):
    try:
        with connect_db() as conn:
            init_db(conn)
            return ok({"sessions": list_sessions(conn, user["user_id"])})
    except Exception as exc:
        return fail(exc)


@router.post("/api/sessions")
def create_session(request: SessionCreateRequest, user: dict = Depends(csrf_user)):
    try:
        session_id = f"web-{uuid.uuid4().hex[:12]}"
        title = request.title.strip() or "新会话"
        with connect_db() as conn:
            init_db(conn)
            conn.execute(
                "INSERT INTO sessions(session_id, title, owner_id, created_at, updated_at) "
                "VALUES (?, ?, ?, datetime('now'), datetime('now'))",
                (session_id, title, user["user_id"]),
            )
        return ok({"session_id": session_id, "title": title})
    except Exception as exc:
        return fail(exc)


@router.get("/api/sessions/{session_id}")
def session_history(session_id: str, user: dict = Depends(current_user)):
    try:
        clean_session_id = session_id.strip() or "web-default"
        with connect_db() as conn:
            init_db(conn)
            require_session(conn, clean_session_id, user["user_id"])
            state = load_session_state(conn, clean_session_id)
            session_row = conn.execute(
                "SELECT title FROM sessions WHERE session_id = ?", (clean_session_id,)
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
        return ok({
            "session_id": clean_session_id,
            "title": str(session_row["title"] or clean_session_id) if session_row else clean_session_id,
            "turn_count": state.get("turn_count", 0),
            "active_domain": state.get("active_domain", ""),
            "active_doc_ids": state.get("active_doc_ids", []),
            "turns": turns,
        })
    except Exception as exc:
        return fail(exc)


@router.post("/api/sessions/reset")
def reset_session_api(request: SessionResetRequest, user: dict = Depends(csrf_user)):
    try:
        session_id = request.session_id.strip() or "web-default"
        with connect_db() as conn:
            init_db(conn)
            require_session(conn, session_id, user["user_id"])
            reset_session(conn, session_id)
            conn.execute(
                "INSERT INTO sessions(session_id, title, owner_id, created_at, updated_at) "
                "VALUES (?, ?, ?, datetime('now'), datetime('now'))",
                (session_id, session_id, user["user_id"]),
            )
        return ok({"session_id": session_id, "reset": True})
    except Exception as exc:
        return fail(exc)


@router.patch("/api/sessions/{session_id}")
def rename_session(session_id: str, request: SessionRenameRequest, user: dict = Depends(csrf_user)):
    try:
        clean_session_id = session_id.strip()
        if not clean_session_id:
            raise ValueError("session_id 不能为空")
        with connect_db() as conn:
            init_db(conn)
            require_session(conn, clean_session_id, user["user_id"])
            update_session_title(conn, clean_session_id, request.title)
        return ok({"session_id": clean_session_id, "title": request.title.strip()})
    except Exception as exc:
        return fail(exc)


@router.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, user: dict = Depends(csrf_user)):
    try:
        clean_session_id = session_id.strip()
        if not clean_session_id:
            raise ValueError("session_id 不能为空")
        with connect_db() as conn:
            init_db(conn)
            require_session(conn, clean_session_id, user["user_id"])
            reset_session(conn, clean_session_id)
        return ok({"session_id": clean_session_id, "deleted": True})
    except Exception as exc:
        return fail(exc)
