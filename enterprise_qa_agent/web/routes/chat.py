import contextlib
import io

from fastapi import APIRouter, Depends

from enterprise_qa_agent.chat_qa_agent.src.memory_store import connect_db, init_db, update_session_title
from enterprise_qa_agent.src.chat.domain_registry import (
    BUILTIN_DOMAINS, INDEX_LOCK, owned_domain_keys, visible_domains,
)
from enterprise_qa_agent.web.authorization import require_owned_domain, require_session
from enterprise_qa_agent.web.response import fail, ok
from enterprise_qa_agent.web.schemas import ChatRequest
from scripts.web_auth import consume_chat_quota, csrf_user

router = APIRouter()


@router.post("/api/chat")
def chat(request: ChatRequest, user: dict = Depends(csrf_user)):
    try:
        session_id = request.session_id.strip()
        if not session_id:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="请先创建会话")
        with connect_db() as conn:
            init_db(conn)
            require_session(conn, session_id, user["user_id"])
            if request.domain and request.domain not in BUILTIN_DOMAINS:
                require_owned_domain(request.domain, user["user_id"])
            consume_chat_quota(conn, user["user_id"])
            output = io.StringIO()
            with INDEX_LOCK, visible_domains(owned_domain_keys(user["user_id"])), contextlib.redirect_stdout(output):
                from scripts import web_api
                final_state = web_api.run_one_turn(conn, session_id, request.domain, request.query.strip())
            session_row = conn.execute(
                "SELECT title, COUNT(t.id) AS turn_count FROM sessions s "
                "LEFT JOIN turns t ON t.session_id = s.session_id "
                "WHERE s.session_id = ? GROUP BY s.session_id", (session_id,)
            ).fetchone()
            if session_row and int(session_row["turn_count"] or 0) == 1 and str(session_row["title"] or "") == session_id:
                update_session_title(conn, session_id, request.query.strip())
        from scripts import web_api
        response = web_api.state_to_response(
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
