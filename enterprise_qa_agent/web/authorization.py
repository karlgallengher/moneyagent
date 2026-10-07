from fastapi import HTTPException

from enterprise_qa_agent.src.chat.domain_registry import load_domains
from scripts.web_auth import require_session as _require_session


def require_owned_domain(domain: str, user_id: str) -> None:
    if load_domains().get(domain, {}).get("owner_id") != user_id:
        raise HTTPException(status_code=404, detail="领域不存在")


def require_session(conn, session_id: str, user_id: str) -> None:
    _require_session(conn, session_id, user_id)
