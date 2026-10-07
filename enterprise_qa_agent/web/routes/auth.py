import os

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from enterprise_qa_agent.chat_qa_agent.src.memory_store import connect_db
from enterprise_qa_agent.web.response import ok
from enterprise_qa_agent.web.schemas import LoginRequest
from scripts.web_auth import (
    COOKIE_NAME,
    SESSION_SECONDS,
    authenticate,
    csrf_user,
    current_user,
    issue_session,
    revoke_session,
)

router = APIRouter()
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    *(
        item.strip().rstrip("/")
        for item in os.environ.get("MONEYAGENT_ALLOWED_ORIGINS", "").split(",")
        if item.strip()
    ),
]


@router.post("/api/auth/login")
def login(credentials: LoginRequest, request: Request, response: Response):
    origin = request.headers.get("origin")
    if origin and origin not in {str(request.base_url).rstrip("/"), *ALLOWED_ORIGINS}:
        raise HTTPException(status_code=403, detail="请求来源无效")
    with connect_db() as conn:
        user = authenticate(conn, credentials.username, credentials.password)
        if user is None:
            raise HTTPException(status_code=401, detail="账号或密码错误，或尝试次数过多")
        token, csrf = issue_session(conn, user["user_id"])
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=SESSION_SECONDS,
        httponly=True,
        samesite="strict",
        secure=request.url.hostname not in {"localhost", "127.0.0.1", "testserver"},
        path="/",
    )
    return ok({"username": user["username"], "csrf_token": csrf})


@router.get("/api/auth/me")
def me(user: dict = Depends(current_user)):
    return ok({"username": user["username"], "csrf_token": user["csrf_token"]})


@router.post("/api/auth/logout")
def logout(request: Request, response: Response, user: dict = Depends(csrf_user)):
    with connect_db() as conn:
        revoke_session(conn, request.cookies.get(COOKIE_NAME, ""))
    response.delete_cookie(COOKIE_NAME, path="/", samesite="strict")
    return ok({"logged_out": True})
