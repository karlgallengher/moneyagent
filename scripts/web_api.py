from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from enterprise_qa_agent.chat_qa_agent.src.session_runner import run_one_turn
from enterprise_qa_agent.src.chat.api import state_to_response
from enterprise_qa_agent.web.routes.auth import router as auth_router
from enterprise_qa_agent.web.routes.chat import router as chat_router
from enterprise_qa_agent.web.routes.domains import router as domains_router
from enterprise_qa_agent.web.routes.health import router as health_router
from enterprise_qa_agent.web.routes.sessions import router as sessions_router
from scripts.web_auth import csrf_user, current_user


ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    *(
        item.strip().rstrip("/")
        for item in os.environ.get("MONEYAGENT_ALLOWED_ORIGINS", "").split(",")
        if item.strip()
    ),
]

app = FastAPI(title="MoneyAgent Web API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(health_router)
app.include_router(domains_router)
app.include_router(sessions_router)
app.include_router(chat_router)


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install web API dependencies with: pip install fastapi uvicorn") from exc
    uvicorn.run("scripts.web_api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
