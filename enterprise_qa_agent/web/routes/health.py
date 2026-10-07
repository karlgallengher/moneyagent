from fastapi import APIRouter

from enterprise_qa_agent.web.response import ok

router = APIRouter()


@router.get("/api/health")
def health():
    return ok({"status": "ready"})
