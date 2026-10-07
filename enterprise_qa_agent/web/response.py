from typing import Any

from fastapi import HTTPException


def ok(data: Any) -> dict[str, Any]:
    return {"ok": True, "data": data, "error": None}


def fail(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, HTTPException):
        raise exc
    return {
        "ok": False,
        "data": None,
        "error": {"type": type(exc).__name__, "message": str(exc)},
    }
