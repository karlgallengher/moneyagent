from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
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


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=1, max_length=1024)
