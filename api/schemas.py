"""Pydantic request and response contracts for the FastAPI backend."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, EmailStr, Field


class UserOut(BaseModel):
    id: str
    organization_id: str
    organization_name: str
    email: str
    full_name: str
    role: str


class AuthRegisterRequest(BaseModel):
    organization_name: str = Field(min_length=2, max_length=255)
    full_name: str = Field(default="", max_length=255)
    email: EmailStr
    password: str = Field(min_length=10, max_length=256)


class AuthLoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class HealthResponse(BaseModel):
    status: Literal["ok"]
    database_enabled: bool
    product: str = "MaintenanceCopilot AI"


class ConversationCreateRequest(BaseModel):
    title: str = Field(default="New maintenance case", max_length=255)


class ConversationMemoryRequest(BaseModel):
    memory_lines: list[str] = Field(default_factory=list, max_length=40)


class ConversationOut(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[dict[str, Any]]
    memory: list[str]


class ChatAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    response_mode: Literal["Auto", "Diagnostic", "Procedural", "Conceptual"] = "Auto"
    use_case_memory: bool = True


class ChatAskResponse(BaseModel):
    answer: str
    evidence: list[dict[str, Any]]
    conversation: ConversationOut
    latency_ms: int


class UploadResponse(BaseModel):
    saved_files: list[str]
    indexing_started: bool
    message: str


class IndexRequest(BaseModel):
    ocr_mode: Literal["auto", "always", "off"] = "auto"
    force_rebuild: bool = False


class IndexResponse(BaseModel):
    summary: dict[str, Any]


class DocumentOut(BaseModel):
    id: str
    file_name: str
    status: str
    size_bytes: int | None = None
    created_at: str
