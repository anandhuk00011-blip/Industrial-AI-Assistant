"""FastAPI backend for MaintenanceCopilot AI."""

from __future__ import annotations

import re
import time
import uuid
from datetime import datetime
import os
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import SQLAlchemyError

from api.schemas import (
    AuthLoginRequest,
    AuthRegisterRequest,
    AuthResponse,
    ChatAskRequest,
    ChatAskResponse,
    ConversationCreateRequest,
    ConversationMemoryRequest,
    ConversationOut,
    DocumentOut,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    UploadResponse,
    UserOut,
)
from api.security import create_access_token, current_user
from config import initialize_storage
from core.tenant import resolve_tenant
from database.database import create_database, database_enabled, session_scope
from main import index_pdfs
from query import ask_copilot, get_database_summary, reload_vector_database
from repositories.document_repository import DocumentRepository
from services.audit_chat_service import (
    add_message_record,
    clear_conversation_messages,
    create_conversation_record,
    delete_conversation_record,
    list_conversations,
    list_document_records,
    log_audit,
    record_uploaded_file,
    update_conversation_memory,
    update_conversation_title,
)
from services.auth_service import AuthUser, authenticate, create_account

SUPPORTED_UPLOAD_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}
MAX_UPLOAD_FILES = int(os.getenv("MAX_UPLOAD_FILES", "10"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "200")) * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024
CORS_ALLOW_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8501,http://127.0.0.1:8501",
    ).split(",")
    if origin.strip()
]

app = FastAPI(
    title="MaintenanceCopilot AI API",
    description="Industrial knowledge management and maintenance assistance backend.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    initialize_storage()
    create_database()


def user_out(user: AuthUser) -> UserOut:
    return UserOut(
        id=user.id,
        organization_id=user.organization_id,
        organization_name=user.organization_name,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
    )


def mode_code(label: str) -> str | None:
    return {
        "Diagnostic": "MODE A",
        "Procedural": "MODE B",
        "Conceptual": "MODE C",
    }.get(label)


def safe_filename(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(" .")
    return name or f"document-{uuid.uuid4().hex[:8]}"


def tenant_uploads_dir(user: AuthUser) -> Path:
    tenant = resolve_tenant(user.organization_id)
    return tenant.uploads_dir


def tenant_context(user: AuthUser):
    return resolve_tenant(user.organization_id)


def ensure_tenant_file_path(user: AuthUser, file_path: str) -> Path:
    uploads_dir = tenant_uploads_dir(user).resolve()
    path = Path(file_path).resolve()
    try:
        path.relative_to(uploads_dir)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Document path is outside this tenant workspace.",
        ) from exc
    return path


def validate_uploaded_file(path: Path, suffix: str) -> None:
    if suffix == ".pdf":
        with path.open("rb") as handle:
            if b"%PDF-" not in handle.read(1024):
                path.unlink(missing_ok=True)
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid PDF file.")
    if suffix == ".docx":
        with path.open("rb") as handle:
            if handle.read(2) != b"PK":
                path.unlink(missing_ok=True)
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid DOCX file.")


async def save_upload_stream(upload: UploadFile, target: Path, suffix: str) -> None:
    bytes_written = 0
    try:
        with target.open("wb") as handle:
            while True:
                chunk = await upload.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
                    )
                handle.write(chunk)
        validate_uploaded_file(target, suffix)
    except Exception:
        target.unlink(missing_ok=True)
        raise


def load_conversation_for_user(user: AuthUser, conversation_id: str) -> dict[str, Any]:
    for conversation in list_conversations(user.organization_id, user.id):
        if conversation["id"] == conversation_id:
            return conversation
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")


def refresh_conversation(user: AuthUser, conversation_id: str) -> ConversationOut:
    return ConversationOut(**load_conversation_for_user(user, conversation_id))


def run_indexing_job(
    *,
    organization_id: str,
    input_folder: Path,
    ocr_mode: str,
    force_rebuild: bool,
) -> None:
    index_pdfs(
        input_folder=input_folder,
        ocr_mode=ocr_mode,
        force_rebuild=force_rebuild,
        organization_id=organization_id,
    )
    reload_vector_database(force=True, organization_id=organization_id)


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", database_enabled=database_enabled())


@app.post("/api/auth/register", response_model=AuthResponse, tags=["auth"])
def register(payload: AuthRegisterRequest) -> AuthResponse:
    try:
        user = create_account(
            organization_name=payload.organization_name,
            email=str(payload.email),
            password=payload.password,
            full_name=payload.full_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return AuthResponse(access_token=create_access_token(user), user=user_out(user))


@app.post("/api/auth/login", response_model=AuthResponse, tags=["auth"])
def login(payload: AuthLoginRequest) -> AuthResponse:
    user = authenticate(str(payload.email), payload.password)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password.")
    return AuthResponse(access_token=create_access_token(user), user=user_out(user))


@app.get("/api/me", response_model=UserOut, tags=["auth"])
def me(user: AuthUser = Depends(current_user)) -> UserOut:
    return user_out(user)


@app.get("/api/knowledge/summary", response_model=dict[str, Any], tags=["knowledge"])
def knowledge_summary(user: AuthUser = Depends(current_user)) -> dict[str, Any]:
    return get_database_summary(user.organization_id)


@app.post("/api/knowledge/index", response_model=IndexResponse, tags=["knowledge"])
def index_knowledge(payload: IndexRequest, user: AuthUser = Depends(current_user)) -> IndexResponse:
    summary = index_pdfs(
        input_folder=tenant_uploads_dir(user),
        ocr_mode=payload.ocr_mode,
        force_rebuild=payload.force_rebuild,
        organization_id=user.organization_id,
    )
    reload_vector_database(force=True, organization_id=user.organization_id)
    return IndexResponse(summary=summary)


@app.post("/api/documents/upload", response_model=UploadResponse, tags=["documents"])
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    ocr_mode: str = "auto",
    force_rebuild: bool = False,
    index_after_upload: bool = True,
    user: AuthUser = Depends(current_user),
) -> UploadResponse:
    if ocr_mode not in {"auto", "always", "off"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid OCR mode.")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Upload at most {MAX_UPLOAD_FILES} files at a time.",
        )

    uploads_dir = tenant_uploads_dir(user)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[str] = []

    for upload in files:
        filename = safe_filename(upload.filename or "document")
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {suffix or 'unknown'}",
            )

        target = uploads_dir / filename
        if target.exists():
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            target = uploads_dir / f"{target.stem}__rev-{stamp}{target.suffix}"

        await save_upload_stream(upload, target, suffix)
        try:
            record_uploaded_file(
                organization_id=user.organization_id,
                user_id=user.id,
                file_path=target,
                original_name=upload.filename or filename,
            )
        except SQLAlchemyError as exc:
            target.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database is busy while recording the upload. Close duplicate local servers and try again.",
            ) from exc
        saved_files.append(target.name)

    if index_after_upload:
        background_tasks.add_task(
            run_indexing_job,
            organization_id=user.organization_id,
            input_folder=uploads_dir,
            ocr_mode=ocr_mode,
            force_rebuild=force_rebuild,
        )

    return UploadResponse(
        saved_files=saved_files,
        indexing_started=index_after_upload,
        message="Files uploaded. Indexing has started." if index_after_upload else "Files uploaded.",
    )


@app.get("/api/documents", response_model=list[DocumentOut], tags=["documents"])
def documents(user: AuthUser = Depends(current_user)) -> list[DocumentOut]:
    return [DocumentOut(**record) for record in list_document_records(user.organization_id)]


@app.delete("/api/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["documents"])
def delete_document(document_id: str, user: AuthUser = Depends(current_user)) -> None:
    tenant = tenant_context(user)
    repository = DocumentRepository(tenant)
    with session_scope() as session:
        document = repository.get_by_id(session, document_id)
        if document is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
        file_path = ensure_tenant_file_path(user, document.file_path)
        file_name = document.file_name
        repository.delete_by_id(session, document_id)

    if file_path.exists():
        file_path.unlink()

    log_audit(
        organization_id=user.organization_id,
        user_id=user.id,
        action="document.deleted",
        details={"document_id": document_id, "file_name": file_name},
    )
    index_pdfs(
        input_folder=tenant.uploads_dir,
        ocr_mode="auto",
        force_rebuild=True,
        organization_id=user.organization_id,
    )
    reload_vector_database(force=True, organization_id=user.organization_id)


@app.get("/api/conversations", response_model=list[ConversationOut], tags=["chat"])
def conversations(user: AuthUser = Depends(current_user)) -> list[ConversationOut]:
    return [ConversationOut(**item) for item in list_conversations(user.organization_id, user.id)]


@app.post("/api/conversations", response_model=ConversationOut, tags=["chat"])
def create_conversation(
    payload: ConversationCreateRequest,
    user: AuthUser = Depends(current_user),
) -> ConversationOut:
    conversation = create_conversation_record(
        organization_id=user.organization_id,
        user_id=user.id,
        title=payload.title,
    )
    return ConversationOut(**conversation)


@app.patch("/api/conversations/{conversation_id}/memory", response_model=ConversationOut, tags=["chat"])
def update_memory(
    conversation_id: str,
    payload: ConversationMemoryRequest,
    user: AuthUser = Depends(current_user),
) -> ConversationOut:
    load_conversation_for_user(user, conversation_id)
    update_conversation_memory(
        organization_id=user.organization_id,
        user_id=user.id,
        conversation_id=conversation_id,
        memory_lines=payload.memory_lines,
    )
    return refresh_conversation(user, conversation_id)


@app.delete("/api/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["chat"])
def delete_conversation(conversation_id: str, user: AuthUser = Depends(current_user)) -> None:
    load_conversation_for_user(user, conversation_id)
    delete_conversation_record(
        organization_id=user.organization_id,
        user_id=user.id,
        conversation_id=conversation_id,
    )


@app.delete(
    "/api/conversations/{conversation_id}/messages",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["chat"],
)
def clear_messages(conversation_id: str, user: AuthUser = Depends(current_user)) -> None:
    load_conversation_for_user(user, conversation_id)
    clear_conversation_messages(
        organization_id=user.organization_id,
        user_id=user.id,
        conversation_id=conversation_id,
    )


@app.post("/api/conversations/{conversation_id}/ask", response_model=ChatAskResponse, tags=["chat"])
def ask_question(
    conversation_id: str,
    payload: ChatAskRequest,
    user: AuthUser = Depends(current_user),
) -> ChatAskResponse:
    conversation = load_conversation_for_user(user, conversation_id)
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question is required.")

    history_before = conversation.get("messages", [])[-8:]
    memory = conversation.get("memory", []) if payload.use_case_memory else []

    if not conversation.get("messages"):
        update_conversation_title(
            organization_id=user.organization_id,
            conversation_id=conversation_id,
            title=question[:58],
        )

    add_message_record(
        organization_id=user.organization_id,
        user_id=user.id,
        conversation_id=conversation_id,
        role="user",
        content=question,
    )

    started = time.perf_counter()
    answer, evidence = ask_copilot(
        question,
        conversation_history=history_before,
        user_memory=memory,
        force_mode=mode_code(payload.response_mode),
        organization_id=user.organization_id,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)

    add_message_record(
        organization_id=user.organization_id,
        user_id=user.id,
        conversation_id=conversation_id,
        role="bot",
        content=answer,
        citations=evidence,
        latency_ms=latency_ms,
    )

    return ChatAskResponse(
        answer=answer,
        evidence=evidence,
        conversation=refresh_conversation(user, conversation_id),
        latency_ms=latency_ms,
    )
