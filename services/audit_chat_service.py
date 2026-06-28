"""Database-backed chat, upload, and audit persistence."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from database.database import session_scope
from database.models import AuditLog, ChatMessage, ChatSession, Document


def now_label(value: Any) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    return ""


def log_audit(
    *,
    organization_id: str,
    user_id: str | None,
    action: str,
    details: dict[str, Any] | None = None,
) -> None:
    with session_scope() as session:
        session.add(
            AuditLog(
                organization_id=uuid.UUID(str(organization_id)),
                user_id=uuid.UUID(str(user_id)) if user_id else None,
                action=action,
                details=details or {},
            )
        )


def list_conversations(organization_id: str, user_id: str) -> list[dict[str, Any]]:
    with session_scope() as session:
        sessions = (
            session.query(ChatSession)
            .filter(
                ChatSession.organization_id == uuid.UUID(str(organization_id)),
                ChatSession.user_id == uuid.UUID(str(user_id)),
            )
            .order_by(ChatSession.created_at.desc())
            .all()
        )
        conversations: list[dict[str, Any]] = []
        for chat in sessions:
            messages = (
                session.query(ChatMessage)
                .filter(
                    ChatMessage.organization_id == uuid.UUID(str(organization_id)),
                    ChatMessage.session_id == chat.id,
                )
                .order_by(ChatMessage.created_at.asc())
                .all()
            )
            updated_at = messages[-1].created_at if messages else chat.created_at
            conversations.append(
                {
                    "id": str(chat.id),
                    "title": chat.title,
                    "created_at": now_label(chat.created_at),
                    "updated_at": now_label(updated_at),
                    "messages": [
                        {
                            "id": str(message.id),
                            "role": "bot" if message.role == "assistant" else message.role,
                            "content": message.content,
                            "evidence": message.citations or [],
                            "time": now_label(message.created_at),
                        }
                        for message in messages
                    ],
                    "memory": chat.memory_lines or [],
                }
            )
        return conversations


def create_conversation_record(
    *,
    organization_id: str,
    user_id: str,
    title: str = "New maintenance case",
) -> dict[str, Any]:
    with session_scope() as session:
        chat = ChatSession(
            id=uuid.uuid4(),
            organization_id=uuid.UUID(str(organization_id)),
            user_id=uuid.UUID(str(user_id)),
            title=title,
        )
        session.add(chat)
        session.flush()
        session.add(
            AuditLog(
                organization_id=chat.organization_id,
                user_id=chat.user_id,
                action="chat.session_created",
                details={"session_id": str(chat.id), "title": title},
            )
        )
        return {
            "id": str(chat.id),
            "title": chat.title,
            "created_at": now_label(chat.created_at),
            "updated_at": now_label(chat.created_at),
            "messages": [],
            "memory": chat.memory_lines or [],
        }


def update_conversation_title(
    *,
    organization_id: str,
    conversation_id: str,
    title: str,
) -> None:
    with session_scope() as session:
        chat = (
            session.query(ChatSession)
            .filter(
                ChatSession.organization_id == uuid.UUID(str(organization_id)),
                ChatSession.id == uuid.UUID(str(conversation_id)),
            )
            .one_or_none()
        )
        if chat:
            chat.title = title[:255]


def update_conversation_memory(
    *,
    organization_id: str,
    user_id: str,
    conversation_id: str,
    memory_lines: list[str],
) -> None:
    clean_lines = [line[:1000] for line in memory_lines[:40] if line.strip()]
    with session_scope() as session:
        chat = (
            session.query(ChatSession)
            .filter(
                ChatSession.organization_id == uuid.UUID(str(organization_id)),
                ChatSession.user_id == uuid.UUID(str(user_id)),
                ChatSession.id == uuid.UUID(str(conversation_id)),
            )
            .one_or_none()
        )
        if chat:
            chat.memory_lines = clean_lines
            session.add(
                AuditLog(
                    organization_id=uuid.UUID(str(organization_id)),
                    user_id=uuid.UUID(str(user_id)),
                    action="chat.memory_updated",
                    details={"session_id": conversation_id, "memory_lines": clean_lines},
                )
            )


def add_message_record(
    *,
    organization_id: str,
    user_id: str,
    conversation_id: str,
    role: str,
    content: str,
    citations: list[dict[str, Any]] | None = None,
    latency_ms: int | None = None,
) -> dict[str, Any]:
    db_role = "assistant" if role == "bot" else role
    with session_scope() as session:
        message = ChatMessage(
            id=uuid.uuid4(),
            session_id=uuid.UUID(str(conversation_id)),
            organization_id=uuid.UUID(str(organization_id)),
            role=db_role,
            content=content,
            citations=citations,
            latency_ms=latency_ms,
        )
        session.add(message)
        session.add(
            AuditLog(
                organization_id=message.organization_id,
                user_id=uuid.UUID(str(user_id)),
                action=f"chat.message_{db_role}",
                details={
                    "session_id": str(conversation_id),
                    "message_id": str(message.id),
                    "citations": citations or [],
                },
            )
        )
        session.flush()
        return {
            "id": str(message.id),
            "role": role,
            "content": content,
            "evidence": citations or [],
            "time": now_label(message.created_at),
        }


def delete_conversation_record(
    *,
    organization_id: str,
    user_id: str,
    conversation_id: str,
) -> None:
    with session_scope() as session:
        chat = (
            session.query(ChatSession)
            .filter(
                ChatSession.organization_id == uuid.UUID(str(organization_id)),
                ChatSession.user_id == uuid.UUID(str(user_id)),
                ChatSession.id == uuid.UUID(str(conversation_id)),
            )
            .one_or_none()
        )
        if chat:
            session.delete(chat)
            session.add(
                AuditLog(
                    organization_id=uuid.UUID(str(organization_id)),
                    user_id=uuid.UUID(str(user_id)),
                    action="chat.session_deleted",
                    details={"session_id": conversation_id},
                )
            )


def clear_conversation_messages(
    *,
    organization_id: str,
    user_id: str,
    conversation_id: str,
) -> None:
    with session_scope() as session:
        session.query(ChatMessage).filter(
            ChatMessage.organization_id == uuid.UUID(str(organization_id)),
            ChatMessage.session_id == uuid.UUID(str(conversation_id)),
        ).delete(synchronize_session=False)
        session.add(
            AuditLog(
                organization_id=uuid.UUID(str(organization_id)),
                user_id=uuid.UUID(str(user_id)),
                action="chat.messages_cleared",
                details={"session_id": conversation_id},
            )
        )


def record_uploaded_file(
    *,
    organization_id: str,
    user_id: str,
    file_path: Path,
    original_name: str,
) -> None:
    with session_scope() as session:
        session.add(
            AuditLog(
                organization_id=uuid.UUID(str(organization_id)),
                user_id=uuid.UUID(str(user_id)),
                action="document.uploaded",
                details={
                    "original_name": original_name,
                    "stored_name": file_path.name,
                    "file_path": str(file_path),
                    "size_bytes": file_path.stat().st_size if file_path.exists() else None,
                },
            )
        )


def list_document_records(organization_id: str) -> list[dict[str, Any]]:
    with session_scope() as session:
        docs = (
            session.query(Document)
            .filter(Document.organization_id == uuid.UUID(str(organization_id)))
            .order_by(Document.created_at.desc())
            .all()
        )
        return [
            {
                "id": str(doc.id),
                "file_name": doc.file_name,
                "status": doc.status,
                "size_bytes": doc.file_size_bytes,
                "created_at": now_label(doc.created_at),
            }
            for doc in docs
        ]
