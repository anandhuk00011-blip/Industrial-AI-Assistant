"""Persistence layer for document metadata and chunks."""

from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session, joinedload

from core.tenant import TenantContext
from database.database import database_enabled, session_scope
from database.models import Document, DocumentChunk, DocumentStatus
from repositories.organization_repository import ensure_tenant_organization

logger = logging.getLogger(__name__)


def compute_file_checksum(path: Path) -> str:
    """Return the MD5 checksum for a file on disk."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DocumentRepository:
    """PostgreSQL-backed document registry scoped to one organization."""

    def __init__(self, tenant: TenantContext) -> None:
        self.tenant = tenant

    def list_documents(self, session: Session) -> list[Document]:
        return (
            session.query(Document)
            .options(joinedload(Document.chunks))
            .filter(Document.organization_id == self.tenant.organization_id)
            .all()
        )

    def delete_all(self, session: Session) -> None:
        session.query(Document).filter(
            Document.organization_id == self.tenant.organization_id
        ).delete(synchronize_session=False)

    def delete_missing_from_disk(
        self,
        session: Session,
        active_checksums: set[str],
    ) -> int:
        removed = 0
        for document in self.list_documents(session):
            if document.md5_checksum not in active_checksums:
                logger.info(
                    "Removing stale document registry entry: %s",
                    document.file_name,
                )
                session.delete(document)
                removed += 1
        return removed

    def get_by_checksum(self, session: Session, checksum: str) -> Document | None:
        return (
            session.query(Document)
            .options(joinedload(Document.chunks))
            .filter(
                Document.organization_id == self.tenant.organization_id,
                Document.md5_checksum == checksum,
            )
            .one_or_none()
        )

    def get_by_id(self, session: Session, document_id: uuid.UUID | str) -> Document | None:
        return (
            session.query(Document)
            .options(joinedload(Document.chunks))
            .filter(
                Document.organization_id == self.tenant.organization_id,
                Document.id == uuid.UUID(str(document_id)),
            )
            .one_or_none()
        )

    def delete_by_id(self, session: Session, document_id: uuid.UUID | str) -> Document | None:
        document = self.get_by_id(session, document_id)
        if document is None:
            return None
        session.delete(document)
        session.flush()
        return document

    def create_processing(
        self,
        session: Session,
        *,
        file_name: str,
        file_path: Path,
        mime_type: str,
        checksum: str,
    ) -> Document:
        document = Document(
            id=uuid.uuid4(),
            organization_id=self.tenant.organization_id,
            file_name=file_name,
            file_path=str(file_path),
            file_size_bytes=file_path.stat().st_size,
            mime_type=mime_type,
            md5_checksum=checksum,
            status=DocumentStatus.PROCESSING.value,
        )
        session.add(document)
        session.flush()
        return document

    def replace_document(
        self,
        session: Session,
        existing: Document | None,
    ) -> None:
        if existing is not None:
            session.delete(existing)
            session.flush()

    def persist_chunks(
        self,
        session: Session,
        document: Document,
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        enriched: list[dict[str, Any]] = []
        for chunk in chunks:
            vector_index_id = f"{document.id}_{chunk['chunk_id']}"
            chunk_metadata = {
                "vector_index_id": vector_index_id,
                "embedding_backend": chunk.get("embedding_backend"),
                "section_title": chunk.get("section_title", "General"),
                "machine_type": chunk.get("machine_type", "Unknown"),
                "manufacturer": chunk.get("manufacturer", "Unknown"),
                "revision": chunk.get("revision", "unknown"),
                "language": chunk.get("language", "unknown"),
                "extraction": chunk.get("extraction", "native"),
                "source_file": chunk.get("source_file", document.file_name),
            }
            db_chunk = DocumentChunk(
                id=uuid.uuid4(),
                document_id=document.id,
                organization_id=self.tenant.organization_id,
                chunk_index=int(chunk["chunk_id"]),
                content=str(chunk["text"]),
                page_number=int(chunk["page"]) if isinstance(chunk.get("page"), int) else None,
                chunk_metadata=chunk_metadata,
            )
            session.add(db_chunk)
            enriched.append(
                {
                    **chunk,
                    "document_id": str(document.id),
                    "vector_index_id": vector_index_id,
                }
            )
        document.status = DocumentStatus.INDEXED.value
        document.error_message = None
        return enriched

    def mark_failed(self, session: Session, document: Document, message: str) -> None:
        document.status = DocumentStatus.FAILED.value
        document.error_message = message

    def chunks_to_metadata(self, session: Session) -> list[dict[str, Any]]:
        rows = (
            session.query(DocumentChunk)
            .join(Document)
            .options(joinedload(DocumentChunk.document))
            .filter(Document.organization_id == self.tenant.organization_id)
            .order_by(DocumentChunk.chunk_index)
            .all()
        )
        metadata: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            extra = row.chunk_metadata or {}
            metadata.append(
                {
                    "chunk_id": row.chunk_index,
                    "internal_idx": index,
                    "document_id": str(row.document_id),
                    "source_file": row.document.file_name,
                    "filename": row.document.file_name,
                    "page": row.page_number or 1,
                    "page_number": row.page_number or 1,
                    "section_title": extra.get("section_title", "General"),
                    "machine_type": extra.get("machine_type", "Unknown"),
                    "manufacturer": extra.get("manufacturer", "Unknown"),
                    "revision": extra.get("revision", "unknown"),
                    "language": extra.get("language", "unknown"),
                    "source": row.document.file_path,
                    "text": row.content,
                    "extraction": extra.get("extraction", "database"),
                    "embedding_backend": extra.get("embedding_backend"),
                    "vector_index_id": extra.get("vector_index_id"),
                }
            )
        return metadata


def load_chunk_metadata(tenant: TenantContext) -> list[dict[str, Any]] | None:
    """Load chunk metadata from PostgreSQL when configured."""
    if not database_enabled():
        return None
    repository = DocumentRepository(tenant)
    with session_scope() as session:
        ensure_tenant_organization(session, tenant)
        metadata = repository.chunks_to_metadata(session)
    return metadata
