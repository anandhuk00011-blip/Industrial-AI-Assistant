"""
Document indexing orchestration for multi-tenant Maintenance Copilot.

Coordinates extraction, PostgreSQL metadata, and tenant-scoped FAISS indexes.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Callable

import faiss

from core.tenant import TenantContext, migrate_legacy_tenant_storage, resolve_tenant
from database.database import database_enabled, session_scope
from database.models import DocumentStatus
from repositories.document_repository import DocumentRepository, compute_file_checksum
from repositories.organization_repository import ensure_tenant_organization
from repositories.vector_repository import VectorRepository

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict[str, Any]], None]


class IndexingService:
    """Tenant-scoped indexing pipeline with PostgreSQL and file-store sync."""

    def __init__(self, tenant: TenantContext) -> None:
        self.tenant = tenant
        self.documents = DocumentRepository(tenant)
        self.vectors = VectorRepository(tenant)

    def index_documents(
        self,
        input_folder: Path,
        *,
        ocr_mode: str,
        force_rebuild: bool,
        target_chunk_len: int,
        overlap_sentences: int,
        progress_callback: ProgressCallback | None = None,
        process_document: Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]],
        encode_chunks: Callable[[list[dict[str, Any]]], Any],
        get_embedding_backend_name: Callable[[], str],
        file_signature: Callable[[Path, str], dict[str, Any]],
        signature_matches: Callable[[Any, dict[str, Any]], bool],
        deduplicate_chunks: Callable[[list[dict[str, Any]]], tuple[list[dict[str, Any]], int]],
        reset_chunk_ids: Callable[[list[dict[str, Any]]], None],
        supported_extensions: set[str],
        emit: Callable[..., None],
    ) -> dict[str, Any]:
        from main import DEFAULT_OCR_MODE  # local import avoids circular dependency at module load

        input_folder = Path(input_folder)
        input_folder.mkdir(parents=True, exist_ok=True)

        selected_ocr_mode = (ocr_mode or DEFAULT_OCR_MODE).lower()
        if selected_ocr_mode not in {"auto", "always", "off"}:
            selected_ocr_mode = "auto"

        document_files = sorted(
            path
            for path in input_folder.iterdir()
            if path.is_file() and path.suffix.lower() in supported_extensions
        )

        summary: dict[str, Any] = {
            "organization_id": str(self.tenant.organization_id),
            "files_seen": len(document_files),
            "files_processed": 0,
            "files_skipped": 0,
            "chunks_added": 0,
            "chunks_total": 0,
            "vectors_total": 0,
            "ocr_pages": 0,
            "document_types": {},
            "warnings": [],
            "database_sync": database_enabled(),
        }

        if not document_files:
            if force_rebuild:
                if database_enabled():
                    with session_scope() as session:
                        ensure_tenant_organization(session, self.tenant)
                        self.documents.delete_all(session)
                self.vectors.clear()
            emit(progress_callback, "done", message="No supported documents found.")
            return summary

        if database_enabled():
            return self._index_with_database(
                document_files=document_files,
                selected_ocr_mode=selected_ocr_mode,
                force_rebuild=force_rebuild,
                target_chunk_len=target_chunk_len,
                overlap_sentences=overlap_sentences,
                progress_callback=progress_callback,
                process_document=process_document,
                encode_chunks=encode_chunks,
                get_embedding_backend_name=get_embedding_backend_name,
                summary=summary,
                emit=emit,
            )

        return self._index_with_files(
            input_folder=input_folder,
            document_files=document_files,
            selected_ocr_mode=selected_ocr_mode,
            force_rebuild=force_rebuild,
            target_chunk_len=target_chunk_len,
            overlap_sentences=overlap_sentences,
            progress_callback=progress_callback,
            process_document=process_document,
            encode_chunks=encode_chunks,
            file_signature=file_signature,
            signature_matches=signature_matches,
            deduplicate_chunks=deduplicate_chunks,
            reset_chunk_ids=reset_chunk_ids,
            summary=summary,
            emit=emit,
        )

    def _index_with_database(
        self,
        *,
        document_files: list[Path],
        selected_ocr_mode: str,
        force_rebuild: bool,
        target_chunk_len: int,
        overlap_sentences: int,
        progress_callback: ProgressCallback | None,
        process_document: Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]],
        encode_chunks: Callable[[list[dict[str, Any]]], Any],
        get_embedding_backend_name: Callable[[], str],
        summary: dict[str, Any],
        emit: Callable[..., None],
    ) -> dict[str, Any]:
        all_chunks: list[dict[str, Any]] = []
        new_chunks: list[dict[str, Any]] = []
        rebuild_required = force_rebuild
        backend_name = get_embedding_backend_name()

        with session_scope() as session:
            ensure_tenant_organization(session, self.tenant)

            if force_rebuild:
                logger.info(
                    "Rebuilding tenant index for organization %s",
                    self.tenant.organization_id,
                )
                self.documents.delete_all(session)
                self.vectors.clear()

            active_checksums: set[str] = set()
            for document_path in document_files:
                suffix = document_path.suffix.lower()
                summary["document_types"][suffix] = summary["document_types"].get(suffix, 0) + 1
                try:
                    checksum = compute_file_checksum(document_path)
                except OSError as exc:
                    summary["warnings"].append(f"Could not hash {document_path.name}: {exc}")
                    continue
                active_checksums.add(checksum)

            removed = self.documents.delete_missing_from_disk(session, active_checksums)
            if removed:
                summary["warnings"].append(f"Removed {removed} stale document record(s).")
                rebuild_required = True

            next_chunk_id = 0
            for document_path in document_files:
                suffix = document_path.suffix.lower()
                try:
                    checksum = compute_file_checksum(document_path)
                except OSError:
                    continue

                existing = self.documents.get_by_checksum(session, checksum)
                if (
                    existing is not None
                    and existing.status == DocumentStatus.INDEXED.value
                    and not force_rebuild
                ):
                    summary["files_skipped"] += 1
                    emit(
                        progress_callback,
                        "skip_file",
                        message=f"Skipping unchanged document: {document_path.name}",
                        file=document_path.name,
                    )
                    for chunk in sorted(existing.chunks, key=lambda item: item.chunk_index):
                        all_chunks.append(
                            {
                                "chunk_id": chunk.chunk_index,
                                "document_id": str(existing.id),
                                "source_file": existing.file_name,
                                "filename": existing.file_name,
                                "page": chunk.page_number or 1,
                                "page_number": chunk.page_number or 1,
                                "section_title": (chunk.chunk_metadata or {}).get(
                                    "section_title", "General"
                                ),
                                "machine_type": (chunk.chunk_metadata or {}).get(
                                    "machine_type", "Unknown"
                                ),
                                "manufacturer": (chunk.chunk_metadata or {}).get(
                                    "manufacturer", "Unknown"
                                ),
                                "revision": (chunk.chunk_metadata or {}).get(
                                    "revision", "unknown"
                                ),
                                "language": (chunk.chunk_metadata or {}).get(
                                    "language", "unknown"
                                ),
                                "source": existing.file_path,
                                "text": chunk.content,
                                "extraction": (chunk.chunk_metadata or {}).get(
                                    "extraction", "database-cache"
                                ),
                                "embedding_backend": backend_name,
                            }
                        )
                    next_chunk_id = len(all_chunks)
                    continue

                if existing is not None:
                    rebuild_required = True

                emit(
                    progress_callback,
                    "process_file",
                    message=f"Processing document: {document_path.name}",
                    file=document_path.name,
                )
                self.documents.replace_document(session, existing)
                document = self.documents.create_processing(
                    session,
                    file_name=document_path.name,
                    file_path=document_path,
                    mime_type=f"application/{suffix.strip('.')}",
                    checksum=checksum,
                )
                session.flush()

                file_chunks, file_stats = process_document(
                    document_path,
                    document_path.name,
                    next_chunk_id,
                    target_chunk_len=target_chunk_len,
                    overlap_sentences=overlap_sentences,
                    ocr_mode=selected_ocr_mode,
                    progress_callback=progress_callback,
                )

                summary["files_processed"] += 1
                summary["chunks_added"] += len(file_chunks)
                summary["ocr_pages"] += int(file_stats.get("ocr_pages", 0))
                summary["warnings"].extend(file_stats.get("warnings", []))

                if file_chunks:
                    for chunk in file_chunks:
                        chunk["embedding_backend"] = backend_name
                    enriched = self.documents.persist_chunks(session, document, file_chunks)
                    all_chunks.extend(enriched)
                    new_chunks.extend(enriched)
                    next_chunk_id = len(all_chunks)
                else:
                    self.documents.mark_failed(
                        session,
                        document,
                        "No extractable content was found in the document.",
                    )

        if not all_chunks:
            self.vectors.clear()
            summary["chunks_total"] = 0
            summary["vectors_total"] = 0
        elif summary["files_processed"] == 0 and self.tenant.index_path.exists():
            existing_index = self.vectors.read_index()
            summary["chunks_total"] = len(all_chunks)
            summary["vectors_total"] = existing_index.ntotal if existing_index is not None else 0
        elif new_chunks and not rebuild_required and self.tenant.index_path.exists():
            emit(
                progress_callback,
                "embed",
                message=f"Embedding {len(new_chunks)} new chunks.",
            )
            existing_index = self.vectors.read_index()
            if existing_index is not None and existing_index.ntotal == len(all_chunks) - len(new_chunks):
                matrix = encode_chunks(new_chunks)
                existing_index.add(matrix)
                self.vectors.save_index_only(existing_index, all_chunks)
                summary["chunks_total"] = len(all_chunks)
                summary["vectors_total"] = existing_index.ntotal
            else:
                rebuild_required = True

        if all_chunks and (rebuild_required or summary["vectors_total"] == 0):
            emit(
                progress_callback,
                "embed",
                message=f"Building vector index for {len(all_chunks)} chunks.",
            )
            matrix = encode_chunks(all_chunks)
            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            self.vectors.save_index_only(index, all_chunks)
            summary["chunks_total"] = len(all_chunks)
            summary["vectors_total"] = index.ntotal

        emit(
            progress_callback,
            "done",
            message=(
                f"Index ready: {summary['chunks_total']} chunks, "
                f"{summary['vectors_total']} vectors."
            ),
        )
        return summary

    def _index_with_files(
        self,
        *,
        input_folder: Path,
        document_files: list[Path],
        selected_ocr_mode: str,
        force_rebuild: bool,
        target_chunk_len: int,
        overlap_sentences: int,
        progress_callback: ProgressCallback | None,
        process_document: Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]],
        encode_chunks: Callable[[list[dict[str, Any]]], Any],
        file_signature: Callable[[Path, str], dict[str, Any]],
        signature_matches: Callable[[Any, dict[str, Any]], bool],
        deduplicate_chunks: Callable[[list[dict[str, Any]]], tuple[list[dict[str, Any]], int]],
        reset_chunk_ids: Callable[[list[dict[str, Any]]], None],
        summary: dict[str, Any],
        emit: Callable[..., None],
    ) -> dict[str, Any]:
        if force_rebuild:
            index = None
            chunks_data: list[dict[str, Any]] = []
            processed_files: dict[str, Any] = {}
            rebuild_required = True
        else:
            index, chunks_data, processed_files = self.vectors.load()
            chunks_data, duplicate_count = deduplicate_chunks(chunks_data)
            rebuild_required = duplicate_count > 0
            if duplicate_count:
                summary["warnings"].append(f"Removed {duplicate_count} duplicate chunks.")
            if index is not None and index.ntotal != len(chunks_data):
                rebuild_required = True
                summary["warnings"].append("Index and metadata counts differed; rebuilding.")

        current_files = {path.name for path in document_files}
        stale_files = set(processed_files) - current_files
        if stale_files:
            chunks_data = [
                chunk for chunk in chunks_data if chunk.get("source_file") not in stale_files
            ]
            for filename in stale_files:
                processed_files.pop(filename, None)
            rebuild_required = True

        new_chunks: list[dict[str, Any]] = []

        for document_path in document_files:
            suffix = document_path.suffix.lower()
            summary["document_types"][suffix] = summary["document_types"].get(suffix, 0) + 1
            signature = file_signature(document_path, selected_ocr_mode)
            previous_signature = processed_files.get(document_path.name)

            if not force_rebuild and signature_matches(previous_signature, signature):
                summary["files_skipped"] += 1
                emit(
                    progress_callback,
                    "skip_file",
                    message=f"Skipping unchanged file: {document_path.name}",
                    file=document_path.name,
                )
                continue

            emit(
                progress_callback,
                "process_file",
                message=f"Processing document: {document_path.name}",
                file=document_path.name,
            )

            old_count = len(chunks_data)
            chunks_data = [
                chunk for chunk in chunks_data if chunk.get("source_file") != document_path.name
            ]
            if len(chunks_data) != old_count:
                rebuild_required = True

            file_chunks, file_stats = process_document(
                document_path,
                document_path.name,
                len(chunks_data) + len(new_chunks),
                target_chunk_len=target_chunk_len,
                overlap_sentences=overlap_sentences,
                ocr_mode=selected_ocr_mode,
                progress_callback=progress_callback,
            )

            summary["files_processed"] += 1
            summary["chunks_added"] += len(file_chunks)
            summary["ocr_pages"] += int(file_stats.get("ocr_pages", 0))
            summary["warnings"].extend(file_stats.get("warnings", []))

            if file_chunks:
                new_chunks.extend(file_chunks)
                processed_files[document_path.name] = signature

        if rebuild_required:
            chunks_data.extend(new_chunks)
            reset_chunk_ids(chunks_data)
            emit(
                progress_callback,
                "embed",
                message=f"Rebuilding vector index for {len(chunks_data)} chunks.",
            )
            matrix = encode_chunks(chunks_data)
            if matrix.size:
                index = self.vectors.build_index(matrix)
                self.vectors.save(index, chunks_data, processed_files)
        elif new_chunks:
            emit(
                progress_callback,
                "embed",
                message=f"Embedding {len(new_chunks)} new chunks.",
            )
            matrix = encode_chunks(new_chunks)
            if matrix.size:
                if index is None:
                    index = self.vectors.build_index(matrix)
                else:
                    index.add(matrix)
                chunks_data.extend(new_chunks)
                self.vectors.save(index, chunks_data, processed_files)

        summary["chunks_total"] = len(chunks_data)
        summary["vectors_total"] = index.ntotal if index is not None else 0
        emit(
            progress_callback,
            "done",
            message=(
                f"Index ready: {summary['chunks_total']} chunks, "
                f"{summary['vectors_total']} vectors."
            ),
        )
        return summary


def index_documents_for_tenant(
    input_folder: str | Path,
    *,
    organization_id: uuid.UUID | str | None = None,
    ocr_mode: str | None = None,
    force_rebuild: bool = False,
    target_chunk_len: int | None = None,
    overlap_sentences: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Public entry point used by ``main.py`` and future FastAPI routes."""
    from main import (
        DEFAULT_CHUNK_LENGTH,
        DEFAULT_OVERLAP_SENTENCES,
        SUPPORTED_DOCUMENT_EXTENSIONS,
        _emit,
        deduplicate_chunks,
        encode_chunks,
        file_signature,
        get_embedding_backend_name,
        process_single_document,
        reset_chunk_ids,
        signature_matches,
    )

    tenant = resolve_tenant(organization_id)
    migrate_legacy_tenant_storage(tenant)
    service = IndexingService(tenant)

    return service.index_documents(
        Path(input_folder),
        ocr_mode=ocr_mode or "auto",
        force_rebuild=force_rebuild,
        target_chunk_len=target_chunk_len or DEFAULT_CHUNK_LENGTH,
        overlap_sentences=overlap_sentences or DEFAULT_OVERLAP_SENTENCES,
        progress_callback=progress_callback,
        process_document=process_single_document,
        encode_chunks=encode_chunks,
        get_embedding_backend_name=get_embedding_backend_name,
        file_signature=file_signature,
        signature_matches=signature_matches,
        deduplicate_chunks=deduplicate_chunks,
        reset_chunk_ids=reset_chunk_ids,
        supported_extensions=SUPPORTED_DOCUMENT_EXTENSIONS,
        emit=_emit,
    )
