"""
Retrieval and answer generation facade for Maintenance Copilot.

Heavy lifting lives in ``services.retrieval_service`` so FastAPI routes and the
Streamlit pilot can share the same tenant-scoped retrieval contract.
"""

from __future__ import annotations

from typing import Any

from config import initialize_storage
from core.tenant import resolve_tenant
from services.retrieval_service import RetrievalService, get_retrieval_service

initialize_storage()

DEFAULT_TENANT = resolve_tenant()
INDEX_PATH = DEFAULT_TENANT.index_path
MAPPING_PATH = DEFAULT_TENANT.mapping_path


def reload_vector_database(force: bool = False, organization_id: str | None = None) -> dict[str, Any]:
    """Load the tenant vector index and refresh retrieval caches."""
    return get_retrieval_service(organization_id).reload(force=force)


def get_database_summary(organization_id: str | None = None) -> dict[str, Any]:
    """Return retrieval status for the active tenant."""
    service = get_retrieval_service(organization_id)
    return service.summary()


def ensure_vector_database_loaded(organization_id: str | None = None) -> bool:
    """Ensure the tenant vector index is loaded in memory."""
    return get_retrieval_service(organization_id).ensure_loaded()


def ask_copilot(
    user_question: str,
    conversation_history: list[dict[str, Any]] | None = None,
    user_memory: list[str] | str | None = None,
    force_mode: str | None = None,
    organization_id: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Return a grounded answer and source evidence for a technician question."""
    return get_retrieval_service(organization_id).ask(
        user_question,
        conversation_history=conversation_history,
        user_memory=user_memory,
        force_mode=force_mode,
    )


__all__ = [
    "RetrievalService",
    "ask_copilot",
    "ensure_vector_database_loaded",
    "get_database_summary",
    "get_retrieval_service",
    "reload_vector_database",
]
