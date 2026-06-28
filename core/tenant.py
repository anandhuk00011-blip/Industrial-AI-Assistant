"""
Multi-tenant context for document storage and vector indexes.

Each organization receives an isolated directory tree under ``data/tenants/``.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from config import BASE_DIR, DEFAULT_ORGANIZATION_ID

logger = logging.getLogger(__name__)

TENANTS_ROOT = BASE_DIR / "data" / "tenants"
DEFAULT_TENANT_SLUG = "default"


@dataclass(frozen=True, slots=True)
class TenantContext:
    """Resolved tenant scope for indexing, retrieval, and storage."""

    organization_id: uuid.UUID
    slug: str = DEFAULT_TENANT_SLUG

    @property
    def root(self) -> Path:
        return TENANTS_ROOT / str(self.organization_id)

    @property
    def uploads_dir(self) -> Path:
        return self.root / "uploads"

    @property
    def faiss_dir(self) -> Path:
        return self.root / "faiss"

    @property
    def index_path(self) -> Path:
        return self.faiss_dir / "maintenance_index.faiss"

    @property
    def mapping_path(self) -> Path:
        return self.faiss_dir / "chunks_mapping.pkl"

    @property
    def processed_files_path(self) -> Path:
        return self.faiss_dir / "processed_files.pkl"

    def ensure_directories(self) -> None:
        """Create tenant-scoped directories."""
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_dir.mkdir(parents=True, exist_ok=True)


def resolve_tenant(
    organization_id: uuid.UUID | str | None = None,
    slug: str = DEFAULT_TENANT_SLUG,
) -> TenantContext:
    """
    Resolve the active tenant context.

    Priority:
    1. Explicit ``organization_id`` argument
    2. ``DEFAULT_ORGANIZATION_ID`` environment variable
    3. Built-in default UUID for single-tenant pilot mode
    """
    if organization_id is None:
        organization_id = DEFAULT_ORGANIZATION_ID

    if isinstance(organization_id, str):
        organization_id = uuid.UUID(organization_id)

    tenant = TenantContext(organization_id=organization_id, slug=slug)
    tenant.ensure_directories()
    return tenant


def migrate_legacy_tenant_storage(tenant: TenantContext) -> list[str]:
    """
    Move single-tenant pilot files into the tenant directory once.

    Migrates from ``data/uploads`` and ``data/faiss`` when the tenant tree
    is still empty.
    """
    from config import (
        CHUNKS_MAPPING_PATH,
        FAISS_INDEX_PATH,
        PROCESSED_FILES_PATH,
        UPLOADS_DIR,
    )

    moved: list[str] = []
    legacy_pairs = (
        (UPLOADS_DIR, tenant.uploads_dir),
        (FAISS_INDEX_PATH, tenant.index_path),
        (CHUNKS_MAPPING_PATH, tenant.mapping_path),
        (PROCESSED_FILES_PATH, tenant.processed_files_path),
    )

    for source, destination in legacy_pairs:
        if source.is_dir():
            if any(tenant.uploads_dir.iterdir()) if tenant.uploads_dir.exists() else False:
                continue
            for item in source.iterdir():
                if item.name == ".gitkeep" or not item.is_file():
                    continue
                target = destination / item.name
                if target.exists():
                    continue
                destination.mkdir(parents=True, exist_ok=True)
                item.replace(target)
                moved.append(f"{item.name} -> {target.relative_to(BASE_DIR)}")
            continue

        if source.is_file() and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
            moved.append(f"{source.name} -> {destination.relative_to(BASE_DIR)}")

    if moved:
        logger.info("Migrated legacy storage for tenant %s: %s", tenant.organization_id, moved)
    return moved
