"""FAISS vector index persistence scoped to a tenant."""

from __future__ import annotations

import logging
import pickle
from typing import Any

import faiss
import numpy as np

from core.tenant import TenantContext

logger = logging.getLogger(__name__)


class VectorRepository:
    """Read and write tenant-scoped FAISS indexes and local metadata caches."""

    def __init__(self, tenant: TenantContext) -> None:
        self.tenant = tenant
        self.tenant.ensure_directories()

    @property
    def stamp(self) -> tuple[float, float] | None:
        if not self.tenant.index_path.exists() or not self.tenant.mapping_path.exists():
            return None
        return (
            self.tenant.index_path.stat().st_mtime,
            self.tenant.mapping_path.stat().st_mtime,
        )

    def load(self) -> tuple[Any | None, list[dict[str, Any]], dict[str, Any]]:
        index = None
        chunks: list[dict[str, Any]] = []
        processed_files: dict[str, Any] = {}

        paths = (
            self.tenant.index_path,
            self.tenant.mapping_path,
            self.tenant.processed_files_path,
        )
        if not all(path.exists() for path in paths):
            return index, chunks, processed_files

        try:
            index = faiss.read_index(str(self.tenant.index_path))
            with self.tenant.mapping_path.open("rb") as handle:
                chunks = pickle.load(handle)
            with self.tenant.processed_files_path.open("rb") as handle:
                processed_files = pickle.load(handle)
        except Exception as exc:
            logger.exception("Failed to load vector store for tenant %s", self.tenant.organization_id)
            raise RuntimeError(f"Vector store read error: {exc}") from exc

        return index, chunks, processed_files

    def save(
        self,
        index: Any,
        chunks: list[dict[str, Any]],
        processed_files: dict[str, Any],
    ) -> None:
        if index is None:
            raise RuntimeError("Cannot save an empty FAISS index.")

        faiss.write_index(index, str(self.tenant.index_path))
        with self.tenant.mapping_path.open("wb") as handle:
            pickle.dump(chunks, handle)
        with self.tenant.processed_files_path.open("wb") as handle:
            pickle.dump(processed_files, handle)
        logger.info(
            "Saved vector index for tenant %s (%s chunks, %s vectors).",
            self.tenant.organization_id,
            len(chunks),
            index.ntotal,
        )

    def save_index_only(self, index: Any, chunks: list[dict[str, Any]]) -> None:
        """Persist index and mapping without updating processed-file signatures."""
        if index is None:
            raise RuntimeError("Cannot save an empty FAISS index.")
        faiss.write_index(index, str(self.tenant.index_path))
        with self.tenant.mapping_path.open("wb") as handle:
            pickle.dump(chunks, handle)

    def clear(self) -> None:
        """Remove tenant-scoped FAISS artifacts."""
        for path in (
            self.tenant.index_path,
            self.tenant.mapping_path,
            self.tenant.processed_files_path,
        ):
            if path.exists():
                path.unlink()

    def read_index(self) -> Any | None:
        if not self.tenant.index_path.exists():
            return None
        return faiss.read_index(str(self.tenant.index_path))

    @staticmethod
    def build_index(embeddings: np.ndarray) -> Any:
        if embeddings.size == 0:
            raise RuntimeError("Cannot build a FAISS index from zero embeddings.")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index
