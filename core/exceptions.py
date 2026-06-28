"""Domain-specific exceptions for Maintenance Copilot services."""

from __future__ import annotations


class MaintenanceCopilotError(Exception):
    """Base exception for application errors."""


class DatabaseUnavailableError(MaintenanceCopilotError):
    """Raised when PostgreSQL is required but not configured or reachable."""


class IndexingError(MaintenanceCopilotError):
    """Raised when document indexing fails."""


class RetrievalError(MaintenanceCopilotError):
    """Raised when vector retrieval fails."""


class GenerationError(MaintenanceCopilotError):
    """Raised when LLM answer generation fails."""
