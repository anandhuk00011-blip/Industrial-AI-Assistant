"""Core domain primitives shared across services."""

from core.tenant import TenantContext, resolve_tenant

__all__ = ["TenantContext", "resolve_tenant"]
