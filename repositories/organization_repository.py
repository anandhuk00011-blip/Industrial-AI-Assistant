"""Persistence layer for organization (tenant) records."""

from __future__ import annotations

import logging
import os

from sqlalchemy.orm import Session

from core.tenant import TenantContext
from database.models import Organization

logger = logging.getLogger(__name__)

DEFAULT_ORGANIZATION_NAME = os.getenv("DEFAULT_ORGANIZATION_NAME", "Default Organization")


class OrganizationRepository:
    """PostgreSQL-backed tenant registry."""

    def ensure_exists(self, session: Session, tenant: TenantContext) -> Organization:
        """
        Guarantee the tenant organization row exists before document writes.

        Creates the organization on first use so indexing does not fail with a
        foreign-key violation when ``DATABASE_URL`` is configured but seed has
        not been run yet.
        """
        organization = (
            session.query(Organization)
            .filter(Organization.id == tenant.organization_id)
            .one_or_none()
        )
        if organization is not None:
            return organization

        organization = Organization(
            id=tenant.organization_id,
            name=DEFAULT_ORGANIZATION_NAME,
        )
        session.add(organization)
        session.flush()
        logger.info(
            "Registered organization %s (%s) for tenant bootstrap.",
            organization.name,
            organization.id,
        )
        return organization


def ensure_tenant_organization(session: Session, tenant: TenantContext) -> Organization:
    """Convenience wrapper used by indexing and retrieval services."""
    return OrganizationRepository().ensure_exists(session, tenant)
