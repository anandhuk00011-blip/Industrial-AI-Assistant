# database/seed.py
from __future__ import annotations

import logging
import uuid

from config import DEFAULT_ORGANIZATION_ID
from core.tenant import TenantContext
from database.database import SessionLocal, database_enabled
from database.models import User, UserRole
from repositories.organization_repository import DEFAULT_ORGANIZATION_NAME, OrganizationRepository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def seed_initial_tenant() -> None:
    if not database_enabled():
        logger.error("DATABASE_URL is not configured. Cannot seed PostgreSQL.")
        return

    session = SessionLocal()
    try:
        tenant = TenantContext(organization_id=DEFAULT_ORGANIZATION_ID)
        org = OrganizationRepository().ensure_exists(session, tenant)
        if org.name != DEFAULT_ORGANIZATION_NAME:
            org.name = DEFAULT_ORGANIZATION_NAME
        session.flush()
        logger.info("Seed organization ready: %s (%s)", org.name, org.id)

        user = session.query(User).filter_by(email="engineer@demo.com").first()
        if not user:
            user = User(
                id=uuid.uuid4(),
                organization_id=org.id,
                email="engineer@demo.com",
                password_hash="pbkdf2:sha256:default_hash_for_dev_only",
                full_name="Anand",
                role=UserRole.OPERATOR.value,
                is_active=True,
            )
            session.add(user)
            logger.info("Created seed user: %s (%s)", user.full_name, user.email)
        else:
            logger.info("Seed user already exists: %s", user.email)

        session.commit()
        logger.info("Database seeding completed.")
    except Exception as exc:
        session.rollback()
        logger.error("Seeding failed: %s", exc)
        raise
    finally:
        session.close()


if __name__ == "__main__":
    seed_initial_tenant()
