"""Authentication and tenant bootstrap for MaintenanceCopilot AI."""

from __future__ import annotations

import hashlib
import hmac
import os
import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from database.database import create_database, session_scope
from database.models import AuditLog, Organization, User, UserRole

PBKDF2_ITERATIONS = int(os.getenv("PASSWORD_HASH_ITERATIONS", "260000"))


@dataclass(frozen=True)
class AuthUser:
    id: str
    organization_id: str
    email: str
    full_name: str
    role: str
    organization_name: str

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "email": self.email,
            "full_name": self.full_name,
            "role": self.role,
            "organization_name": self.organization_name,
        }


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations_raw, salt_hex, digest_hex = stored_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(iterations_raw),
        )
        return hmac.compare_digest(digest.hex(), digest_hex)
    except Exception:
        return False


def initialize_auth_storage() -> None:
    create_database()


def _auth_user_from_model(user: User) -> AuthUser:
    return AuthUser(
        id=str(user.id),
        organization_id=str(user.organization_id),
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        organization_name=user.organization.name if user.organization else "Organization",
    )


def create_account(
    *,
    organization_name: str,
    email: str,
    password: str,
    full_name: str,
) -> AuthUser:
    clean_email = email.strip().lower()
    if not clean_email or "@" not in clean_email:
        raise ValueError("Enter a valid work email.")
    if len(password) < 10:
        raise ValueError("Password must be at least 10 characters.")
    if not organization_name.strip():
        raise ValueError("Organization name is required.")

    with session_scope() as session:
        existing = session.query(User).filter(User.email == clean_email).one_or_none()
        if existing is not None:
            raise ValueError("An account with this email already exists.")

        organization = Organization(
            id=uuid.uuid4(),
            name=organization_name.strip(),
            plan_tier="enterprise",
        )
        user = User(
            id=uuid.uuid4(),
            organization_id=organization.id,
            email=clean_email,
            password_hash=hash_password(password),
            full_name=full_name.strip() or clean_email.split("@", 1)[0],
            role=UserRole.ADMIN.value,
            is_active=True,
        )
        session.add(organization)
        session.add(user)
        session.flush()
        session.add(
            AuditLog(
                organization_id=organization.id,
                user_id=user.id,
                action="auth.account_created",
                details={"email": clean_email, "organization": organization.name},
            )
        )
        session.refresh(user)
        return _auth_user_from_model(user)


def authenticate(email: str, password: str) -> AuthUser | None:
    clean_email = email.strip().lower()
    with session_scope() as session:
        user = session.query(User).filter(User.email == clean_email).one_or_none()
        if user is None or not user.is_active:
            return None
        if not verify_password(password, user.password_hash):
            session.add(
                AuditLog(
                    organization_id=user.organization_id,
                    user_id=user.id,
                    action="auth.login_failed",
                    details={"email": clean_email},
                )
            )
            return None
        session.add(
            AuditLog(
                organization_id=user.organization_id,
                user_id=user.id,
                action="auth.login_success",
                details={"email": clean_email},
            )
        )
        session.refresh(user)
        return _auth_user_from_model(user)


def has_any_users(session: Session) -> bool:
    return session.query(User.id).first() is not None
