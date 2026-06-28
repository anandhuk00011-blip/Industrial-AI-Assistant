"""API authentication helpers."""

from __future__ import annotations

import os
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from database.database import session_scope
from database.models import User
from services.auth_service import AuthUser

ACCESS_TOKEN_MAX_AGE_SECONDS = int(os.getenv("ACCESS_TOKEN_MAX_AGE_SECONDS", "28800"))
API_SECRET_KEY = os.getenv("API_SECRET_KEY") or os.getenv("SECRET_KEY") or "dev-maintenance-copilot-secret"

bearer_scheme = HTTPBearer(auto_error=False)
serializer = URLSafeTimedSerializer(API_SECRET_KEY, salt="maintenance-copilot-api")


def create_access_token(user: AuthUser) -> str:
    return serializer.dumps(
        {
            "user_id": user.id,
            "organization_id": user.organization_id,
            "email": user.email,
        }
    )


def verify_access_token(token: str) -> dict[str, Any]:
    try:
        payload = serializer.loads(token, max_age=ACCESS_TOKEN_MAX_AGE_SECONDS)
    except SignatureExpired as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired. Please sign in again.",
        ) from exc
    except BadSignature as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token.",
        ) from exc

    if not isinstance(payload, dict) or not payload.get("user_id"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token.",
        )
    return payload


def _auth_user_from_db_id(user_id: str) -> AuthUser:
    with session_scope() as session:
        user = session.query(User).filter(User.id == user_id, User.is_active.is_(True)).one_or_none()
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User no longer exists or is inactive.",
            )
        return AuthUser(
            id=str(user.id),
            organization_id=str(user.organization_id),
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            organization_name=user.organization.name if user.organization else "Organization",
        )


def current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> AuthUser:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
        )
    payload = verify_access_token(credentials.credentials)
    return _auth_user_from_db_id(str(payload["user_id"]))
