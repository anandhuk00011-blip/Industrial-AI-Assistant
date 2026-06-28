"""
Database engine, sessions, and SQLAlchemy base.

PostgreSQL is optional during local pilot mode. Services fall back to file-based
metadata when ``DATABASE_URL`` is not configured.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import DATA_DIR, DATABASE_URL

logger = logging.getLogger(__name__)
DATA_DIR.mkdir(parents=True, exist_ok=True)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""


engine: Engine | None = None
SessionLocal: sessionmaker[Session] | None = None

LOCAL_DATABASE_URL = f"sqlite:///{(DATA_DIR / 'maintenance_copilot.db').as_posix()}"
USE_LOCAL_DATABASE = os.getenv("USE_LOCAL_DATABASE", "false").lower() in {"1", "true", "yes"}
ACTIVE_DATABASE_URL = LOCAL_DATABASE_URL if USE_LOCAL_DATABASE else (DATABASE_URL or LOCAL_DATABASE_URL)

if USE_LOCAL_DATABASE:
    logger.warning(
        "USE_LOCAL_DATABASE is enabled. Using local SQLite database at %s.",
        LOCAL_DATABASE_URL,
    )
elif not DATABASE_URL:
    logger.warning(
        "DATABASE_URL is not configured. Using local SQLite database at %s. "
        "Set DATABASE_URL to your Neon Postgres connection string in production.",
        LOCAL_DATABASE_URL,
    )

engine_kwargs: dict[str, object] = {
    "pool_pre_ping": True,
    "pool_recycle": 3600,
    "future": True,
    "echo": False,
}
if ACTIVE_DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {
        "check_same_thread": False,
        "timeout": int(os.getenv("SQLITE_BUSY_TIMEOUT_SECONDS", "60")),
    }

engine = create_engine(ACTIVE_DATABASE_URL, **engine_kwargs)

if ACTIVE_DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute(
            f"PRAGMA busy_timeout={int(os.getenv('SQLITE_BUSY_TIMEOUT_SECONDS', '60')) * 1000}"
        )
        cursor.close()
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def database_enabled() -> bool:
    """Return True when PostgreSQL connectivity is configured."""
    return engine is not None and SessionLocal is not None


def get_db() -> Generator[Session, None, None]:
    """Yield a database session and ensure cleanup."""
    if SessionLocal is None:
        raise RuntimeError(
            "DATABASE_URL is not configured. Add DATABASE_URL to your .env file."
        )
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""
    if SessionLocal is None:
        raise RuntimeError(
            "DATABASE_URL is not configured. Add DATABASE_URL to your .env file."
        )
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_database_connection() -> bool:
    """Verify database connectivity."""
    if engine is None:
        return False
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logger.info("Database connected.")
        return True
    except SQLAlchemyError as exc:
        logger.exception("Database connection failed.")
        raise RuntimeError("Unable to connect to database.") from exc


def create_database() -> None:
    """Create all registered SQLAlchemy tables."""
    if engine is None:
        raise RuntimeError("DATABASE_URL is not configured.")

    import database.models  # noqa: F401 — register models on Base metadata

    logger.info("Initializing PostgreSQL tables via SQLAlchemy...")
    Base.metadata.create_all(bind=engine)
    _apply_lightweight_migrations(engine)
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    logger.info("Provisioned tables: %s", ", ".join(sorted(existing_tables)) or "(none)")


def _apply_lightweight_migrations(active_engine: Engine) -> None:
    """Add non-destructive columns needed by newer app versions."""
    inspector = inspect(active_engine)
    if "chat_sessions" not in inspector.get_table_names():
        return

    existing_columns = {column["name"] for column in inspector.get_columns("chat_sessions")}
    if "memory_lines" in existing_columns:
        return

    ddl = (
        "ALTER TABLE chat_sessions ADD COLUMN memory_lines JSONB NOT NULL DEFAULT '[]'::jsonb"
        if active_engine.dialect.name == "postgresql"
        else "ALTER TABLE chat_sessions ADD COLUMN memory_lines JSON NOT NULL DEFAULT '[]'"
    )
    with active_engine.begin() as connection:
        connection.execute(text(ddl))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    check_database_connection()
    create_database()
    print("Database initialized successfully.")
