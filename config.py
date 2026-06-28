"""
Application configuration and storage path definitions.

All runtime file locations are defined here so uploads, vector indexes,
and chat history follow a single, predictable layout under ``data/``.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------------------------
# Data layout (production pilot)
# ---------------------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
FAISS_DIR = DATA_DIR / "faiss"
CHAT_HISTORY_DIR = DATA_DIR / "chat_history"

FAISS_INDEX_PATH = FAISS_DIR / "maintenance_index.faiss"
CHUNKS_MAPPING_PATH = FAISS_DIR / "chunks_mapping.pkl"
PROCESSED_FILES_PATH = FAISS_DIR / "processed_files.pkl"
CHAT_HISTORY_PATH = CHAT_HISTORY_DIR / "chat_history.json"

# Backward-compatible aliases used by existing modules
INPUT_FOLDER = UPLOADS_DIR
INDEX_PATH = FAISS_INDEX_PATH
MAPPING_PATH = CHUNKS_MAPPING_PATH
CACHE_TRACKER_PATH = PROCESSED_FILES_PATH
HISTORY_PATH = CHAT_HISTORY_PATH

# Legacy locations kept only for one-time migration
LEGACY_UPLOADS_DIR = BASE_DIR / "data_input"
LEGACY_FAISS_ARTIFACTS = (
    BASE_DIR / "maintenance_index.faiss",
    BASE_DIR / "chunks_mapping.pkl",
    BASE_DIR / "processed_files.pkl",
)
LEGACY_CHAT_HISTORY_PATH = BASE_DIR / "chat_history.json"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL")

# Default tenant for single-org pilot / Streamlit mode. Override per request in SaaS API.
_DEFAULT_ORG = os.getenv(
    "DEFAULT_ORGANIZATION_ID",
    "00000000-0000-4000-8000-000000000001",
)
DEFAULT_ORGANIZATION_ID = uuid.UUID(_DEFAULT_ORG)


def ensure_data_directories() -> None:
    """Create the standard data directories if they do not exist."""
    for directory in (UPLOADS_DIR, FAISS_DIR, CHAT_HISTORY_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _move_file(source: Path, destination: Path, moved: list[str]) -> None:
    if not source.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        logger.info("Skipping legacy file %s; destination already exists.", source.name)
        return
    shutil.move(str(source), str(destination))
    moved.append(f"{source.name} -> {destination.relative_to(BASE_DIR)}")
    logger.info("Migrated %s to %s", source, destination)


def _move_upload_tree(source_dir: Path, destination_dir: Path, moved: list[str]) -> None:
    if not source_dir.is_dir():
        return
    destination_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        if item.name == ".gitkeep":
            continue
        target = destination_dir / item.name
        if item.is_file():
            if target.exists():
                logger.info("Skipping legacy upload %s; destination already exists.", item.name)
                continue
            shutil.move(str(item), str(target))
            moved.append(f"{item.name} -> {target.relative_to(BASE_DIR)}")
            logger.info("Migrated upload %s to %s", item, target)


def migrate_legacy_storage() -> dict[str, list[str]]:
    """
    Move artifacts from legacy locations into ``data/``.

    Legacy layout:
    - ``data_input/`` uploaded documents
    - project-root ``*.faiss`` / ``*.pkl`` vector artifacts
    - project-root ``chat_history.json``
    """
    ensure_data_directories()
    moved_files: list[str] = []

    _move_upload_tree(LEGACY_UPLOADS_DIR, UPLOADS_DIR, moved_files)

    legacy_targets = (
        (LEGACY_FAISS_ARTIFACTS[0], FAISS_INDEX_PATH),
        (LEGACY_FAISS_ARTIFACTS[1], CHUNKS_MAPPING_PATH),
        (LEGACY_FAISS_ARTIFACTS[2], PROCESSED_FILES_PATH),
        (LEGACY_CHAT_HISTORY_PATH, CHAT_HISTORY_PATH),
    )
    for source, destination in legacy_targets:
        _move_file(source, destination, moved_files)

    return {"moved": moved_files}


def initialize_storage() -> dict[str, list[str]]:
    """Ensure directories exist and migrate any legacy files once at startup."""
    ensure_data_directories()
    return migrate_legacy_storage()
