"""Tests for centralized storage path configuration."""

from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path


class StoragePathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temp_dir.name)
        self.original_sys_path = list(sys.path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()
        sys.path[:] = self.original_sys_path
        for module_name in ("config", "main", "query", "app"):
            sys.modules.pop(module_name, None)

    def _load_config_in_isolated_root(self) -> object:
        config_source = Path(__file__).resolve().parents[1] / "config.py"
        (self.project_root / "config.py").write_text(
            config_source.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        sys.path.insert(0, str(self.project_root))
        sys.modules.pop("config", None)
        return importlib.import_module("config")

    def test_ensure_data_directories_creates_expected_layout(self) -> None:
        config = self._load_config_in_isolated_root()
        config.ensure_data_directories()

        self.assertTrue((self.project_root / "data" / "uploads").is_dir())
        self.assertTrue((self.project_root / "data" / "faiss").is_dir())
        self.assertTrue((self.project_root / "data" / "chat_history").is_dir())

    def test_migrate_legacy_storage_moves_artifacts(self) -> None:
        config = self._load_config_in_isolated_root()

        legacy_uploads = self.project_root / "data_input"
        legacy_uploads.mkdir(parents=True)
        (legacy_uploads / "manual.pdf").write_bytes(b"pdf")

        (self.project_root / "maintenance_index.faiss").write_bytes(b"index")
        (self.project_root / "chunks_mapping.pkl").write_bytes(b"mapping")
        (self.project_root / "processed_files.pkl").write_bytes(b"cache")
        (self.project_root / "chat_history.json").write_text("[]", encoding="utf-8")

        result = config.migrate_legacy_storage()

        moved = result["moved"]
        self.assertTrue(any("manual.pdf" in entry for entry in moved))
        self.assertTrue((self.project_root / "data" / "uploads" / "manual.pdf").is_file())
        self.assertTrue((self.project_root / "data" / "faiss" / "maintenance_index.faiss").is_file())
        self.assertTrue((self.project_root / "data" / "faiss" / "chunks_mapping.pkl").is_file())
        self.assertTrue((self.project_root / "data" / "faiss" / "processed_files.pkl").is_file())
        self.assertTrue((self.project_root / "data" / "chat_history" / "chat_history.json").is_file())
        self.assertFalse((self.project_root / "maintenance_index.faiss").exists())

    def test_migrate_legacy_storage_does_not_overwrite_existing_targets(self) -> None:
        config = self._load_config_in_isolated_root()
        config.ensure_data_directories()

        existing = self.project_root / "data" / "faiss" / "maintenance_index.faiss"
        existing.write_bytes(b"existing")
        (self.project_root / "maintenance_index.faiss").write_bytes(b"legacy")

        result = config.migrate_legacy_storage()

        self.assertEqual(result["moved"], [])
        self.assertEqual(existing.read_bytes(), b"existing")


if __name__ == "__main__":
    unittest.main()
