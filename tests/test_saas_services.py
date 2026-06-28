"""Tests for SaaS tenant context and service wiring."""

from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
import uuid
from pathlib import Path


class TenantContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temp_dir.name)
        self.original_sys_path = list(sys.path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()
        sys.path[:] = self.original_sys_path
        for module_name in list(sys.modules):
            if module_name in {"config", "core.tenant", "services.indexing_service", "services.retrieval_service"}:
                sys.modules.pop(module_name, None)

    def _write_config(self, organization_id: str) -> None:
        config_source = Path(__file__).resolve().parents[1] / "config.py"
        contents = config_source.read_text(encoding="utf-8")
        contents = contents.replace(
            '    "00000000-0000-4000-8000-000000000001",',
            f'    "{organization_id}",',
        )
        (self.project_root / "config.py").write_text(contents, encoding="utf-8")

    def _load_tenant_module(self, organization_id: str):
        self._write_config(organization_id)
        for package in ("core", "services", "repositories"):
            (self.project_root / package).mkdir(exist_ok=True)
            init_path = self.project_root / package / "__init__.py"
            if not init_path.exists():
                init_path.write_text("", encoding="utf-8")

        for relative in (
            "core/tenant.py",
            "core/exceptions.py",
            "repositories/vector_repository.py",
            "repositories/document_repository.py",
            "database/database.py",
            "database/models.py",
        ):
            source = Path(__file__).resolve().parents[1] / relative
            target = self.project_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

        sys.path.insert(0, str(self.project_root))
        sys.modules.pop("config", None)
        sys.modules.pop("core.tenant", None)
        config = importlib.import_module("config")
        tenant_module = importlib.import_module("core.tenant")
        tenant_module.BASE_DIR = self.project_root
        tenant_module.TENANTS_ROOT = self.project_root / "data" / "tenants"
        return config, importlib.import_module("core.tenant")

    def test_resolve_tenant_creates_isolated_paths(self) -> None:
        org_id = "11111111-1111-4111-8111-111111111111"
        _, tenant_module = self._load_tenant_module(org_id)
        tenant = tenant_module.resolve_tenant()

        self.assertEqual(tenant.organization_id, uuid.UUID(org_id))
        self.assertTrue(str(tenant.uploads_dir).endswith(f"data\\tenants\\{org_id}\\uploads"))
        self.assertTrue(tenant.uploads_dir.is_dir())
        self.assertTrue(tenant.faiss_dir.is_dir())


if __name__ == "__main__":
    unittest.main()
