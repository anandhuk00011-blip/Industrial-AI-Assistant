"""Smoke tests for the FastAPI SaaS backend."""

from __future__ import annotations

import os
import uuid
import unittest
from pathlib import Path

os.environ["DATABASE_URL"] = ""

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402
from core.tenant import resolve_tenant  # noqa: E402
from database.database import session_scope  # noqa: E402
from database.models import Document, DocumentStatus  # noqa: E402


class FastAPIBackendTests(unittest.TestCase):
    def test_register_me_conversation_and_memory_flow(self) -> None:
        email = f"api-test-{uuid.uuid4().hex[:10]}@example.com"
        with TestClient(app) as client:
            register_response = client.post(
                "/api/auth/register",
                json={
                    "organization_name": "API Regression Factory",
                    "full_name": "Test Engineer",
                    "email": email,
                    "password": "StrongPassword123!",
                },
            )
            self.assertEqual(register_response.status_code, 200, register_response.text)
            token = register_response.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}

            me_response = client.get("/api/me", headers=headers)
            self.assertEqual(me_response.status_code, 200, me_response.text)
            self.assertEqual(me_response.json()["email"], email)

            conversation_response = client.post(
                "/api/conversations",
                headers=headers,
                json={"title": "Pump overheating"},
            )
            self.assertEqual(conversation_response.status_code, 200, conversation_response.text)
            conversation_id = conversation_response.json()["id"]

            memory_response = client.patch(
                f"/api/conversations/{conversation_id}/memory",
                headers=headers,
                json={"memory_lines": ["Pump A overheats after 20 minutes under load."]},
            )
            self.assertEqual(memory_response.status_code, 200, memory_response.text)
            self.assertEqual(
                memory_response.json()["memory"],
                ["Pump A overheats after 20 minutes under load."],
            )

    def test_document_delete_is_tenant_scoped(self) -> None:
        with TestClient(app) as client:
            user_a = self._register(client, "tenant-a")
            user_b = self._register(client, "tenant-b")

            tenant_a = resolve_tenant(user_a["user"]["organization_id"])
            document_path = tenant_a.uploads_dir / f"manual-{uuid.uuid4().hex[:8]}.pdf"
            document_path.write_bytes(b"%PDF- test")
            document_id = uuid.uuid4()

            with session_scope() as session:
                session.add(
                    Document(
                        id=document_id,
                        organization_id=uuid.UUID(user_a["user"]["organization_id"]),
                        file_name=document_path.name,
                        file_path=str(document_path),
                        file_size_bytes=document_path.stat().st_size,
                        mime_type="application/pdf",
                        md5_checksum=uuid.uuid4().hex,
                        status=DocumentStatus.INDEXED.value,
                    )
                )

            other_tenant_response = client.delete(
                f"/api/documents/{document_id}",
                headers={"Authorization": f"Bearer {user_b['access_token']}"},
            )
            self.assertEqual(other_tenant_response.status_code, 404, other_tenant_response.text)
            self.assertTrue(document_path.exists())

            owner_response = client.delete(
                f"/api/documents/{document_id}",
                headers={"Authorization": f"Bearer {user_a['access_token']}"},
            )
            self.assertEqual(owner_response.status_code, 204, owner_response.text)
            self.assertFalse(Path(document_path).exists())

    @staticmethod
    def _register(client: TestClient, label: str) -> dict[str, str]:
        response = client.post(
            "/api/auth/register",
            json={
                "organization_name": f"{label} Factory",
                "full_name": "Test Engineer",
                "email": f"{label}-{uuid.uuid4().hex[:10]}@example.com",
                "password": "StrongPassword123!",
            },
        )
        if response.status_code != 200:
            raise AssertionError(response.text)
        return response.json()


if __name__ == "__main__":
    unittest.main()
