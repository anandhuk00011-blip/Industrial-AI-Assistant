"""Architecture guard for the Streamlit frontend boundary."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


class StreamlitFrontendBoundaryTests(unittest.TestCase):
    def test_app_uses_fastapi_client_not_backend_services(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        tree = ast.parse(app_path.read_text(encoding="utf-8"))
        forbidden_roots = {"main", "query", "database", "services", "repositories", "core"}
        imports: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", 1)[0])

        self.assertTrue(
            imports.isdisjoint(forbidden_roots),
            f"Streamlit app must call FastAPI instead of importing backend modules: {imports & forbidden_roots}",
        )


if __name__ == "__main__":
    unittest.main()
