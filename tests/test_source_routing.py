"""Tests for manual vs fallback answer source tags."""

from __future__ import annotations

import unittest

from services.retrieval_service import RetrievalService


class SourceRoutingTests(unittest.TestCase):
    def test_no_index_returns_safety_fallback_tag(self) -> None:
        service = object.__new__(RetrievalService)
        service.ensure_loaded = lambda: False

        answer, evidence = service.ask("The spindle is overheating after 20 minutes")

        self.assertTrue(answer.startswith("[ANSWER: GENERAL SAFETY GUIDANCE]"))
        self.assertIn("not found in the uploaded manuals", answer)
        self.assertIn("LOTO", answer)
        self.assertEqual(evidence, [])

    def test_manual_answer_tag_is_enforced(self) -> None:
        answer = RetrievalService._ensure_source_tag(
            "[SOURCE: FALLBACK]\nWrong prefix but useful content.",
            "[ANSWER: VERIFIED FROM MANUAL]",
        )

        self.assertTrue(answer.startswith("[ANSWER: VERIFIED FROM MANUAL]"))
        self.assertNotIn("[SOURCE: FALLBACK]", answer)


if __name__ == "__main__":
    unittest.main()
