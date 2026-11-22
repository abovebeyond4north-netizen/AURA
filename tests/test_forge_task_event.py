"""Tests for the opt-in Skill Forge event exporter; no agent efficacy claim."""
from pathlib import Path
import json
import os
import tempfile
import unittest

from scripts.forge_task_event import emit, read_reviewed_workflow


VALID = {
    "name": "review-regression-evidence",
    "description": "Review completed regression evidence and summarize pass/fail status.",
    "when": "A new task has an actual test report and a request for evidence review.",
    "when_not": "No task result is available or results are unrelated to regression tests.",
    "risk": "none",
    "steps": [
        "Inspect the actual test report and identify the executed version.",
        "Separate observed passes, failures, and untested claims.",
        "Report the evidence and unresolved blockers without inventing results.",
    ],
}


class ExporterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workflow = self.root / "workflow.json"
        self.out = self.root / "logs" / "events.jsonl"
        self.save()

    def save(self, data=None):
        self.workflow.write_text(json.dumps(VALID if data is None else data), encoding="utf-8")

    def test_emits_only_whitelisted_fields(self):
        self.save({**VALID, "prompt": "PRIVATE PROMPT", "tool_outputs": "PRIVATE OUTPUT"})
        self.assertTrue(emit(self.workflow, "run-100", "manual-export", True, 0, self.out))
        content = self.out.read_text(encoding="utf-8")
        self.assertNotIn("PRIVATE", content)
        item = json.loads(content)
        self.assertEqual(set(item), {"event", "task_id", "source", "success", "workflow"})
        self.assertEqual(item["workflow"], VALID)
        self.assertEqual(os.stat(self.out).st_mode & 0o777, 0o600)

    def test_rejects_failed_or_unconfirmed_tasks(self):
        for confirmed, code in [(False, 0), (True, 1), (False, 1)]:
            with self.subTest(confirmed=confirmed, code=code):
                with self.assertRaisesRegex(ValueError, "refusing to claim"):
                    emit(self.workflow, "run-100", "agent-cli", confirmed, code, self.out)
        self.assertFalse(self.out.exists())

    def test_deduplicates_and_refuses_conflicting_identity(self):
        self.assertTrue(emit(self.workflow, "run-100", "agent-cli", True, 0, self.out))
        self.assertFalse(emit(self.workflow, "run-100", "agent-cli", True, 0, self.out))
        self.save({**VALID, "description": "changed after prior export"})
        with self.assertRaisesRegex(ValueError, "different content"):
            emit(self.workflow, "run-100", "agent-cli", True, 0, self.out)
        self.assertEqual(len(self.out.read_text().splitlines()), 1)

    def test_rejects_embedded_credentials(self):
        self.save({**VALID, "description": "api_key=secret-value"})
        with self.assertRaisesRegex(ValueError, "credential"):
            read_reviewed_workflow(self.workflow)

    def test_bad_workflow_fails_closed(self):
        for change in ({"name": "Invalid Name"}, {"risk": "unknown"}, {"steps": ["only one"]}):
            with self.subTest(change=change):
                self.save({**VALID, **change})
                with self.assertRaises(ValueError):
                    emit(self.workflow, "run-101", "manual-export", True, 0, self.out)
        self.assertFalse(self.out.exists())

    def test_rejects_bad_ids_and_overwrite(self):
        with self.assertRaisesRegex(ValueError, "invalid source or task ID"):
            emit(self.workflow, "../bad", "manual-export", True, 0, self.out)
        with self.assertRaisesRegex(ValueError, "output must not overwrite"):
            emit(self.workflow, "run-1", "manual-export", True, 0, self.workflow)


if __name__ == "__main__":
    unittest.main()
