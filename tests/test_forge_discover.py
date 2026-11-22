"""Tests that AURA's Forge pipeline can only issue proposal-stage commands."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import subprocess

from scripts.forge_discover import PipelineError, pipeline


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.forge = self.base / "trusted-forge"
        (self.forge / "scripts").mkdir(parents=True)
        for name in ("agent_bridge.py", "skill_forge.py"):
            (self.forge / "scripts" / name).write_text("# not executed in mocked unit tests\n")
        self.work = self.base / "work"
        self.events = self.base / "events.jsonl"
        self.events.write_text('{"event":"task.completed"}\n')

    def _responses(self):
        return [subprocess.CompletedProcess([], 0, json.dumps({
            "ingested": 1, "ignored": 0, "promotion_authorized": False}), ""),
            subprocess.CompletedProcess([], 0, json.dumps([{
                "name": "test-skill", "action": "candidate-created-awaiting-evidence"}]), "")]

    @patch("scripts.forge_discover.subprocess.run")
    def test_pipeline_only_issues_bridge_and_observe(self, runner):
        runner.side_effect = self._responses()
        report = pipeline(self.forge, self.work, self.events)
        self.assertEqual(report["proposal_names"], ["test-skill"])
        self.assertEqual(report["promoted"], 0)
        self.assertEqual(report["installed"], 0)
        self.assertEqual(runner.call_count, 2)
        first, second = [a.args[0] for a in runner.call_args_list]
        self.assertEqual(Path(first[1]).name, "agent_bridge.py")
        self.assertEqual(Path(second[1]).name, "skill_forge.py")
        self.assertEqual(second[2], "observe")
        self.assertNotIn("attest", str(first + second))
        self.assertNotIn("auto-promote", str(first + second))
        self.assertNotIn("install", str(first + second))

    @patch("scripts.forge_discover.subprocess.run")
    def test_no_automatic_promotion_on_unexpected_bridge_output(self, runner):
        runner.return_value = subprocess.CompletedProcess([], 0, '{"ingested":1,"promotion_authorized":true}', "")
        with self.assertRaisesRegex(PipelineError, "proposal-only"):
            pipeline(self.forge, self.work, self.events)
        self.assertEqual(runner.call_count, 1)

    @patch("scripts.forge_discover.subprocess.run")
    def test_reject_unexpected_promotion_action(self, runner):
        runner.side_effect = [self._responses()[0], subprocess.CompletedProcess(
            [], 0, '[{"action":"skill-installed","name":"test"}]', "")]
        with self.assertRaisesRegex(PipelineError, "non-proposal"):
            pipeline(self.forge, self.work, self.events)

    @patch("scripts.forge_discover.subprocess.run")
    def test_malformed_candidate_name_is_refused(self, runner):
        runner.side_effect = [self._responses()[0], subprocess.CompletedProcess(
            [], 0, '[{"action":"candidate-created-awaiting-evidence"}]', "")]
        with self.assertRaisesRegex(PipelineError, "non-proposal"):
            pipeline(self.forge, self.work, self.events)

    @patch("scripts.forge_discover.subprocess.run")
    def test_no_events_does_not_launch_scripts(self, runner):
        self.events.unlink()
        with self.assertRaisesRegex(PipelineError, "events must exist"):
            pipeline(self.forge, self.work, self.events)
        runner.assert_not_called()

    @patch("scripts.forge_discover.subprocess.run")
    def test_symlinked_workspace_or_event_file_is_rejected(self, runner):
        link = self.base / "link"
        link.symlink_to(self.base, target_is_directory=True)
        with self.assertRaisesRegex(PipelineError, "symlink"):
            pipeline(self.forge, link, self.events)
        self.events.unlink()
        self.events.symlink_to(self.base / "missing")
        with self.assertRaisesRegex(PipelineError, "symlink"):
            pipeline(self.forge, self.work, self.events)
        runner.assert_not_called()

    @patch("scripts.forge_discover.subprocess.run")
    def test_timeouts_are_fail_closed(self, runner):
        runner.side_effect = subprocess.TimeoutExpired(["python"], 30)
        with self.assertRaisesRegex(PipelineError, "exceeded"):
            pipeline(self.forge, self.work, self.events)

    @patch("scripts.forge_discover.subprocess.run")
    def test_nonzero_exit_is_fail_closed(self, runner):
        runner.return_value = subprocess.CompletedProcess([], 1, "", "internal detail")
        with self.assertRaisesRegex(PipelineError, "failed with exit code"):
            pipeline(self.forge, self.work, self.events)

    @patch("scripts.forge_discover.subprocess.run")
    def test_no_promotion_for_empty_proposal_list(self, runner):
        runner.side_effect = [self._responses()[0], subprocess.CompletedProcess([], 0, "[]", "")]
        report = pipeline(self.forge, self.work, self.events)
        self.assertEqual(report["candidates_proposed"], 0)
        self.assertEqual(report["mode"], "proposal-only")


if __name__ == "__main__":
    unittest.main()
