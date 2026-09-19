from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import patch
import wave
import zipfile

from scripts.verify_realtime_certification import OPERATIONS, RUNNERS, digest, schema_text, verify


class RealtimeCertificationGateTests(TestCase):
    def test_published_schema_matches_the_verifier(self):
        schema = Path(__file__).resolve().parents[1] / "docs/releases/realtime-certification.schema.json"
        self.assertEqual(schema.read_text(), schema_text())

    def test_schema_canonicalizes_only_redundant_literal_enums(self):
        generated = {"properties": {
            "kind": {"const": "evidence", "enum": ["evidence"]},
            "status": {"enum": ["passed", "failed"]},
            "other": {"const": "a", "enum": ["a", "b"]},
        }}
        with patch("scripts.verify_realtime_certification.RealtimeEvidence.model_json_schema", return_value=generated):
            result = json.loads(schema_text())
        self.assertEqual(result["properties"]["kind"], {"const": "evidence"})
        self.assertEqual(result["properties"]["status"], generated["properties"]["status"])
        self.assertEqual(result["properties"]["other"], generated["properties"]["other"])
        self.assertIn("enum", generated["properties"]["kind"])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.wheel = self.root / "candidate.whl"
        with zipfile.ZipFile(self.wheel, "w") as archive:
            archive.writestr("zhivex_ai/__init__.py", "# synthetic package")
        source = self.root / "src/zhivex_ai/__init__.py"
        source.parent.mkdir(parents=True)
        source.write_text("# synthetic package")
        (self.root / "scripts").mkdir()
        for name in RUNNERS.values():
            (self.root / "scripts" / name).write_text("# synthetic runner")
        fixture = self.root / "tests/fixtures/realtime/orbit-seven.wav"
        fixture.parent.mkdir(parents=True)
        self.audio = bytes(4800)
        with wave.open(str(fixture), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(self.audio)
        self.now = datetime.now(timezone.utc)
        self.report = {
            "schema_version": 1, "kind": "local-live-runtime-evidence", "provider": "openai",
            "requested_model": "gpt-realtime-2.1", "created_at": self.now.isoformat(),
            "wheel_sha256": digest(self.wheel.read_bytes()),
            "source_manifest": {"zhivex_ai/__init__.py": digest(source.read_bytes())},
            **{field: digest((self.root / "scripts" / name).read_bytes()) for field, name in RUNNERS.items()},
            "all_requested_passed": True, "results": [], "head": "revision", "working_tree_dirty": False,
        }
        for sample in (1, 2):
            for op in sorted(OPERATIONS):
                self.report["results"].append({
                    "operation": op, "sample": sample, "status": "passed", "connections_closed": True,
                    "durable_status": "cancelled" if op in {"cancel", "cancel-active"} else "suspended" if op == "approval-suspend" else "completed",
                    "audio_bytes": 2, "marker_matched": True,
                    "tool_executions": 1 if op == "tool-loop" else 0,
                    "resume_model": "gpt-5.6-luna", "recovered_tool_executions": 0 if op == "approval-deny-restart" else 1,
                    "recovery_processes": [{"status": "passed", "output_nonempty": True}]
                    + ([{"status": "failed", "error_type": "ValidationError"}] if op == "approval-concurrent" else []),
                    "input_audio_sha256": digest(self.audio), "input_audio_bytes": len(self.audio), "input_sample_rate_hz": 24000,
                })

    def check(self, report=None, **kwargs):
        verify(report or self.report, wheel=self.wheel, root=self.root, provider="openai", now=self.now, **kwargs)

    def test_complete_local_matrix_is_accepted_without_workflow_claim(self):
        self.check()
        with self.assertRaisesRegex(ValueError, "workflow_provenance_mismatch"):
            self.check(run_id="123", revision="revision")

    def test_expected_workflow_identity_and_clean_source_are_required(self):
        self.report["workflow"] = {
            "repository": "Zhivex/zhivex-ai-sdk-py", "run_id": "123", "sha": "revision", "ref": "refs/heads/main",
            "workflow_ref": "Zhivex/zhivex-ai-sdk-py/.github/workflows/realtime-certification.yml@refs/heads/main",
        }
        self.check(run_id="123", revision="revision")
        for field in ("repository", "run_id", "sha", "ref", "workflow_ref"):
            report = deepcopy(self.report)
            report["workflow"][field] = "wrong"
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(report, run_id="123", revision="revision")
        self.report["working_tree_dirty"] = True
        with self.assertRaisesRegex(ValueError, "unclean_or_wrong_revision"):
            self.check(run_id="123", revision="revision")

    def test_mutated_evidence_is_rejected(self):
        mutations = [
            ("wheel_sha256", "wrong"), ("runner_sha256", "wrong"), ("recovery_tool_sha256", "wrong"),
            ("recovery_runner_sha256", "wrong"), ("source_manifest", {}), ("requested_model", "other"),
            ("all_requested_passed", False), ("schema_version", 2),
            ("created_at", (self.now - timedelta(days=8)).isoformat()),
            ("created_at", (self.now + timedelta(hours=1)).isoformat()),
        ]
        for key, value in mutations:
            report = deepcopy(self.report)
            report[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.check(report)

    def test_each_operation_must_pass_and_duplicates_cannot_fill_gaps(self):
        for index in range(len(self.report["results"])):
            report = deepcopy(self.report)
            report["results"][index]["status"] = "blocked"
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.check(report)
        for key, value in (("connections_closed", False), ("recovered_tool_executions", 2), ("recovery_processes", [])):
            report = deepcopy(self.report)
            result = next(r for r in report["results"] if r["operation"] == "approval-concurrent")
            result[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check(report)
        report = deepcopy(self.report)
        report["results"][-1] = report["results"][0]
        with self.assertRaises(ValueError):
            self.check(report)
        report["results"].pop()
        with self.assertRaises(ValueError):
            self.check(report)

    def test_source_changed_after_wheel_build_is_rejected(self):
        (self.root / "src/zhivex_ai/__init__.py").write_text("# changed")
        with self.assertRaisesRegex(ValueError, "source_mismatch"):
            self.check()

    def test_new_source_missing_from_wheel_is_rejected(self):
        (self.root / "src/zhivex_ai/new_module.py").write_text("# unpackaged source")
        with self.assertRaisesRegex(ValueError, "source_mismatch"):
            self.check()

    def test_saved_reports_are_json_roundtrippable(self):
        self.check(json.loads(json.dumps(self.report)))
