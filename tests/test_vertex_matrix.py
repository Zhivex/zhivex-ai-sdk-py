from copy import deepcopy
from datetime import datetime, timedelta, timezone
from unittest import TestCase
from pathlib import Path
import tempfile

from scripts.run_vertex_matrix import TARGETS, evidence_inputs, validate_report


class VertexMatrixTests(TestCase):
    def test_mistral_text_only_evidence_cannot_certify_expanded_target(self):
        target = TARGETS["mistral-medium"]
        started = datetime.now(timezone.utc) - timedelta(seconds=5)
        report = {
            "schema_version": 1, "provider": "vertex", "model": target.model,
            "location": target.location, "wheel_sha256": "a" * 64,
            "evidence_status": "integration-only",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "operations": {"generation": "passed", "streaming": "passed"},
        }
        self.assertFalse(validate_report(report, target, "a" * 64, started))
        report["operations"].update({"agent-tool": "passed", "vision": "passed"})
        self.assertTrue(validate_report(report, target, "a" * 64, started))

    def test_evidence_inputs_track_helpers_fixtures_and_dependency_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = ("scripts/check.py", "scripts/helpers/fixture.py",
                     "tests/fixtures/vertex/input.png", "pyproject.toml", "uv.lock")
            for name in files:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"initial")
            (root / ".env").write_text("must-not-be-recorded")
            before = evidence_inputs(root)
            self.assertEqual(set(before), set(files))
            for name in files:
                with self.subTest(name=name):
                    path = root / name
                    path.write_bytes(b"changed")
                    self.assertNotEqual(evidence_inputs(root), before)
                    path.write_bytes(b"initial")
            (root / "scripts/new_helper.py").write_text("new")
            self.assertNotEqual(evidence_inputs(root), before)
            (root / "scripts/new_helper.py").unlink()
            (root / files[1]).unlink()
            self.assertNotEqual(evidence_inputs(root), before)

    def test_live_audio_requires_fixture_verification_and_cleanup(self):
        target = TARGETS['live-audio']
        started = datetime.now(timezone.utc) - timedelta(seconds=5)
        report = {
            'schema_version': 1, 'provider': 'vertex', 'model': target.model,
            'location': 'us-central1', 'wheel_sha256': 'a' * 64,
            'evidence_status': 'integration-only',
            'recorded_at': datetime.now(timezone.utc).isoformat(),
            'operations': dict.fromkeys(target.operations, 'passed'),
        }
        self.assertTrue(validate_report(report, target, 'a' * 64, started))
        for operation in ('tts-fixture-transcription', 'audio-to-audio-turn', 'cleanup'):
            with self.subTest(operation=operation):
                bad = deepcopy(report)
                del bad['operations'][operation]
                self.assertFalse(validate_report(bad, target, 'a' * 64, started))

    def test_fail_closed_identity_and_required_operations(self):
        target = TARGETS['gemma']
        started = datetime.now(timezone.utc) - timedelta(seconds=5)
        report = {
            'schema_version': 1, 'provider': 'vertex', 'model': target.model,
            'location': 'global', 'wheel_sha256': 'a' * 64,
            'evidence_status': 'integration-only',
            'recorded_at': datetime.now(timezone.utc).isoformat(),
            'operations': dict.fromkeys(target.operations, 'passed'),
        }
        self.assertTrue(validate_report(report, target, 'a' * 64, started))
        for field, value in (
            ('operations', {}), ('wheel_sha256', 'b' * 64), ('model', 'another-model'),
            ('provider', 'gemini'), ('location', 'us-central1'),
            ('evidence_status', 'certified'), ('recorded_at', started.isoformat().replace('+00:00', '')),
            ('recorded_at', (started - timedelta(seconds=1)).isoformat()),
            ('recorded_at', (started + timedelta(days=1)).isoformat()),
        ):
            with self.subTest(field=field, value=value):
                bad = {**report, field: value}
                self.assertFalse(validate_report(bad, target, 'a' * 64, started))
        bad = deepcopy(report)
        bad['operations']['vision'] = 'not-completed'
        self.assertFalse(validate_report(bad, target, 'a' * 64, started))
        bad = deepcopy(report)
        bad['operations']['extra'] = 'failed'
        self.assertFalse(validate_report(bad, target, 'a' * 64, started))
