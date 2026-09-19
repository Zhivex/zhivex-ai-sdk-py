from copy import deepcopy
from datetime import datetime, timezone
import tempfile
from pathlib import Path
from unittest import TestCase
import wave
import zipfile

from scripts.verify_gpt_live_certification import FIXTURES, digest, verify


class GPTLiveCertificationTests(TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / 'src/zhivex_ai/__init__.py'
        source.parent.mkdir(parents=True)
        source.write_text('# package')
        runner = self.root / 'scripts/certify_gpt_live.py'
        runner.parent.mkdir()
        runner.write_text('# runner')
        self.wheel = self.root / 'candidate.whl'
        with zipfile.ZipFile(self.wheel, 'w') as archive:
            archive.writestr('zhivex_ai/__init__.py', source.read_bytes())
        for fixture in FIXTURES.values():
            path = self.root / 'tests/fixtures' / fixture
            path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(path), 'wb') as audio:
                audio.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
                audio.writeframes(bytes(4800))
        self.report = {
            'schema_version': 1, 'provider': 'openai', 'model': 'gpt-live-1', 'backend_model': 'gpt-5.6-luna',
            'created_at': datetime.now(timezone.utc).isoformat(), 'wheel_sha256': digest(self.wheel.read_bytes()),
            'source_manifest': {'zhivex_ai/__init__.py': digest(source.read_bytes())},
            'runner_sha256': digest(runner.read_bytes()), 'all_passed': True, 'results': [],
        }
        for sample in (1, 2):
            for op in FIXTURES:
                self.report['results'].append({
                    'operation': op, 'sample': sample, 'status': 'passed',
                    **dict.fromkeys(('startup', 'finalized', 'usage_present', 'connection_closed', 'marker_matched', 'input_transcript_received'), True),
                    'audio_bytes': 100, 'audio_before_interrupt': 20, 'provider_errors': [],
                    'input_audio_sha256': digest(bytes(4800)), 'backend_states': ['completed'],
                    'tool_executions': 1, 'delegation_count': 1, 'events': {'session.commentary.appended': 1},
                })

    def check(self, report=None, **kwargs):
        verify(report or self.report, wheel=self.wheel, root=self.root, **kwargs)

    def test_complete_local_matrix(self):
        self.check()

    def test_rejects_incomplete_duplicate_or_unverified_cases(self):
        for field, value in (('marker_matched', False), ('finalized', False), ('usage_present', False),
                             ('connection_closed', False), ('audio_bytes', 0), ('input_audio_sha256', 'bad')):
            with self.subTest(field=field):
                report = deepcopy(self.report)
                report['results'][0][field] = value
                with self.assertRaises(ValueError):
                    self.check(report)
        report = deepcopy(self.report)
        report['results'][-1] = report['results'][0]
        with self.assertRaises(ValueError):
            self.check(report)
        report['results'].pop()
        with self.assertRaises(ValueError):
            self.check(report)

    def test_rejects_wrong_source_runner_and_fixture(self):
        for path in ('src/zhivex_ai/extra.py', 'scripts/certify_gpt_live.py'):
            with self.subTest(path=path):
                target = self.root / path
                previous = target.read_bytes() if target.exists() else None
                target.write_text('changed')
                with self.assertRaises(ValueError):
                    self.check()
                if previous is None:
                    target.unlink()
                else:
                    target.write_bytes(previous)

    def test_requires_protected_workflow_and_attempt(self):
        with self.assertRaises(ValueError):
            self.check(run_id='1', revision='sha', attempt='2')
        self.report.update(head='sha', working_tree_dirty=False, workflow={
            'repository': 'Zhivex/zhivex-ai-sdk-py', 'run_id': '1', 'attempt': '2', 'sha': 'sha',
            'ref': 'refs/heads/main',
            'workflow_ref': 'Zhivex/zhivex-ai-sdk-py/.github/workflows/realtime-certification.yml@refs/heads/main',
        })
        self.check(run_id='1', revision='sha', attempt='2')
        with self.assertRaises(ValueError):
            self.check(run_id='1', revision='sha', attempt='3')
