"""Verify full GPT-Live matrices against the exact wheel, source and workflow."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import zipfile

OPERATIONS = {"audio-roundtrip", "interruption", "agent-delegation"}
FIXTURES = {
    "audio-roundtrip": "realtime/orbit-seven.wav",
    "interruption": "gpt_live/interrupt.wav",
    "agent-delegation": "gpt_live/delegation.wav",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify(report, *, wheel, root, run_id=None, revision=None, attempt=None):
    import wave

    def require(condition, code):
        if not condition:
            raise ValueError(code)

    require(report.get('schema_version') == 1 and report.get('provider') == 'openai'
            and report.get('model') == 'gpt-live-1' and report.get('backend_model') == 'gpt-5.6-luna', 'wrong_target')
    timestamp = datetime.fromisoformat(report['created_at'])
    now = datetime.now(timezone.utc)
    require(timestamp.tzinfo is not None and now - timedelta(days=7) <= timestamp <= now + timedelta(minutes=5), 'stale_evidence')
    require(report.get('wheel_sha256') == digest(wheel.read_bytes()), 'wheel_mismatch')
    with zipfile.ZipFile(wheel) as archive:
        manifest = {name: digest(archive.read(name)) for name in archive.namelist()
                    if name.startswith('zhivex_ai/') and name.endswith('.py')}
    source = {str(p.relative_to(root / 'src')): digest(p.read_bytes()) for p in (root / 'src/zhivex_ai').rglob('*.py')}
    require(bool(manifest) and source == manifest == report.get('source_manifest'), 'source_mismatch')
    require(report.get('runner_sha256') == digest((root / 'scripts/certify_gpt_live.py').read_bytes()), 'runner_mismatch')
    require(report.get('all_passed') is True, 'failed_matrix')
    results = report.get('results', [])
    require(len(results) == 6, 'incomplete_matrix')
    seen = set()
    for result in results:
        operation, sample = result.get('operation'), result.get('sample')
        require(operation in OPERATIONS and type(sample) is int and sample in (1, 2), 'unknown_case')
        require((operation, sample) not in seen, 'duplicate_case')
        seen.add((operation, sample))
        require(all(result.get(key) is True for key in ('startup', 'finalized', 'usage_present', 'connection_closed', 'marker_matched', 'input_transcript_received'))
                and result.get('status') == 'passed', 'failed_case')
        require(result.get('provider_errors') == [] and result.get('audio_bytes', 0) > 0, 'missing_audio_or_provider_error')
        with wave.open(str(root / 'tests/fixtures' / FIXTURES[operation])) as audio:
            require((audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, 24000), 'fixture_format')
            pcm = audio.readframes(audio.getnframes())
        require(result.get('input_audio_sha256') == digest(pcm), 'fixture_mismatch')
        if operation == 'interruption':
            require(0 < result.get('audio_before_interrupt', 0) < result['audio_bytes'], 'missing_interruption')
        if operation == 'agent-delegation':
            require(result.get('backend_states') == ['completed'] and result.get('tool_executions') == 1
                    and result.get('delegation_count') == 1
                    and result.get('events', {}).get('session.commentary.appended', 0) > 0, 'delegation_incomplete')
    if any(value is not None for value in (run_id, revision, attempt)):
        require(all((run_id, revision, attempt)), 'missing_workflow_identity')
        require(report.get('head') == revision and report.get('working_tree_dirty') is False, 'unclean_revision')
        require(report.get('workflow') == {
            'repository': 'Zhivex/zhivex-ai-sdk-py', 'run_id': run_id, 'attempt': attempt,
            'sha': revision, 'ref': 'refs/heads/main',
            'workflow_ref': 'Zhivex/zhivex-ai-sdk-py/.github/workflows/realtime-certification.yml@refs/heads/main',
        }, 'workflow_mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--run-id')
    parser.add_argument('--revision')
    parser.add_argument('--attempt')
    args = parser.parse_args()
    try:
        verify(json.loads(args.evidence.read_text()), wheel=args.wheel, root=args.root,
               run_id=args.run_id, revision=args.revision, attempt=args.attempt)
    except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as error:
        print(f'GPT-Live certification rejected: {type(error).__name__}')
        return 1
    print('GPT-Live complete matrix verified' + (' in expected workflow' if args.run_id else ' locally'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
