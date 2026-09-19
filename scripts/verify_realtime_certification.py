"""Fail-closed verification of exact-artifact realtime evidence.

Local evidence can satisfy integration checks, never protected-workflow provenance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import zipfile
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OperationEvidence(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")
    operation: str
    sample: int = Field(ge=1, le=3)
    status: Literal["passed", "failed", "blocked"]
    connections_closed: bool | None = None
    durable_status: str | None = None


class RealtimeEvidence(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")
    schema_version: Literal[1]
    kind: Literal["local-live-runtime-evidence"]
    provider: Literal["openai", "gemini"]
    requested_model: str
    created_at: str
    wheel_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    runner_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    recovery_runner_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    recovery_tool_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_manifest: dict[str, str]
    all_requested_passed: bool
    results: list[OperationEvidence]


def schema_text():
    return json.dumps(RealtimeEvidence.model_json_schema(), indent=2, sort_keys=True) + "\n"

OPERATIONS = frozenset({
    "response", "audio-output", "audio-roundtrip", "tool-loop", "approval-suspend",
    "cancel", "cancel-active", "approval-allow-restart", "approval-deny-restart", "approval-concurrent",
})
TARGETS = {
    "openai": ("gpt-realtime-2.1", "gpt-5.6-luna"),
    "gemini": ("gemini-3.8-live", "gemini-3.8-flash"),
}
RUNNERS = {
    "runner_sha256": "certify_live_runtime.py",
    "recovery_runner_sha256": "resume_live_certification.py",
    "recovery_tool_sha256": "live_certification_tools.py",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify(report, *, wheel, root, provider, now=None, required_samples=2, run_id=None, revision=None):
    RealtimeEvidence.model_validate(report)
    def require(condition, code):
        if not condition:
            raise ValueError(code)

    require(report.get("schema_version") == 1, "unsupported_schema")
    require(report.get("kind") == "local-live-runtime-evidence", "unsupported_evidence_kind")
    require(report.get("provider") == provider and provider in TARGETS, "unexpected_provider")
    model, resume_model = TARGETS[provider]
    require(report.get("requested_model") == model, "unexpected_model")
    timestamp = datetime.fromisoformat(report["created_at"])
    now = now or datetime.now(timezone.utc)
    require(timestamp.tzinfo is not None and now - timedelta(days=7) <= timestamp <= now + timedelta(minutes=5), "stale_or_future_evidence")
    require(report.get("wheel_sha256") == digest(wheel.read_bytes()), "wheel_digest_mismatch")
    with zipfile.ZipFile(wheel) as archive:
        manifest = {name: digest(archive.read(name)) for name in archive.namelist()
                    if name.startswith("zhivex_ai/") and name.endswith(".py")}
    require(bool(manifest) and report.get("source_manifest") == manifest, "wheel_manifest_mismatch")
    source_manifest = {str(path.relative_to(root / "src")): digest(path.read_bytes())
                       for path in (root / "src/zhivex_ai").rglob("*.py")}
    require(source_manifest == manifest, "source_mismatch")
    for field, name in RUNNERS.items():
        require(report.get(field) == digest((root / "scripts" / name).read_bytes()), "runner_mismatch")
    require(report.get("all_requested_passed") is True, "incomplete_or_failed_matrix")
    results = report.get("results", [])
    require(len(results) == len(OPERATIONS) * required_samples, "incorrect_matrix_size")
    seen = set()
    for result in results:
        op = result.get("operation")
        sample = result.get("sample")
        require(op in OPERATIONS and type(sample) is int and 1 <= sample <= required_samples, "unknown_operation_or_sample")
        require((op, sample) not in seen, "duplicate_operation_sample")
        seen.add((op, sample))
        require(result.get("status") == "passed" and result.get("connections_closed") is True, "failed_or_open_connection")
        expected = "cancelled" if op in {"cancel", "cancel-active"} else "suspended" if op == "approval-suspend" else "completed"
        require(result.get("durable_status") == expected, "incorrect_durable_status")
        if op in {"audio-output", "audio-roundtrip", "cancel-active"}:
            require(result.get("audio_bytes", 0) > 0, "missing_audio_output")
        if op in {"audio-roundtrip", "tool-loop"}:
            require(result.get("marker_matched") is True, "missing_marker")
        if op == "tool-loop":
            require(result.get("tool_executions") == 1, "tool_effect_count")
        if op.startswith("approval-"):
            require(result.get("tool_executions") == 0, "tool_executed_before_approval")
        if op in {"approval-allow-restart", "approval-deny-restart", "approval-concurrent"}:
            require(result.get("resume_model") == resume_model, "unexpected_resume_model")
            require(result.get("recovered_tool_executions") == (0 if op == "approval-deny-restart" else 1), "recovery_effect_count")
            children = result.get("recovery_processes", [])
            require(len(children) == (2 if op == "approval-concurrent" else 1), "missing_recovery_process")
            require(sum(c.get("status") == "passed" and c.get("output_nonempty") is True for c in children) == 1, "recovery_not_completed")
            if op == "approval-concurrent":
                require(sum(c.get("status") == "failed" and c.get("error_type") == "ValidationError" for c in children) == 1, "unexpected_claim_failure")
        if op == "audio-roundtrip":
            import wave
            fixture = root / "tests/fixtures/realtime" / ("orbit-seven.wav" if provider == "openai" else "orbit-seven-16k.wav")
            with wave.open(str(fixture)) as wav:
                audio = wav.readframes(wav.getnframes())
                rate = wav.getframerate()
            require(result.get("input_audio_sha256") == digest(audio) and result.get("input_audio_bytes") == len(audio)
                    and result.get("input_sample_rate_hz") == rate, "audio_fixture_mismatch")
    if run_id is not None or revision is not None:
        require(bool(run_id) and bool(revision), "missing_workflow_expectation")
        require(report.get("head") == revision and report.get("working_tree_dirty") is False, "unclean_or_wrong_revision")
        require(report.get("workflow") == {
            "repository": "Zhivex/zhivex-ai-sdk-py", "run_id": run_id,
            "sha": revision, "ref": "refs/heads/main",
            "workflow_ref": "Zhivex/zhivex-ai-sdk-py/.github/workflows/realtime-certification.yml@refs/heads/main",
        }, "workflow_provenance_mismatch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--provider", choices=TARGETS, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-id")
    parser.add_argument("--revision")
    args = parser.parse_args()
    try:
        verify(json.loads(args.evidence.read_text()), wheel=args.wheel, root=args.root, provider=args.provider,
               run_id=args.run_id, revision=args.revision)
    except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as error:
        print(f"Realtime certification rejected: {type(error).__name__}")
        return 1
    print("Realtime certification matrix verified" + (" in expected workflow" if args.run_id else " locally"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
