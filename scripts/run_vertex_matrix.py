"""Reproduce reviewed Vertex integration targets on one installed wheel.

This baseline matrix does not certify the full platform or convert local evidence
into release certification. Individual targets remain inspectable and replayable.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Target:
    script: str
    model: str
    operations: tuple[str, ...]
    arguments: tuple[str, ...] = ()
    location: str = "global"


TARGETS = {
    "mistral-ocr": Target("verify_vertex_mistral_integration.py", "mistral-ocr-2505", ("ocr-document",), ("--model", "mistral-ocr-2505", "--kind", "ocr"), location="us-central1"),
    "codestral-fim": Target("verify_vertex_mistral_integration.py", "codestral-2", ("fim-generation", "fim-streaming"), ("--model", "codestral-2", "--kind", "fim"), location="us-central1"),
    "mistral-small": Target("verify_vertex_chat_integration.py", "mistralai/mistral-small-2503", ("generation", "streaming", "vision"), ("--model", "mistralai/mistral-small-2503", "--location", "us-central1", "--basic", "--vision"), location="us-central1"),
    "codestral": Target("verify_vertex_chat_integration.py", "mistralai/codestral-2", ("generation", "streaming"), ("--model", "mistralai/codestral-2", "--location", "us-central1", "--basic"), location="us-central1"),
    "mistral-medium": Target("verify_vertex_chat_integration.py", "mistralai/mistral-medium-3", ("generation", "streaming", "agent-tool", "vision"), ("--model", "mistralai/mistral-medium-3", "--location", "us-central1", "--vision", "--operation", "generation", "--operation", "streaming", "--operation", "agent-tool", "--operation", "vision"), location="us-central1"),
    "gemini-cache": Target("verify_vertex_native_integration.py", "gemini-3.8-flash", ("cache-create", "cache-get", "cache-update", "cache-generation", "cache-delete", "cache-absent"), ("--model", "gemini-3.8-flash", "--cache-only")),
    "live-audio": Target("verify_vertex_live_integration.py", "gemini-live-2.5-flash-native-audio", ("tts-fixture", "tts-fixture-transcription", "setup", "audio-to-audio-turn", "cleanup"), ("--audio-input", "--location", "us-central1"), location="us-central1"),
    "gemini-core": Target("verify_vertex_integration.py", "gemini-3.8-flash", ("generation", "streaming", "structured-output", "agent-tool", "count-tokens", "portable-retrieval"), ("--adc", "--location", "global", "--model", "gemini-3.8-flash")),
    "gemini-pro": Target("verify_vertex_integration.py", "gemini-3.1-pro-preview", ("generation", "streaming", "structured-output", "agent-tool", "count-tokens", "portable-retrieval"), ("--adc", "--location", "global", "--model", "gemini-3.1-pro-preview")),
    "gemini-pro-customtools": Target("verify_vertex_integration.py", "gemini-3.1-pro-preview-customtools", ("generation", "streaming", "structured-output", "agent-tool", "count-tokens", "portable-retrieval"), ("--adc", "--location", "global", "--model", "gemini-3.1-pro-preview-customtools")),
    "gemini-lite": Target("verify_vertex_integration.py", "gemini-3.5-flash-lite", ("generation", "streaming", "structured-output", "agent-tool", "count-tokens", "portable-retrieval"), ("--adc", "--location", "global", "--model", "gemini-3.5-flash-lite")),
    "gemma": Target("verify_vertex_chat_integration.py", "google/gemma-4-26b-a4b-it-maas", ("generation", "streaming", "structured-output", "agent-tool", "vision"), ("--vision",)),
    "gpt-oss-120b": Target("verify_vertex_chat_integration.py", "openai/gpt-oss-120b-maas", ("generation", "streaming", "structured-output", "agent-tool"), ("--model", "openai/gpt-oss-120b-maas", "--max-tokens", "1024")),
    "gemini-hosted": Target("verify_vertex_hosted_tools_integration.py", "gemini-3.8-flash", ("google-search-citations", "code-execution", "url-context")),
    "gemini-pro-hosted": Target("verify_vertex_hosted_tools_integration.py", "gemini-3.1-pro-preview", ("google-search-citations", "code-execution", "url-context"), ("--model", "gemini-3.1-pro-preview")),
    "gemini-pro-customtools-hosted": Target("verify_vertex_hosted_tools_integration.py", "gemini-3.1-pro-preview-customtools", ("google-search-citations", "code-execution", "url-context"), ("--model", "gemini-3.1-pro-preview-customtools")),
    "gemini-multimodal": Target("verify_vertex_multimodal_integration.py", "gemini-3.8-flash", ("pdf-input", "tts-fixture", "audio-transcription")),
    "gemini-pro-multimodal": Target("verify_vertex_multimodal_integration.py", "gemini-3.1-pro-preview", ("pdf-input", "tts-fixture", "audio-transcription"), ("--model", "gemini-3.1-pro-preview")),
    "gemini-pro-customtools-multimodal": Target("verify_vertex_multimodal_integration.py", "gemini-3.1-pro-preview-customtools", ("pdf-input", "tts-fixture", "audio-transcription"), ("--model", "gemini-3.1-pro-preview-customtools")),
    "gemini-video-input": Target("verify_vertex_video_input_integration.py", "gemini-3.8-flash", ("video-temporal-sequence",)),
    "gemini-pro-video-input": Target("verify_vertex_video_input_integration.py", "gemini-3.1-pro-preview", ("video-temporal-sequence",), ("--model", "gemini-3.1-pro-preview")),
    "gemini-pro-customtools-video-input": Target("verify_vertex_video_input_integration.py", "gemini-3.1-pro-preview-customtools", ("video-temporal-sequence",), ("--model", "gemini-3.1-pro-preview-customtools")),
    "transcribe": Target("verify_vertex_multimodal_integration.py", "gemini-3.5-transcribe-preview", ("tts-fixture", "audio-transcription"), ("--transcription-only", "--model", "gemini-3.5-transcribe-preview")),
    "transcribe-live": Target("verify_vertex_transcribe_live_integration.py", "gemini-3.5-transcribe-live-preview", ("tts-fixture", "setup", "audio-transcription", "cleanup")),
    "translate-live": Target("verify_vertex_translate_live_integration.py", "gemini-3.5-live-translate-preview", ("tts-fixture", "setup", "audio-translation", "cleanup")),
    "omni-video": Target("verify_vertex_omni_integration.py", "gemini-omni-1.1-flash-preview", ("video",)),
    "image-pro": Target("verify_vertex_media_integration.py", "gemini-3-pro-image", ("image-generation",), ("--model", "gemini-3-pro-image", "--kind", "image")),
    "image-pro-edit": Target("verify_vertex_media_integration.py", "gemini-3-pro-image", ("image-edit",), ("--model", "gemini-3-pro-image", "--kind", "image-edit")),
    "image-flash": Target("verify_vertex_media_integration.py", "gemini-3.1-flash-image", ("image-generation",), ("--model", "gemini-3.1-flash-image", "--kind", "image")),
    "image-flash-edit": Target("verify_vertex_media_integration.py", "gemini-3.1-flash-image", ("image-edit",), ("--model", "gemini-3.1-flash-image", "--kind", "image-edit")),
    "image-lite": Target("verify_vertex_media_integration.py", "gemini-3.1-flash-lite-image", ("image-generation",), ("--model", "gemini-3.1-flash-lite-image", "--kind", "image")),
    "image-lite-edit": Target("verify_vertex_media_integration.py", "gemini-3.1-flash-lite-image", ("image-edit",), ("--model", "gemini-3.1-flash-lite-image", "--kind", "image-edit")),
    "embedding-2": Target("verify_vertex_embedding_integration.py", "gemini-embedding-2", ("text", "multiple-inputs", "image", "document", "video", "audio", "combined-text-image"), ("--model", "gemini-embedding-2", "--location", "us", "--image", "--batch", "--multimodal"), location="us"),
}


def validate_report(report: dict, target: Target, digest: str, started_at: datetime) -> bool:
    """Require identity, freshness and every operation; an empty success cannot pass."""
    if not isinstance(report, dict):
        return False
    if report.get("schema_version") != 1 or report.get("provider") != "vertex":
        return False
    if report.get("model") != target.model or report.get("location") != target.location:
        return False
    if report.get("wheel_sha256") != digest or report.get("evidence_status") != "integration-only":
        return False
    try:
        recorded = datetime.fromisoformat(report["recorded_at"])
        if recorded.tzinfo is None or not started_at <= recorded <= datetime.now(timezone.utc):
            return False
    except (KeyError, ValueError, TypeError):
        return False
    operations = report.get("operations")
    return isinstance(operations, dict) and all(operations.get(name) == "passed" for name in target.operations) and all(value == "passed" for value in operations.values())


def evidence_inputs(root: Path) -> dict[str, str]:
    """Hash local check code, fixtures and dependency declarations, never secrets."""
    paths = set((root / "scripts").rglob("*.py"))
    paths.update(path for path in (root / "tests/fixtures/vertex").rglob("*") if path.is_file())
    paths.update(root / name for name in ("pyproject.toml", "uv.lock"))
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", choices=tuple(TARGETS), action="append")
    args = parser.parse_args()
    selected = list(dict.fromkeys(args.target or TARGETS))
    # Never overwrite or resume uncertain dispatches automatically.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    wheel = args.wheel.resolve()
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    controls = {"run_vertex_matrix.py", "verify_vertex_integration.py", *(TARGETS[name].script for name in selected)}
    inputs = evidence_inputs(ROOT)
    summary = {
        "schema_version": 1, "provider": "vertex", "evidence_status": "integration-only",
        "scope": "reviewed-baseline-targets", "wheel_sha256": digest,
        "source_revision": revision, "worktree_dirty": dirty,
        "control_sha256": {name: hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest() for name in sorted(controls)},
        "input_sha256": inputs, "inputs_unchanged": True,
        "selected_targets": selected, "all_baseline_targets_selected": set(selected) == set(TARGETS),
        "recorded_at": datetime.now(timezone.utc).isoformat(), "targets": {},
    }
    summary_path = args.output_dir / "matrix.json"

    def save():
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    def unchanged():
        try:
            return evidence_inputs(ROOT) == inputs
        except OSError:
            return False

    save()
    for name in selected:
        if not unchanged():
            summary["inputs_unchanged"] = False
            save()
            break
        target = TARGETS[name]
        output = (args.output_dir / f"{name}.json").resolve()
        started = datetime.now(timezone.utc)
        summary["targets"][name] = {"status": "pending", "report": output.name}
        save()
        command = [sys.executable, str(ROOT / "scripts" / target.script), "--wheel", str(wheel), "--project", args.project, "--output", str(output), *target.arguments]
        status = "failed"
        try:
            result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), timeout=600, check=False)
            report = json.loads(output.read_text())
            if result.returncode == 0 and validate_report(report, target, digest, started):
                status = "passed"
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
        summary["targets"][name]["status"] = status
        if not unchanged():
            summary["inputs_unchanged"] = False
            summary["targets"][name]["status"] = "invalid-inputs-changed"
        save()
        print(f"{name}: {summary['targets'][name]['status']}", flush=True)
    return int(not summary["inputs_unchanged"] or len(summary["targets"]) != len(selected)
               or any(item["status"] != "passed" for item in summary["targets"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
