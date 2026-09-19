"""Bounded local live evidence for an explicitly supplied installed wheel.

No raw provider payloads, prompts, credentials or audio are written to evidence.
This is local integration evidence, not protected-workflow release certification.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import zipfile
import wave

from dotenv import load_dotenv
import zhivex_ai
from zhivex_ai import (
    Agent, ApprovalDecision, AudioFrame, RunLimits, ToolDefinition, create_gemini,
    create_openai, create_sqlite_agent_run_store,
)
from zhivex_ai.live import (
    RealtimeConnectOptions, RealtimeSessionConfig, open_websocket_connection,
    stream_live_agent,
)


if __package__:
    from .live_certification_tools import recovery_tool
else:
    from live_certification_tools import recovery_tool


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_code(value):
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.\[\]-]{1,100}", value) else None


def transient_connection_failure(outcome):
    if outcome.get("status") != "failed":
        return False
    if outcome.get("error_type") not in {"ConnectionClosedError", "ValidationError"}:
        return False
    events = outcome.get("events") if isinstance(outcome.get("events"), dict) else {}
    return bool(events.get("error")) and not outcome.get("wire_events") and not outcome.get("diagnostics")


async def probe(provider_name, model_id, operation, resume_model=None, _attempt=1):
    recovery = operation in {"approval-allow-restart", "approval-deny-restart", "approval-concurrent"}
    if recovery and not resume_model:
        return {"status": "blocked", "reason": "missing_resume_model"}
    input_rate = 16000 if provider_name == "gemini" else 24000
    audio_input = None
    if operation == "audio-roundtrip":
        fixture_name = "orbit-seven-16k.wav" if provider_name == "gemini" else "orbit-seven.wav"
        fixture = Path(__file__).resolve().parents[1] / "tests/fixtures/realtime" / fixture_name
        with wave.open(str(fixture)) as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, input_rate):
                raise ValueError("invalid_audio_fixture_format")
            audio_input = wav.readframes(wav.getnframes())
        if not 4800 <= len(audio_input) <= 480000:
            raise ValueError("invalid_audio_fixture_duration")
    wire = Counter()
    diagnostics = []
    connections = []
    executions = []
    audio_bytes = 0
    events = Counter()

    class RecordingConnection:
        def __init__(self, inner):
            self.inner = inner
            self.closed = False

        async def send_json(self, payload):
            await self.inner.send_json(payload)

        async def recv_json(self):
            payload = await self.inner.recv_json()
            if isinstance(payload, dict):
                event_type = safe_code(payload.get("type"))
                if event_type:
                    wire[event_type] += 1
                else:
                    for key in ("serverContent", "toolCall", "setupComplete", "error"):
                        if key in payload:
                            wire[key] += 1
                error = payload.get("error")
                if isinstance(error, dict):
                    diagnostics.append({k: safe_code(error.get(k)) for k in ("code", "type", "status", "param")})
            return payload

        async def close(self):
            try:
                await self.inner.close()
            finally:
                self.closed = True

    async def factory(url, headers, options):
        connection = RecordingConnection(await open_websocket_connection(url, headers=headers, options=options))
        connections.append(connection)
        return connection

    key = os.getenv("OPENAI_API_KEY") if provider_name == "openai" else (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    if not key:
        return {"status": "blocked", "reason": "missing_credentials"}
    provider = (create_openai if provider_name == "openai" else create_gemini)(api_key=key, realtime_connection_factory=factory)
    config = RealtimeSessionConfig(
        provider_options=({"max_output_tokens": 128} if provider_name == "openai" else {
            "generationConfig": {"maxOutputTokens": 256}, "outputAudioTranscription": {},
        }),
    )
    if operation in {"audio-output", "audio-roundtrip", "cancel-active"} or provider_name == "gemini":
        config.output_audio_media_type = "audio/pcm"
        config.output_sample_rate_hz = 24_000
        if provider_name == "openai":
            config.voice = "alloy"
    if audio_input is not None:
        config.input_audio_media_type = "audio/pcm"
        config.input_sample_rate_hz = input_rate
        if provider_name == "openai":
            config.turn_detection = {"type": "none"}
        else:
            config.provider_options["inputAudioTranscription"] = {}
            config.provider_options["realtimeInputConfig"] = {
                "automaticActivityDetection": {"disabled": False, "silenceDurationMs": 500},
            }
    prompt = "Say only: live check passed."
    if operation == "cancel-active":
        prompt = "Count slowly from one to one hundred."
    tools = {}
    approval = None
    if operation in {"tool-loop", "approval-suspend"} or recovery:
        prompt = "Call lookup exactly once with no arguments. Then say its returned marker. Do not guess it."

        def lookup(_input):
            executions.append("lookup")
            return {"marker": "ORBIT_SEVEN"}

        tools = {"lookup": ToolDefinition(
            name="lookup", description="Returns the marker required to answer the user.",
            schema={"type": "object", "properties": {}, "additionalProperties": False},
            execute=lookup, requires_approval=operation == "approval-suspend" or recovery,
        )}
        if operation == "approval-suspend" or recovery:
            def approval(_request):
                return ApprovalDecision.require_human("Synthetic certification approval")
    started = time.monotonic()
    outcome = {}
    with tempfile.TemporaryDirectory(prefix="zhivex-live-probe-") as directory:
        store = create_sqlite_agent_run_store(str(Path(directory) / "runs.sqlite3"))
        if recovery:
            tools = {"lookup": recovery_tool(str(Path(directory) / "effects.log"))}
        agent = Agent(
            name="live-certification", model=provider.realtime_model(model_id),
            instructions="Follow the request exactly and keep output very short.",
            tools=tools, approval_policy=approval, run_store=store,
            run_limits=RunLimits(max_tool_calls=2, max_wall_time_ms=40_000),
        )
        try:
            async with asyncio.timeout(50):
                async with stream_live_agent(
                    agent=agent, prompt=None if operation in {"cancel", "audio-roundtrip"} else prompt,
                    realtime_config=config, connect_options=RealtimeConnectOptions(timeout_ms=10_000),
                    idempotency_key="probe", stream_buffer_size=4096,
                ) as stream:
                    async for event in stream.event_stream():
                        events[event.type] += 1
                        if event.type == "realtime-audio-output":
                            audio_bytes += len(event.audio)
                        if audio_input is not None and event.type == "realtime-start":
                            # Pace microphone-sized frames so provider VAD sees speech and trailing silence.
                            padded_audio = audio_input + bytes(input_rate * 2)
                            chunk_size = input_rate // 5
                            for offset in range(0, len(padded_audio), chunk_size):
                                chunk = padded_audio[offset:offset + chunk_size]
                                await stream.send_audio(AudioFrame(
                                    data=chunk, media_type=f"audio/pcm;rate={input_rate}", sample_rate_hz=input_rate, channels=1,
                                    is_final=offset + chunk_size >= len(padded_audio),
                                ))
                                await asyncio.sleep(0.1)
                        if (operation == "cancel" and event.type == "realtime-start") or (
                            operation == "cancel-active" and event.type == "realtime-audio-output"
                        ):
                            await stream.aclose()
                            break
                    if operation not in {"cancel", "cancel-active"}:
                        result = await stream.collect()
                        outcome["output_nonempty"] = bool(result.text)
                        normalized = re.sub(r"[^A-Z0-9]", "", result.text.upper())
                        outcome["marker_matched"] = "ORBITSEVEN" in normalized or "ORBIT7" in normalized
                state = await store.find_by_idempotency_key("probe")
                expected = "cancelled" if operation in {"cancel", "cancel-active"} else "suspended" if operation == "approval-suspend" or recovery else "completed"
                assert state is not None and state.status == expected, "unexpected_durable_status"
                if operation == "tool-loop":
                    assert len(executions) == 1, "tool_execution_count"
                    assert outcome["marker_matched"], "missing_tool_result_output"
                if operation == "approval-suspend" or recovery:
                    assert not executions and len(state.pending_approvals) == 1, "approval_not_suspended"
                if operation in {"audio-output", "audio-roundtrip", "cancel-active"}:
                    assert audio_bytes > 0, "missing_audio"
                if operation == "audio-roundtrip":
                    assert outcome["marker_matched"], "missing_audio_prompt_answer"
                    outcome["input_audio_sha256"] = digest(audio_input)
                    outcome["input_audio_bytes"] = len(audio_input)
                    outcome["input_sample_rate_hz"] = input_rate
                if operation == "response":
                    assert outcome["output_nonempty"] or audio_bytes > 0, "missing_output"
                assert connections and all(connection.closed for connection in connections), "connection_not_closed"
                if recovery:
                    effects = Path(directory) / "effects.log"
                    helper = Path(__file__).with_name("resume_live_certification.py")
                    async def resume_in_process():
                        process = await asyncio.create_subprocess_exec(
                            sys.executable, str(helper), "--database", str(Path(directory) / "runs.sqlite3"),
                            "--effects", str(effects), "--provider", provider_name, "--model", resume_model,
                            "--approved", "no" if operation == "approval-deny-restart" else "yes",
                            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                        )
                        try:
                            stdout, _stderr = await asyncio.wait_for(process.communicate(), timeout=40)
                            assert process.returncode == 0, "recovery_process_failed"
                            return json.loads(stdout)
                        finally:
                            if process.returncode is None:
                                process.kill()
                                await process.wait()
                    children = await asyncio.gather(*[
                        resume_in_process() for _ in range(2 if operation == "approval-concurrent" else 1)
                    ])
                    executed = len(effects.read_text().splitlines()) if effects.exists() else 0
                    successes = [child for child in children if child["status"] == "passed"]
                    outcome.update(recovery_processes=children, recovered_tool_executions=executed,
                                   resume_model=resume_model)
                    assert len(successes) == 1 and successes[0]["output_nonempty"], "recovery_not_completed_once"
                    assert executed == (0 if operation == "approval-deny-restart" else 1), "recovery_effect_count"
                    if operation == "approval-concurrent":
                        assert sum(child.get("error_type") == "ValidationError" for child in children) == 1, "unexpected_claim_failure"
                    state = await store.find_by_idempotency_key("probe")
                    assert state.status == "completed", "recovery_not_durable"
                outcome.update(status="passed", durable_status=state.status)
        except Exception as error:
            state = await store.find_by_idempotency_key("probe")
            response = getattr(error, "response", None)
            outcome.update(status="failed", error_type=type(error).__name__, http_status=getattr(response, "status_code", None),
                           durable_status=state.status if state else None)
            if isinstance(error, AssertionError):
                outcome["assertion"] = safe_code(str(error))
        outcome.update(duration_ms=round((time.monotonic()-started)*1000), audio_bytes=audio_bytes,
                       tool_executions=len(executions), events=dict(events), wire_events=dict(wire),
                       diagnostics=diagnostics, connections_closed=bool(connections) and all(c.closed for c in connections))
    if transient_connection_failure(outcome) and _attempt < 3:
        await asyncio.sleep(0.5 * _attempt)
        return await probe(provider_name, model_id, operation, resume_model=resume_model, _attempt=_attempt + 1)
    outcome["attempts"] = _attempt
    return outcome


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--provider", choices=["openai", "gemini"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--resume-model")
    parser.add_argument("--operations", default="response,audio-output,audio-roundtrip,tool-loop,approval-suspend,cancel,cancel-active")
    parser.add_argument("--repeat", type=int, choices=range(1, 4), default=1)
    args = parser.parse_args()
    operations = args.operations.split(",")
    if set(operations) - {"response", "audio-output", "audio-roundtrip", "tool-loop", "approval-suspend", "cancel", "cancel-active", "approval-allow-restart", "approval-deny-restart", "approval-concurrent"}:
        parser.error("Unsupported operation")
    load_dotenv(args.env_file, override=False)
    installed = Path(zhivex_ai.__file__).resolve().parent
    with zipfile.ZipFile(args.wheel) as wheel:
        manifest = {name: digest(wheel.read(name)) for name in wheel.namelist() if name.startswith("zhivex_ai/") and name.endswith(".py")}
    assert manifest and all(digest((installed.parent / name).read_bytes()) == value for name,value in manifest.items()), "installed_wheel_mismatch"
    root = Path(__file__).resolve().parents[1]
    report = {
        "schema_version": 1, "kind": "local-live-runtime-evidence", "release_certified": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": args.provider, "requested_model": args.model,
        "wheel_sha256": digest(args.wheel.read_bytes()), "package_version": importlib.metadata.version("zhivex-ai-sdk"),
        "runner_sha256": digest(Path(__file__).read_bytes()),
        "recovery_tool_sha256": digest(Path(__file__).with_name("live_certification_tools.py").read_bytes()),
        "recovery_runner_sha256": digest(Path(__file__).with_name("resume_live_certification.py").read_bytes()), "source_manifest": manifest,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root)),
        "workflow": {key: os.getenv(env) for key, env in {
            "repository": "GITHUB_REPOSITORY", "run_id": "GITHUB_RUN_ID", "sha": "GITHUB_SHA",
            "ref": "GITHUB_REF", "workflow_ref": "GITHUB_WORKFLOW_REF",
        }.items()},
        "results": [],
        "not_tested": [*([] if "audio-roundtrip" in operations else ["audio-input"]), "voice-session-reconnect", "long-session-backpressure", *([] if {"approval-allow-restart", "approval-deny-restart", "approval-concurrent"}.issubset(operations) else ["approval-resume"]), "browser-token", "webrtc"],
    }
    for sample in range(1, args.repeat + 1):
        for operation in operations:
            result = await probe(args.provider, args.model, operation, args.resume_model)
            report["results"].append({"operation": operation, "sample": sample, **result})
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2)+"\n")
            print(args.provider, operation, result["status"], flush=True)
    report["all_requested_passed"] = all(item["status"] == "passed" for item in report["results"])
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    return 0 if report["all_requested_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
