"""Bounded GPT-Live exact-wheel certification; receipts contain no session content."""
from __future__ import annotations

import argparse
import asyncio
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import wave
import zipfile

from dotenv import load_dotenv
import zhivex_ai
from zhivex_ai import Agent, RunLimits, ToolDefinition, create_openai, create_sqlite_agent_run_store
from zhivex_ai.live import RealtimeConnectOptions

ROOT = Path(__file__).resolve().parents[1]
OPERATIONS = ("audio-roundtrip", "interruption", "agent-delegation")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fixture_path(operation):
    return ROOT / "tests/fixtures" / (
        "realtime/orbit-seven.wav" if operation == "audio-roundtrip" else
        "gpt_live/interrupt.wav" if operation == "interruption" else "gpt_live/delegation.wav"
    )


def audio_fixture(operation):
    with wave.open(str(fixture_path(operation))) as source:
        assert (source.getnchannels(), source.getsampwidth(), source.getframerate()) == (1, 2, 24000)
        audio = source.readframes(source.getnframes())
    assert 4800 <= len(audio) <= 480000
    return audio


async def probe(operation):
    session = None
    tasks = []
    counts = Counter()
    transcript = []
    input_transcript = []
    audio_bytes = 0
    effects = []
    backend_states = []
    acknowledgments = set()
    interruption_started = asyncio.Event()
    audio_before_interrupt = 0
    delegation_ids = set()
    errors = []
    outcome = {"operation": operation, "status": "failed"}
    pcm = audio_fixture(operation)
    provider = create_openai()
    with tempfile.TemporaryDirectory(prefix="gpt-live-cert-") as directory:
        store = create_sqlite_agent_run_store(str(Path(directory) / "runs.sqlite"))

        def lookup(_):
            effects.append(1)
            return {"code": "ORBIT_SEVEN"}

        backend = Agent(
            name="live-backend", model=provider("gpt-5.6-luna"),
            instructions="Use lookup exactly once to obtain the secret launch code. Return only that code.",
            tools={"lookup": ToolDefinition(name="lookup", description="Look up the secret launch code.",
                   schema={"type": "object", "properties": {}, "additionalProperties": False}, execute=lookup)},
            run_store=store, run_limits=RunLimits(max_steps=3, max_wall_time_ms=25000, max_tool_calls=1),
        )
        try:
            session = await provider.native.live().connect({
                "model": "gpt-live-1", "instructions": (
                    "Speak concisely in English. Always delegate secret launch code lookups to the backend. "
                    "Never invent the code. When the backend returns, speak its code exactly."
                    if operation == "agent-delegation" else
                    "This is a speech repetition test. Follow the user's speaking instructions exactly. "
                    "Repeat requested words directly. Do not delegate speech repetition or counting."
                ),
                "audio": {"format": {"type": "audio/pcm", "rate": 24000}, "output": {"voice": "marin"}},
                "delegation": {"type": "client"},
            }, RealtimeConnectOptions(timeout_ms=10000))
            outcome["startup"] = True

            async def feed():
                nonlocal audio_before_interrupt
                if operation == "interruption":
                    while not interruption_started.is_set():
                        await session.append_audio(bytes(4800))
                        await asyncio.sleep(.1)
                    audio_before_interrupt = audio_bytes
                else:
                    # Establish the continuous audio timeline before the fixture.
                    for _ in range(10):
                        await session.append_audio(bytes(4800))
                        await asyncio.sleep(.1)
                for offset in range(0, len(pcm), 4800):
                    await session.append_audio(pcm[offset:offset + 4800])
                    await asyncio.sleep(.1)
                while True:
                    await session.append_audio(bytes(4800))
                    await asyncio.sleep(.1)

            async def delegate(event):
                result = await session.run_delegation(
                    event, agent=backend, prompt="".join(input_transcript),
                    max_tokens=128, max_retries=0, timeout_ms=20000,
                )
                backend_states.append(result.state.status if result.state else None)

            tasks.append(asyncio.create_task(feed()))
            if operation == "interruption":
                await session.append_context(
                    "Start counting aloud from one to one hundred now, slowly, without stopping until the user interrupts.",
                    kind="instructions", event_id="start-counting",
                )
            async with asyncio.timeout(45):
                while True:
                    for task in tasks:
                        if task.done():
                            task.result()
                    event = await session.receive()
                    kind = event.get("type")
                    counts[kind] += 1
                    if kind == "error":
                        code = event.get("error", {}).get("code")
                        errors.append(code if isinstance(code, str) and code.replace("_", "").isalnum() else "provider_error")
                        raise AssertionError("provider_error")
                    if kind == "session.input_transcript.delta":
                        input_transcript.append(event.get("delta", ""))
                    if kind == "session.output_transcript.delta":
                        transcript.append(event.get("delta", ""))
                    if kind == "session.output_audio.delta":
                        audio_bytes += len(base64.b64decode(event["delta"], validate=True))
                        if operation == "interruption" and not interruption_started.is_set():
                            transcript.clear()
                            interruption_started.set()
                    if kind == "session.commentary.appended":
                        acknowledgments.add(event.get("client_event_id"))
                    if kind == "session.delegation.created" and operation == "agent-delegation":
                        identifier = event.get("delegation", {}).get("id")
                        if identifier not in delegation_ids:
                            assert not delegation_ids, "unexpected_second_delegation"
                            delegation_ids.add(identifier)
                            tasks.append(asyncio.create_task(delegate(event)))
                    normalized = "".join(c for c in "".join(transcript).lower() if c.isalnum())
                    marker = ("lunarnine" in normalized or "lunar9" in normalized) if operation == "interruption" else ("orbitseven" in normalized or "orbit7" in normalized)
                    backend_ok = operation != "agent-delegation" or (
                        backend_states == ["completed"] and len(effects) == 1
                        and all(f"result-{identifier}" in acknowledgments for identifier in delegation_ids)
                    )
                    if marker and audio_bytes > audio_before_interrupt and backend_ok:
                        outcome.update(marker_matched=True, status="passed")
                        break
        except Exception as error:
            outcome.update(status="failed", error_type=type(error).__name__)
            if isinstance(error, AssertionError):
                outcome["assertion"] = str(error)[:100]
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if session:
                try:
                    final = await session.finish(timeout_ms=10000)
                    outcome["finalized"] = final.get("type") == "session.closed"
                    outcome["usage_present"] = "usage" in final
                except Exception as error:
                    outcome.update(status="failed", finalization_error=type(error).__name__)
                outcome["connection_closed"] = session.closed
            outcome.update(
                audio_bytes=audio_bytes, audio_before_interrupt=audio_before_interrupt,
                events=dict(counts), provider_errors=errors, tool_executions=len(effects),
                backend_states=backend_states, delegation_count=len(delegation_ids),
                input_transcript_received=bool(input_transcript), input_audio_sha256=digest(pcm),
            )
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path("/dev/null"))
    parser.add_argument("--repeat", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    load_dotenv(args.env_file)
    with zipfile.ZipFile(args.wheel) as archive:
        manifest = {name: digest(archive.read(name)) for name in archive.namelist()
                    if name.startswith("zhivex_ai/") and name.endswith(".py")}
    installed = Path(zhivex_ai.__file__).resolve().parent.parent
    assert manifest and all(digest((installed / name).read_bytes()) == sha for name, sha in manifest.items()), "installed_wheel_mismatch"
    report = {
        "schema_version": 1, "provider": "openai", "model": "gpt-live-1", "backend_model": "gpt-5.6-luna",
        "created_at": datetime.now(timezone.utc).isoformat(), "wheel_sha256": digest(args.wheel.read_bytes()),
        "source_manifest": manifest, "runner_sha256": digest(Path(__file__).read_bytes()),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "workflow": {k: os.getenv(v) for k, v in {"repository": "GITHUB_REPOSITORY", "run_id": "GITHUB_RUN_ID", "attempt": "GITHUB_RUN_ATTEMPT", "sha": "GITHUB_SHA", "ref": "GITHUB_REF", "workflow_ref": "GITHUB_WORKFLOW_REF"}.items()},
        "results": [],
    }
    for sample in range(1, args.repeat + 1):
        for operation in OPERATIONS:
            result = asyncio.run(probe(operation))
            report["results"].append({**result, "sample": sample})
            print(operation, sample, result["status"], flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["all_passed"] = all(r["status"] == "passed" and r.get("finalized") for r in report["results"])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
