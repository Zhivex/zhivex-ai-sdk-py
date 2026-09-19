"""Bounded Vertex integration checks from an exact installed wheel.

No secrets, prompts, responses, or authenticated URLs are retained. This is
integration evidence, never protected release certification.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

from dotenv import load_dotenv
import zhivex_ai as sdk

ROOT = Path(__file__).resolve().parents[1]


def verify_wheel(wheel: Path) -> str:
    installed = Path(sdk.__file__).resolve().parent
    if ROOT in installed.parents:
        raise SystemExit("Use an installed wheel outside the source checkout.")
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if name.startswith("zhivex_ai/") and name.endswith((".py", ".pyi"))]
        if not names or len(names) != len(set(names)):
            raise SystemExit("Wheel has missing or duplicate package sources.")
        expected = {name.removeprefix("zhivex_ai/") for name in names}
        for directory in (installed, ROOT / "src" / "zhivex_ai"):
            actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file() and path.suffix in {".py", ".pyi"}}
            if actual != expected:
                raise SystemExit("Package source file set does not match the wheel.")
        for name in archive.namelist():
            if name.startswith("zhivex_ai/") and name.endswith((".py", ".pyi")):
                relative = name.removeprefix("zhivex_ai/")
                content = archive.read(name)
                if (installed / relative).read_bytes() != content:
                    raise SystemExit("Installed package does not match the wheel.")
                if (ROOT / "src" / name).read_bytes() != content:
                    raise SystemExit("Source package does not match the wheel.")
    return hashlib.sha256(wheel.read_bytes()).hexdigest()


async def run(args: argparse.Namespace) -> int:
    digest = verify_wheel(args.wheel)
    load_dotenv(ROOT / ".env", override=False)
    os.environ["ZHIVEX_SMOKE_USE_INSTALLED"] = "1"
    sys.path.insert(0, str(ROOT))
    from scripts.run_live_smoke import (
        _run_agent_tool_smoke,
        _run_portable_certification,
        _run_portable_retrieval_certification,
    )
    from zhivex_ai import create_vertex, generate_text
    from zhivex_ai.types import RetryOptions

    credentials = None
    if args.adc:
        import google.auth

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
            quota_project_id=args.project,
        )
    provider = create_vertex(
        location=args.location,
        project_id=args.project,
        credentials=credentials,
        express_mode=args.express,
    )
    model = provider(args.model)
    operations: dict[str, str] = {}
    report = {
        "schema_version": 1,
        "provider": "vertex",
        "model": args.model,
        "mode": "express" if args.express else "standard",
        "location": args.location,
        "adapter": type(model).__name__,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "evidence_status": "integration-only",
        "wheel_sha256": digest,
        "operations": operations,
        "diagnostics": {},
    }
    if args.output.exists():
        raise SystemExit("Choose a new output path; existing evidence is immutable.")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    async def generation() -> None:
        result = await generate_text(
            model=model,
            prompt="Reply with exactly VERTEX_SMOKE_OK.",
            max_tokens=512,
            max_retries=0,
            timeout_ms=20000,
        )
        if result.text.strip().rstrip(".") != "VERTEX_SMOKE_OK":
            raise RuntimeError("generation marker mismatch")

    async def tokens() -> None:
        result = await provider.tokens().count(
            model_id=args.model, prompt="smoke", options=RetryOptions(timeout_ms=20000)
        )
        if not result.total_tokens or result.total_tokens < 1:
            raise RuntimeError("invalid token count")

    async def portable() -> None:
        completed: set[str] = set()
        try:
            await _run_portable_certification(
                provider="vertex", model=model, completed_operations=completed
            )
        finally:
            for operation in ("streaming", "structured-output"):
                operations[operation] = (
                    "passed" if operation in completed else "not-completed"
                )

    checks = [
        ("generation", generation),
        ("portable", portable),
        ("agent-tool", lambda: _run_agent_tool_smoke(provider="vertex", model=model)),
        ("count-tokens", tokens),
        (
            "portable-retrieval",
            lambda: _run_portable_retrieval_certification(
                provider="vertex", model=model
            ),
        ),
    ]
    for name, check in checks:
        try:
            async with asyncio.timeout(90):
                await check()
            operations[name] = "passed"
        except Exception as error:
            operations[name] = "failed"
            report["diagnostics"][name] = {  # type: ignore[index]
                "error_type": type(error).__name__,
                "http_status": getattr(error, "status", None),
            }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{name}: {operations[name]}", flush=True)
    return int(any(state != "passed" for state in operations.values()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--location", default="global")
    parser.add_argument("--express", action="store_true")
    parser.add_argument("--adc", action="store_true")
    parser.add_argument("--project")
    raise SystemExit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()
