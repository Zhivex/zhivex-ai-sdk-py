"""Validate the generated consumer outside the checkout against one exact wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile


def run(
    command: list[str], *, cwd: Path, env: dict[str, str], success: bool = True
) -> str:
    result = subprocess.run(
        command, cwd=cwd, env=env, capture_output=True, text=True, timeout=180
    )
    if (result.returncode == 0) != success:
        raise RuntimeError(
            f"Consumer command {Path(command[0]).name} failed expectation, exit={result.returncode} (output suppressed)."
        )
    return result.stdout


def verify(wheel: Path, *, backend: str, live: bool) -> dict[str, object]:
    wheel = wheel.resolve(strict=True)
    with zipfile.ZipFile(wheel) as archive:
        metadata = archive.read(
            next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        ).decode()
    version = next(
        line.removeprefix("Version: ")
        for line in metadata.splitlines()
        if line.startswith("Version: ")
    )
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
    if backend == "postgres" and not env.get("DATABASE_URL"):
        raise ValueError("Postgres requested but DATABASE_URL is missing")
    if live and not all(env.get(k) for k in ("OPENAI_API_KEY", "OPENAI_MODEL")):
        raise ValueError("Live mode requires explicit provider configuration")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="zhivex-consumer-") as directory:
        root = Path(directory)
        run([sys.executable, "-m", "venv", str(root / "venv")], cwd=root, env=env)
        python = root / "venv/bin/python"
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                str(wheel) + ("[postgres]" if backend == "postgres" else ""),
            ],
            cwd=root,
            env=env,
        )
        origin = run(
            [str(python), "-I", "-c", "import zhivex_ai; print(zhivex_ai.__file__)"],
            cwd=root,
            env=env,
        ).strip()
        if not Path(origin).resolve().is_relative_to((root / "venv").resolve()):
            raise RuntimeError("Consumer imported the checkout")
        run(
            [
                str(python),
                "-I",
                "-m",
                "zhivex_ai.cli",
                "init",
                "demo",
                "--backend",
                backend,
            ],
            cwd=root,
            env=env,
        )
        project = root / "demo"

        def app(command: str, *args: str, success: bool = True) -> dict[str, object]:
            flags = (
                ["--live"]
                if live and command in {"first", "start", "approve", "deny"}
                else []
            )
            return json.loads(
                run(
                    [str(python), "-I", "app.py", command, *args, *flags],
                    cwd=project,
                    env=env,
                    success=success,
                )
            )

        assert app("health")["status"] == "healthy"
        assert app("ready")["status"] == "ready"
        assert app("first")["status"] == "completed"
        first_seconds = time.perf_counter() - started
        pending = app("start")
        assert pending["status"] == "suspended" and pending["tool_results"] == 0
        run_id = str(pending["run_id"])
        approval = str(pending["approval_ids"][0])  # type: ignore[index]
        app(
            "approve",
            "--run-id",
            run_id,
            "--approval-id",
            "wrong-approval",
            success=False,
        )
        assert app("status", "--run-id", run_id)["status"] == "suspended"
        completed = app("approve", "--run-id", run_id, "--approval-id", approval)
        assert (
            completed["status"] == "completed"
            and completed["tool_results"] == 1
            and completed["checkpoint"]
        )
        assert app("status", "--run-id", run_id)["status"] == "completed"
        app("approve", "--run-id", run_id, "--approval-id", approval, success=False)
        cancel = app("start")
        assert app("cancel", "--run-id", str(cancel["run_id"]))["status"] == "cancelled"
        app(
            "approve",
            "--run-id",
            str(cancel["run_id"]),
            "--approval-id",
            str(cancel["approval_ids"][0]),
            success=False,
        )  # type: ignore[index]
        denied = app("start")
        assert (
            app(
                "deny",
                "--run-id",
                str(denied["run_id"]),
                "--approval-id",
                str(denied["approval_ids"][0]),
            )["status"]
            == "completed"
        )  # type: ignore[index]
        run([str(python), "-m", "unittest", "-v"], cwd=project, env=env)
    return {
        "schema_version": 1,
        "status": "passed",
        "sdk_version": version,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "backend": backend,
        "provider_mode": "live" if live else "deterministic",
        "restart": "separate_processes",
        "first_response_seconds": round(first_seconds, 3),
        "total_seconds": round(time.perf_counter() - started, 3),
        "human_timing": "not_measured",
    }


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Acceptance verification requires Python assertions enabled")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--backend", choices=["sqlite", "postgres"], default="sqlite")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = verify(args.wheel, backend=args.backend, live=args.live)
    except Exception as error:
        report = {
            "schema_version": 1,
            "status": "failed",
            "error_type": type(error).__name__,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    raise SystemExit(0 if report["status"] == "passed" else 1)
