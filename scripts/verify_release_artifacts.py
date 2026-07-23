from __future__ import annotations

import argparse
from email.parser import Parser
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import tomllib
import venv
import zipfile


ROOT = Path(__file__).resolve().parents[1]
EXTRAS_IMPORTS = {
    "postgres": ["asyncpg"],
    "mcp": ["mcp"],
    "api": ["fastapi", "uvicorn"],
    "otel": ["opentelemetry", "opentelemetry.sdk"],
    "docx": ["docx"],
}


def _package_version() -> str:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    return str(pyproject["project"]["version"])


def _canonical_project_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _artifact_version(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "_", value)


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True, env=env)


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _venv_bin(venv_dir: Path, name: str) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / f"{name}.exe"
    return venv_dir / "bin" / name


def _create_venv(path: Path) -> Path:
    builder = venv.EnvBuilder(with_pip=True, clear=True)
    try:
        builder.create(path)
    except subprocess.CalledProcessError:
        uv = shutil.which("uv")
        if uv is None:
            raise
        _run([uv, "venv", "--seed", str(path)])
    return _venv_python(path)


def _select_release_artifact(dist_dir: Path, *, version: str, kind: str) -> Path:
    normalized_version = _artifact_version(version)
    if kind == "wheel":
        candidates = sorted(dist_dir.glob("zhivex_ai_sdk-*.whl"))
        matches = [path for path in candidates if path.name.startswith(f"zhivex_ai_sdk-{normalized_version}-")]
    elif kind == "sdist":
        candidates = sorted(dist_dir.glob("zhivex_ai_sdk-*.tar.gz"))
        matches = [path for path in candidates if path.name == f"zhivex_ai_sdk-{normalized_version}.tar.gz"]
    else:
        raise ValueError(f"Unknown release artifact kind: {kind}")

    if len(candidates) != 1 or len(matches) != 1:
        names = ", ".join(path.name for path in candidates) or "none"
        raise RuntimeError(
            f"Expected exactly one {kind} artifact for version {version} in {dist_dir}; found: {names}. "
            "Remove stale or mismatched artifacts and rebuild the release candidate."
        )
    return matches[0]


def _verify_wheel_metadata(wheel: Path, *, expected_version: str) -> None:
    with zipfile.ZipFile(wheel) as archive:
        metadata_names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise RuntimeError(f"Expected exactly one METADATA file in {wheel.name}; found {len(metadata_names)}.")
        metadata = Parser().parsestr(archive.read(metadata_names[0]).decode("utf-8"))

    project_name = str(metadata.get("Name") or "")
    version = str(metadata.get("Version") or "")
    if _canonical_project_name(project_name) != "zhivex-ai-sdk":
        raise RuntimeError(f'Wheel {wheel.name} has unexpected project name {project_name!r}.')
    if version != expected_version:
        raise RuntimeError(
            f'Wheel {wheel.name} metadata version mismatch: expected {expected_version!r}, received {version!r}.'
        )


def _verify_sdist_metadata(sdist: Path, *, expected_version: str) -> None:
    with tarfile.open(sdist, "r:gz") as archive:
        pyproject_members = [
            member
            for member in archive.getmembers()
            if member.isfile() and len(Path(member.name).parts) == 2 and Path(member.name).name == "pyproject.toml"
        ]
        if len(pyproject_members) != 1:
            raise RuntimeError(
                f"Expected exactly one top-level pyproject.toml in {sdist.name}; found {len(pyproject_members)}."
            )
        stream = archive.extractfile(pyproject_members[0])
        if stream is None:
            raise RuntimeError(f"Could not read pyproject.toml from {sdist.name}.")
        pyproject = tomllib.loads(stream.read().decode("utf-8"))

    project = pyproject.get("project") or {}
    project_name = str(project.get("name") or "")
    version = str(project.get("version") or "")
    if _canonical_project_name(project_name) != "zhivex-ai-sdk":
        raise RuntimeError(f'Sdist {sdist.name} has unexpected project name {project_name!r}.')
    if version != expected_version:
        raise RuntimeError(
            f'Sdist {sdist.name} metadata version mismatch: expected {expected_version!r}, received {version!r}.'
        )


def _install(python: Path, requirement: str) -> None:
    _run([str(python), "-m", "pip", "install", "--disable-pip-version-check", requirement])


def _smoke_code(expected_version: str) -> str:
    return textwrap.dedent(
        f"""
        import asyncio
        from importlib import metadata, resources

        import zhivex_ai
        from zhivex_ai import (
            Agent,
            SequentialAgent,
            WorkflowStep,
            create_mock_language_model,
            create_text_message,
            generate_text,
            run_agent,
            tool,
        )
        from zhivex_ai.types import GenerateResult, ModelMessage, ToolCall, ToolCallPart

        assert metadata.version("zhivex-ai-sdk") == {expected_version!r}
        assert "Agent" in zhivex_ai.__all__
        assert "generate_text" in zhivex_ai.__all__
        assert resources.files("zhivex_ai").joinpath("py.typed").is_file()

        async def main():
            model = create_mock_language_model(responses=[GenerateResult(text="hello", finish_reason="stop")])
            text = await generate_text(model=model, prompt="hello")
            assert text.text == "hello"

            tool_executions = []

            def validate_release(input):
                tool_executions.append(dict(input))
                return {{"nonce": input.get("nonce"), "validated": input.get("nonce") == "artifact-smoke"}}

            agent = Agent(
                name="assistant",
                model=create_mock_language_model(
                    responses=[
                        GenerateResult(
                            messages=[
                                ModelMessage(
                                    role="assistant",
                                    parts=[
                                        ToolCallPart(
                                            tool_call=ToolCall(
                                                id="artifact-tool-call",
                                                name="validate_release",
                                                input={{"nonce": "artifact-smoke"}},
                                            )
                                        )
                                    ],
                                )
                            ]
                        ),
                        GenerateResult(
                            text="agent-tool-ok",
                            messages=[create_text_message("assistant", "agent-tool-ok")],
                            finish_reason="stop",
                        ),
                    ]
                ),
                tools={{
                    "validate_release": tool(
                        name="validate_release",
                        schema=dict,
                        execute=validate_release,
                    )
                }},
            )
            run = await run_agent(agent=agent, prompt="go")
            assert run.text == "agent-tool-ok"
            assert tool_executions == [{{"nonce": "artifact-smoke"}}]
            assert len(run.tool_results) == 1
            assert run.tool_results[0].output == {{"nonce": "artifact-smoke", "validated": True}}

            workflow = SequentialAgent(
                name="release_smoke",
                steps=[
                    WorkflowStep(
                        "step",
                        Agent(
                            name="worker",
                            model=create_mock_language_model(
                                responses=[GenerateResult(text="workflow-ok", finish_reason="stop")]
                            ),
                        ),
                        prompt="go",
                        output_key="out",
                    )
                ],
            )
            result = await workflow.run()
            assert result.state["out"] == "workflow-ok"

        asyncio.run(main())
        """
    )


def _run_base_smoke(python: Path, *, expected_version: str) -> None:
    _run([str(python), "-c", _smoke_code(expected_version)])
    cli = _venv_bin(python.parent.parent, "zhivex-skills")
    _run([str(cli), "--help"])


def _run_extra_smoke(python: Path, extra: str) -> None:
    imports = EXTRAS_IMPORTS[extra]
    code = "\n".join(
        [
            "import importlib",
            "import zhivex_ai",
            *[f'importlib.import_module("{name}")' for name in imports],
            'print("ok")',
        ]
    )
    _run([str(python), "-c", code])


def verify_wheel(wheel: Path, *, extras: list[str], expected_version: str) -> None:
    _verify_wheel_metadata(wheel, expected_version=expected_version)
    with tempfile.TemporaryDirectory(prefix="zhivex-wheel-smoke-") as temp_dir:
        python = _create_venv(Path(temp_dir) / "venv")
        _install(python, str(wheel))
        _run_base_smoke(python, expected_version=expected_version)

    for extra in extras:
        with tempfile.TemporaryDirectory(prefix=f"zhivex-extra-{extra}-") as temp_dir:
            python = _create_venv(Path(temp_dir) / "venv")
            requirement = f"zhivex-ai-sdk[{extra}] @ {wheel.resolve().as_uri()}"
            _install(python, requirement)
            _run_extra_smoke(python, extra)


def verify_sdist(sdist: Path, *, expected_version: str) -> None:
    _verify_sdist_metadata(sdist, expected_version=expected_version)
    with tempfile.TemporaryDirectory(prefix="zhivex-sdist-smoke-") as temp_dir:
        python = _create_venv(Path(temp_dir) / "venv")
        _install(python, str(sdist))
        _run_base_smoke(python, expected_version=expected_version)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify built Zhivex AI SDK release artifacts in fresh venvs.")
    parser.add_argument("--dist-dir", default=str(ROOT / "dist"))
    parser.add_argument("--skip-sdist", action="store_true", help="Skip installing the source distribution.")
    parser.add_argument(
        "--extras",
        default="postgres,mcp,api,otel,docx",
        help="Comma-separated optional extras to install and import-check.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dist_dir = Path(args.dist_dir)
    version = _package_version()
    wheel = _select_release_artifact(dist_dir, version=version, kind="wheel")
    sdist = None if args.skip_sdist else _select_release_artifact(dist_dir, version=version, kind="sdist")
    extras = [item.strip() for item in args.extras.split(",") if item.strip()]
    unknown = sorted(set(extras) - set(EXTRAS_IMPORTS))
    if unknown:
        raise ValueError(f"Unknown extras requested: {', '.join(unknown)}")

    env_hint = os.getenv("ZHIVEX_RELEASE_VERIFY_NETWORK")
    if env_hint:
        print(f"ZHIVEX_RELEASE_VERIFY_NETWORK={env_hint}")

    verify_wheel(wheel, extras=extras, expected_version=version)
    if sdist is not None:
        verify_sdist(sdist, expected_version=version)
    print("Release artifacts verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
