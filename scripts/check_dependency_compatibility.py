"""Exercise declared core floors, compatible extra minima, and latest dependencies."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def run(mode: str, python: str, report: Path) -> int:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    status = 1
    versions: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="zhivex-dependency-compat-") as directory:
        work = Path(directory)
        interpreter = work / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        requirements = work / "requirements.txt"
        command = [uv, "pip", "compile", str(ROOT / "pyproject.toml"), "--python-version", python,
                   "--output-file", str(requirements), "--no-header", "--quiet"]
        if mode == "minimum-core":
            from packaging.requirements import Requirement
            pins = []
            for raw in project["dependencies"]:
                requirement = Requirement(raw)
                floors = [part.version for part in requirement.specifier if part.operator == ">="]
                if len(floors) != 1:
                    raise ValueError(f"Expected one declared floor for {requirement.name}")
                pins.append(f"{requirement.name}=={floors[0]}")
            constraints = work / "core-floors.txt"
            constraints.write_text("\n".join(pins) + "\n")
            command += ["--extra", "dev", "--constraint", str(constraints)]
        else:
            command += ["--all-extras"]
        command += ["--resolution", "highest" if mode == "latest" else "lowest-direct"]
        try:
            subprocess.run(command, cwd=ROOT, check=True)
            subprocess.run([uv, "venv", str(work / "venv"), "--python", python], check=True)
            subprocess.run([uv, "pip", "install", "--python", str(interpreter), "-r", str(requirements)], check=True)
            subprocess.run([uv, "pip", "install", "--python", str(interpreter), "--no-deps", "-e", str(ROOT)], check=True)
            output = subprocess.check_output([uv, "pip", "list", "--python", str(interpreter), "--format", "json"], text=True)
            versions = {item["name"]: item["version"] for item in json.loads(output)}
            result = subprocess.run([str(interpreter), "-m", "pytest", "-q"], cwd=ROOT)
            status = result.returncode
        except subprocess.CalledProcessError as error:
            status = error.returncode
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"schema_version": 1, "mode": mode, "python": python,
                                 "status": "passed" if status == 0 else "failed",
                                 "packages": versions}, indent=2, sort_keys=True) + "\n")
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["minimum-core", "minimum-extras", "latest"])
    parser.add_argument("--python", default="3.11")
    parser.add_argument("--report", type=Path, default=ROOT / "dist/dependency-compatibility.json")
    args = parser.parse_args()
    raise SystemExit(run(args.mode, args.python, args.report))
