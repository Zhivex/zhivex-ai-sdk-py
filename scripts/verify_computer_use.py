"""Run offline native computer contracts against an isolated installed wheel."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve(strict=True)
    source = Path(__file__).resolve().parents[1]
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    with tempfile.TemporaryDirectory(prefix="zhivex-computer-wheel-") as temporary:
        root = Path(temporary)
        subprocess.run([sys.executable, "-m", "venv", str(root / "venv")], check=True, env=env)
        python = root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(["uv", "pip", "install", "--python", str(python), str(wheel),
                        f"pytest=={version('pytest')}", f"pytest-asyncio=={version('pytest-asyncio')}"],
                       check=True, env=env, cwd=root)
        origin = subprocess.check_output([str(python), "-I", "-c", "import zhivex_ai; print(zhivex_ai.__file__)"], text=True, env=env, cwd=root).strip()
        if not Path(origin).resolve().is_relative_to((root / "venv").resolve()):
            raise RuntimeError("Wheel consumer imported source instead of the installed distribution.")
        shutil.copyfile(source / "tests/test_computer_use.py", root / "test_computer_use.py")
        subprocess.run([str(python), "-I", "-m", "pytest", "test_computer_use.py", "-q"], check=True, env=env, cwd=root)
        evidence = {"wheel": wheel.name, "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(), "import_origin": origin, "offline_computer_contracts": "passed"}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(evidence, indent=2) + "\n")
        print(json.dumps(evidence))


if __name__ == "__main__":
    main()
