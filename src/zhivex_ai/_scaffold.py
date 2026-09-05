"""Private implementation of the Beta application scaffold."""

from __future__ import annotations

import json
import re
import shutil
from importlib.resources import files
from pathlib import Path


def create_project(name: str, *, backend: str, sdk_version: str) -> Path:
    # Only a new immediate child of the caller's directory is accepted.
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", name):
        raise ValueError(
            "Project name must be a single directory name (letters, digits, - or _)."
        )
    if backend not in {"sqlite", "postgres"}:
        raise ValueError("Unknown storage backend.")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[a-z0-9.]+)?", sdk_version):
        raise ValueError("Unsupported package version.")
    destination = Path.cwd() / name
    destination.mkdir(mode=0o700)  # fails for existing directories and symlinks
    try:
        source = (
            files("zhivex_ai").joinpath("templates/durable/app.py.template").read_text()
        )
        (destination / "app.py").write_text(source.replace("__BACKEND__", backend))
        extra = "[postgres]" if backend == "postgres" else ""
        (destination / "requirements.txt").write_text(
            f"zhivex-ai-sdk{extra}=={sdk_version}\n"
        )
        (destination / ".env.example").write_text(
            "OPENAI_API_KEY=\nOPENAI_MODEL=\nDATABASE_URL=\n"
        )
        (destination / ".gitignore").write_text(
            ".venv/\n.env\n*.sqlite3*\n__pycache__/\n"
        )
        (destination / "test_app.py").write_text(
            "import unittest\nfrom app import ProjectInput, lookup\n\nclass AppTests(unittest.TestCase):\n"
            "    def test_lookup_is_read_only(self):\n"
            '        self.assertEqual(lookup(ProjectInput(project="Apollo")), {"project": "Apollo", "status": "on track"})\n'
        )
        (destination / "pyproject.toml").write_text(
            '[tool.ruff]\ntarget-version = "py311"\n[tool.ruff.lint]\nselect = ["E4", "E7", "E9", "F"]\n'
            '[tool.mypy]\npython_version = "3.11"\ncheck_untyped_defs = true\n'
        )
        (destination / "scaffold.json").write_text(
            json.dumps(
                {"schema_version": 1, "sdk_version": sdk_version, "backend": backend}
            )
            + "\n"
        )
        (destination / "README.md").write_text(
            "# Durable agent application\n\n"
            "Create a venv and install `requirements.txt`. Set DATABASE_URL for Postgres. "
            "Environment files are never loaded automatically.\n\n"
            "Run `python app.py health`, `python app.py ready`, `python app.py first`, then `python app.py start`. "
            "Review its run_id and approval_ids. In a NEW process run "
            "`python app.py approve --run-id RUN --approval-id APPROVAL`; `deny` rejects the tool. "
            "`status --run-id RUN` reads durable state; `cancel --run-id RUN` cancels it.\n\n"
            "Offline is the default. Add `--live` to first/start/approve only after setting "
            "OPENAI_API_KEY and OPENAI_MODEL. Output contains status and identifiers only. "
            "Each command has a 60-second bound; model calls have a 15-second timeout, no retries, and bounded output.\n\n"
            "Validation: `python -m unittest -v`, `ruff check app.py test_app.py`, "
            "`mypy app.py test_app.py`. Install these development tools separately.\n\n"
            "This is an application seed, not a hosted service. Before deployment, supply authentication, "
            "tenant isolation, an authorized approval UI, backups, retention and downstream idempotency. "
            "The read-only synthetic tool has no external side effects. Cancellation cannot undo effects. "
            "Health/readiness are CLI probes, not unauthenticated HTTP endpoints. SQLite is local Beta storage; "
            "Postgres is the documented durable production path. Persisted SDK state can contain messages; protect it.\n\n"
            "Upgrade: back up storage, review SDK release notes, update the exact requirements pin in a branch, "
            "run the isolated approval/restart smoke and your own tests, then deploy. Roll back application "
            "and compatible storage together. The generator never overwrites an existing project.\n"
        )
    except BaseException:
        shutil.rmtree(destination)
        raise
    return destination
