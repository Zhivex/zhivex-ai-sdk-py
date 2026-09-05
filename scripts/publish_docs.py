"""Preserve older documentation versions and publish the pinned PyPI wheel site."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def publish(push: bool, output: Path | None = None) -> None:
    remote = subprocess.check_output(
        ["git", "remote", "get-url", "origin"], cwd=ROOT, text=True
    ).strip()
    exists = subprocess.check_output(
        ["git", "ls-remote", "--heads", "origin", "gh-pages"], cwd=ROOT, text=True
    ).strip()
    with tempfile.TemporaryDirectory(prefix="zhivex-docs-publish-") as temporary:
        site = Path(temporary) / "site"
        if exists:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--single-branch",
                    "--branch",
                    "gh-pages",
                    remote,
                    str(site),
                ],
                check=True,
            )
        else:
            site.mkdir()
            subprocess.run(["git", "init", "-b", "gh-pages", str(site)], check=True)
            subprocess.run(
                ["git", "remote", "add", "origin", remote], check=True, cwd=site
            )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/build_docs.py"),
                "--output",
                str(site),
            ],
            check=True,
            cwd=ROOT,
        )
        if output is not None:
            shutil.copytree(site, output, ignore=shutil.ignore_patterns(".git"))
        version = json.loads((ROOT / "docs/site/published.json").read_text())["version"]
        # Stage only the newly rendered released version and the version index.
        subprocess.run(
            ["git", "add", version, "index.html", "versions.json", ".nojekyll"],
            check=True,
            cwd=site,
        )
        changed = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=site
        ).returncode
        if changed == 0:
            print("Published documentation bytes are unchanged")
            return
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Zhivex Docs",
                "-c",
                "user.email=docs@users.noreply.github.com",
                "commit",
                "-m",
                f"Document published Python SDK {version}",
            ],
            check=True,
            cwd=site,
        )
        if push:
            push_env = os.environ.copy()
            if push_env.get("GH_TOKEN"):
                # Scope the ephemeral Actions credential to this GitHub push.
                # Never persist it in the archive or include it in command output.
                credential = base64.b64encode(
                    ("x-access-token:" + push_env["GH_TOKEN"]).encode()
                ).decode()
                push_env.update(
                    {
                        "GIT_CONFIG_COUNT": "1",
                        "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
                        "GIT_CONFIG_VALUE_0": "AUTHORIZATION: basic " + credential,
                    }
                )
            subprocess.run(
                ["git", "push", "origin", "HEAD:gh-pages"],
                check=True,
                cwd=site,
                env=push_env,
            )
        else:
            print("Dry run verified; use --push after reviewing the generated site")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true")
    parser.add_argument(
        "--output", type=Path, help="Export the reviewed site for Pages deployment"
    )
    args = parser.parse_args()
    publish(args.push, args.output)
