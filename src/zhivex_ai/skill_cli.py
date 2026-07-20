from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
from typing import Any

from .skillpacks import (
    install_skill,
    list_installed_skills,
    publish_skill,
    run_skill,
    validate_skill,
)


def _json_input(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    return dict(json.loads(value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zhivex-skills", description="Manage Zhivex skill packages.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate a skill directory or SKILL.md path.")
    validate_parser.add_argument("path")

    install_parser = subparsers.add_parser("install", help="Install a skill from a local path or registry ref.")
    install_parser.add_argument("source")
    install_parser.add_argument("--project-root")
    install_parser.add_argument("--registry-url")
    install_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Confirm that executable Python code from the configured registry may be installed.",
    )
    install_parser.add_argument("--no-lock", action="store_true")

    list_parser = subparsers.add_parser("list", help="List installed skills for a project.")
    list_parser.add_argument("--project-root")

    lock_parser = subparsers.add_parser("lock", help="Install a skill and refresh the lockfile.")
    lock_parser.add_argument("source")
    lock_parser.add_argument("--project-root")
    lock_parser.add_argument("--registry-url")
    lock_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Confirm that executable Python code from the configured registry may be installed.",
    )

    run_parser = subparsers.add_parser("run", help="Run a skill entrypoint.")
    run_parser.add_argument("name")
    run_parser.add_argument("--entrypoint")
    run_parser.add_argument("--input")
    run_parser.add_argument("--project-root")

    publish_parser = subparsers.add_parser("publish", help="Publish a skill into a static registry directory.")
    publish_parser.add_argument("path")
    publish_parser.add_argument("--registry-dir", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="Validate and print the resolved skill metadata.")
    doctor_parser.add_argument("path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "validate":
        definition = validate_skill(args.path)
        print(
            json.dumps(
                {
                    "name": definition.name,
                    "version": definition.version,
                    "path": definition.path,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "install":
        installed = install_skill(
            args.source,
            project_root=args.project_root,
            lock=not args.no_lock,
            registry_url=args.registry_url,
            trust_remote_code=args.trust_remote_code,
        )
        print(json.dumps(asdict(installed), indent=2))
        return 0
    if args.command == "list":
        items = [asdict(item) for item in list_installed_skills(project_root=args.project_root)]
        print(json.dumps(items, indent=2))
        return 0
    if args.command == "lock":
        installed = install_skill(
            args.source,
            project_root=args.project_root,
            lock=True,
            registry_url=args.registry_url,
            trust_remote_code=args.trust_remote_code,
        )
        print(json.dumps(asdict(installed), indent=2))
        return 0
    if args.command == "run":
        result = asyncio.run(
            run_skill(
                args.name,
                entrypoint=args.entrypoint,
                input=_json_input(args.input),
                project_root=args.project_root,
            )
        )
        print(
            json.dumps(
                {
                    "skill_name": result.skill_name,
                    "skill_version": result.skill_version,
                    "entrypoint": result.entrypoint,
                    "output": result.output,
                    "artifacts": [asdict(artifact) for artifact in result.artifacts],
                    "logs": result.logs,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "publish":
        index = publish_skill(args.path, registry_dir=args.registry_dir)
        print(json.dumps({"registry_url": index.registry_url, "skills": index.skills}, indent=2))
        return 0
    if args.command == "doctor":
        definition = validate_skill(args.path)
        print(
            json.dumps(
                {
                    "name": definition.name,
                    "version": definition.version,
                    "entrypoints": [item.name for item in definition.entrypoints],
                    "dependencies": [item.value for item in definition.dependencies],
                    "resources": definition.resources,
                },
                indent=2,
            )
        )
        return 0
    parser.error(f"Unsupported command: {args.command}")
    return 2
