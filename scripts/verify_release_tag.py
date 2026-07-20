from __future__ import annotations

import argparse
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def _package_version() -> str:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    return str(pyproject["project"]["version"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify that a release tag matches pyproject.toml.")
    parser.add_argument("--tag", required=True, help="Git tag in v<version> form.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    expected = f"v{_package_version()}"
    if args.tag != expected:
        print(f'Release tag mismatch: expected "{expected}", received "{args.tag}".')
        return 1
    print(f'Release tag "{args.tag}" matches package version {_package_version()}.')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
