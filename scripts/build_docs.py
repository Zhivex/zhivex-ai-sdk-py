"""Build versioned MkDocs from a hash-verified published wheel or a labeled candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        data = dict(attrs)
        if data.get("id"):
            self.ids.add(str(data["id"]))
        if tag == "a" and data.get("href"):
            self.links.append(str(data["href"]))
        if tag in {"script", "img", "link"}:
            resource = data.get("src") or data.get("href")
            if resource:
                self.links.append(resource)


def check_site(site: Path, base_path: str = "") -> None:
    """Fail on missing local pages, assets and anchor targets in the built site."""
    documents = {}
    for path in site.rglob("*.html"):
        parser = Links()
        parser.feed(path.read_text())
        documents[path.resolve()] = parser
    errors = []
    for path, parser in documents.items():
        for link in parser.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            # MkDocs uses bare '#' as a non-navigation UI toggle.
            if link == "#":
                continue
            if url.path.startswith("/"):
                if base_path and not url.path.startswith(base_path.rstrip("/") + "/"):
                    errors.append(f"{path.name}: outside deployment prefix {link}")
                    continue
                target = (
                    site / unquote(url.path[len(base_path) :]).lstrip("/")
                ).resolve()
            else:
                target = (
                    (path.parent / unquote(url.path)).resolve() if url.path else path
                )
            if target.is_dir():
                target /= "index.html"
            if not target.is_file():
                errors.append(f"{path.name}: missing {link}")
            elif (
                url.fragment
                and target in documents
                and unquote(url.fragment) not in documents[target].ids
            ):
                errors.append(f"{path.name}: missing anchor {link}")
    if errors:
        raise ValueError("\n".join(errors[:30]))


def download_published(identity: dict[str, str], directory: Path) -> Path:
    url = f"https://pypi.org/pypi/zhivex-ai-sdk/{identity['version']}/json"
    with urllib.request.urlopen(url, timeout=30) as response:
        release = json.load(response)
    matches = [
        item
        for item in release["urls"]
        if item["filename"].endswith(".whl")
        and item["digests"]["sha256"] == identity["wheel_sha256"]
    ]
    if len(matches) != 1:
        raise ValueError("Published wheel identity not found in PyPI metadata")
    item = matches[0]
    if urlsplit(item["url"]).hostname != "files.pythonhosted.org":
        raise ValueError("Unexpected PyPI artifact host")
    wheel = directory / Path(item["filename"]).name
    with urllib.request.urlopen(item["url"], timeout=60) as response:
        wheel.write_bytes(response.read())
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != identity["wheel_sha256"]:
        raise ValueError("Published wheel SHA256 mismatch")
    return wheel


def build(args: argparse.Namespace) -> Path:
    import yaml

    output = args.output.resolve()
    identity = json.loads((ROOT / "docs/site/published.json").read_text())
    identity["channel"] = "published"
    with tempfile.TemporaryDirectory(prefix="zhivex-docs-") as temporary:
        work = Path(temporary)
        if args.wheel:
            wheel = args.wheel.resolve()
            import zipfile
            from email.parser import BytesParser

            with zipfile.ZipFile(wheel) as archive:
                metadata_name = next(
                    n for n in archive.namelist() if n.endswith(".dist-info/METADATA")
                )
                identity["version"] = str(
                    BytesParser().parsebytes(archive.read(metadata_name))["Version"]
                )
            identity["wheel_sha256"] = hashlib.sha256(wheel.read_bytes()).hexdigest()
            identity["source_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip()
            identity["channel"] = "candidate (unpublished; checkout build)"
        else:
            wheel = download_published(identity, work)
        if not re.fullmatch(r"\d+\.\d+\.\d+(?:[a-zA-Z0-9.+-]*)", identity["version"]):
            raise ValueError("Unsafe documentation version")
        venv = work / "venv"
        uv = os.environ.get("UV", "uv")
        subprocess.run([uv, "venv", str(venv), "--python", sys.executable], check=True)
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [uv, "pip", "install", "--python", str(python), str(wheel)], check=True
        )
        identity_path = work / "identity.json"
        identity_path.write_text(json.dumps(identity))
        docs = work / "docs"
        subprocess.run(
            [
                str(python),
                "-I",
                str(ROOT / "scripts/render_docs_reference.py"),
                "--source",
                str(ROOT / "docs/site"),
                "--output",
                str(docs),
                "--metadata",
                str(identity_path),
            ],
            check=True,
            cwd=work,
        )
        # Source links are immutable GitHub blob links. Validate their target in
        # the matching local Git tree, without depending on GitHub HTTP rate limits.
        for page in docs.rglob("*.md"):
            for target in re.findall(
                r"https://github.com/Zhivex/zhivex-ai-sdk-py/blob/([0-9a-f]{40}/[^)\s]+)",
                page.read_text(),
            ):
                commit, relative = target.split("/", 1)
                subprocess.run(
                    ["git", "cat-file", "-e", f"{commit}:{relative.split('#')[0]}"],
                    cwd=ROOT,
                    check=True,
                )
        slug = "candidate" if args.wheel else identity["version"]
        assets = docs / "assets"
        assets.mkdir()
        (assets / "versions.js").write_text(
            (ROOT / "docs/site/versions.js").read_text()
        )
        site = output / slug
        config = {
            "site_name": f"Zhivex AI SDK Python · {identity['version']}"
            + (" · Candidate" if args.wheel else ""),
            "site_description": "Portable Python agents, durable workflows and provider contracts.",
            "site_url": f"https://zhivex.github.io/zhivex-ai-sdk-py/{slug}/",
            "docs_dir": str(docs),
            "site_dir": str(site),
            "theme": {
                "name": "mkdocs",
                "highlightjs": False,
                "color_mode": "light",
                "user_color_mode_toggle": False,
                "navigation_depth": 2,
            },
            "plugins": ["search"],
            "extra_javascript": ["assets/versions.js"],
            "strict": True,
            "validation": {
                "links": {
                    "not_found": "warn",
                    "anchors": "warn",
                    "unrecognized_links": "warn",
                }
            },
            "nav": [
                {"Start": "index.md"},
                {"Quickstart": "quickstart.md"},
                {"Foundation": "foundation.md"},
                {"Agents": "agents.md"},
                {"Workflows": "workflows.md"},
                {"Gateway": "gateway.md"},
                {"Providers": "providers.md"},
                {"Production": "production.md"},
                {
                    "API reference": ["reference/index.md"]
                    + [
                        f"reference/{name}.md"
                        for name in [
                            "root",
                            "workflows",
                            "evals",
                            "protocols",
                            "responses",
                            "experimental",
                            "experimental-providers",
                            "experimental-realtime",
                        ]
                    ]
                },
            ],
        }
        config_path = work / "mkdocs.yml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        subprocess.run(
            [
                sys.executable,
                "-m",
                "mkdocs",
                "build",
                "--strict",
                "-f",
                str(config_path),
            ],
            check=True,
        )
        check_site(site, f"/zhivex-ai-sdk-py/{slug}")
        versions = sorted(
            p.name
            for p in output.iterdir()
            if p.is_dir()
            and (p / "index.html").is_file()
            and re.fullmatch(r"(?:candidate|[0-9][a-zA-Z0-9.+-]*)", p.name)
        )
        (output / "versions.json").write_text(json.dumps(versions) + "\n")
        (output / "index.html").write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1"><title>Zhivex Python documentation</title>'
            "<main><h1>Zhivex AI SDK Python</h1><p>Select a documentation version. The package remains Beta.</p><ul>"
            + "".join(f'<li><a href="{v}/">{v}</a></li>' for v in versions)
            + "</ul></main></html>"
        )
        (output / ".nojekyll").touch()
        check_site(output, "/zhivex-ai-sdk-py")
        print(f"Documentation verified: {site}")
        return site


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel",
        type=Path,
        help="Check an unpublished candidate instead of the pinned PyPI wheel",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "site")
    build(parser.parse_args())
