# Versioned documentation

PY-HU-13 adds a Python-native MkDocs site. Its source is `docs/site/`; the existing
long-form guides remain linked at the documented release's exact source revision.
The stack uses MkDocs' built-in search and theme, with a version selector. No
implementation-module imports are recommended in the generated reference.

## Build and preview

```bash
make dev
make docs-check
python3 -m http.server 8000 --directory site
```

Open `http://localhost:8000/0.23.0/` or the version index at
`http://localhost:8000/`. `make docs-build` builds without rerunning tooling tests.
The uv installation must meet the repository's documented tooling floor.

The default build downloads the exact published wheel recorded in
`docs/site/published.json`. It verifies the hash against both PyPI metadata and
the downloaded bytes, then installs it in a clean venv. `python -I` prevents
checkout imports. The renderer reads public exports and stability metadata from
that installation, renders callable signatures/docstrings and fails if any Stable
root export lacks a reference entry. Docstring absence is not filled with invented
API behavior. The internal stability manifest is read by tooling only.

`documentation-evidence.json` inside each generated version records wheel SHA256,
source commit, installed version, namespace/level inventory, Stable coverage and
snippet execution modes. The installed wheel's runtime dependencies are resolved
at build time; the documentation tooling itself is locked by `uv.lock`.

## Validation boundaries

- All Python snippets are parsed, compiled and checked against the installed public
  namespaces and exported names. Star imports and implementation imports fail.
- The mock-agent snippet is executed in the clean wheel environment. The live-agent
  example is compiled and import-checked; site builds do not spend provider tokens.
- MkDocs strict mode and the HTML checker reject missing pages/assets/anchors.
- Immutable GitHub source links are checked against their exact Git objects. This
  proves target existence without depending on GitHub HTTP availability; it does
  not promise third-party service uptime.
- Every Stable root export appears in the reference. Focused and legacy extension
  exports retain their installed Stable/Beta/Experimental classification.
- The timed human onboarding and full production recipes remain separate HU14/HU16
  acceptance work. This site does not claim their completion or package-wide GA.

## Candidate previews

```bash
.venv/bin/python -m build --wheel --no-isolation
.venv/bin/python scripts/build_docs.py --wheel dist/zhivex_ai_sdk-0.23.0-py3-none-any.whl --output site-candidate
```

Candidate pages use the `candidate/` path and an explicit unpublished label, even
when their package version equals a previously published version. Their hash is
recorded separately. The source revision is the checkout HEAD, with the channel
identifying that this is a checkout build; commit the changes before retaining
release evidence. Candidate output is never included in the publishing command.

## CI and deployment

The `Documentation` workflow runs tests, strict published-wheel documentation and
an isolated candidate preview on every PR and main push. It retains both sites and
their JSON evidence as the `python-sdk-documentation` artifact.

Publishing is a separate manual `workflow_dispatch` on `main`, behind the
`github-pages` environment. Configure repository Pages with GitHub Actions as its
build source. The workflow archives only published-version content on `gh-pages`,
preserves older version directories and deploys the reviewed static artifact with
GitHub Pages. No SDK tag, PyPI upload or stability promotion occurs.

Target URL: `https://zhivex.github.io/zhivex-ai-sdk-py/`. A target URL is not evidence
of a successful deployment; use the Pages deployment run and a reachable page.

For a local deployment rehearsal:

```bash
.venv/bin/python scripts/publish_docs.py
```

This validates and creates an ephemeral Git commit without pushing. The explicit
`--push` flag archives the site; CI additionally uploads and deploys the Pages
artifact. The archive push never uses force and fails on a concurrent update.

To document a new release, update `published.json` only after verifying its actual
PyPI version, SHA256 and source revision. Rebuild and review the guides against
that wheel before invoking the manual publication. Existing version content is
retained, so a new release cannot silently relabel earlier evidence.

Stack documentation: [MkDocs configuration](https://www.mkdocs.org/user-guide/configuration/)
and [static deployment](https://www.mkdocs.org/user-guide/deploying-your-docs/).

## Initial validation (2026-09-05)

Published 0.23.0: 206/206 Stable root symbols documented, 8 namespace reference
pages, 7 Python blocks compiled and public-import checked, one isolated mock-agent
execution and one compile-only live example. A separate candidate build passed.
The browser check verified home/navigation, version selector and search for
`run_agent`, including navigation to its reference anchor with no horizontal
overflow or reported browser errors. The publisher dry run built the complete
archive without pushing. Evidence: [documentation validation](validation/2026-09-05-documentation.json).
