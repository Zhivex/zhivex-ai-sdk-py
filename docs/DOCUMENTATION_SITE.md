# Python documentation portal

The public documentation lives at [sdk.zhivex.ai](https://sdk.zhivex.ai/doc).
The [sdk-page repository](https://github.com/Zhivex/sdk-page) owns its Astro
frontend, generated reference, documentation tests and Vercel deployment.

## Reader entrypoints

- [Python getting started](https://sdk.zhivex.ai/doc/python/getting-started)
- [Python 0.23.0 guides](https://sdk.zhivex.ai/doc/python/0.23.0/index)
- [Python 0.23.0 reference and symbol search](https://sdk.zhivex.ai/doc/python/0.23.0/reference/index)
- [Release notes](https://sdk.zhivex.ai/doc/releases)

The versioned reference describes a published wheel, not the current checkout.
The package remains Beta; individual public APIs retain Stable/Beta/Experimental
labels. Documentation checks do not certify live provider availability.

## Ownership and validation

This SDK repository owns runtime contracts, source docstrings, public stability
manifests, examples, package tests and release artifacts. Run `make check` and
`make release-check` here when required by an SDK change.

The portal owns the wheel-to-documentation pipeline. In a `sdk-page` checkout:

```bash
bun install --frozen-lockfile --ignore-scripts
bun run docs:python:check
bun run docs:check
bun run build
```

Snapshot reproduction downloads the wheel selected by version and SHA256,
installs it in an isolated Python environment, reads the final public stability
manifest and signatures, and checks snippets. Only the registered offline agent
is executed; provider-backed examples are compiled. CI checks the homepage and
guide call signatures, snapshot integrity, and built pages, resources and anchors.
See the [portal runbook](https://github.com/Zhivex/sdk-page#published-python-documentation)
for Python/uv versions, dependency constraints and sibling SDK checkout paths.

## Updating a release

1. Validate and publish the SDK using this repository's release process.
2. In `sdk-page`, add a version directory under `docs/python` with the published
   version, source commit and wheel SHA256. Preserve earlier version directories.
3. Run `bun run docs:python`, update release curation, then run
   `bun run docs:releases` and `bun run docs:reference`.
4. Run the portal checks and review its Vercel preview, including version identity,
   search, public signatures and mobile navigation.
5. Merge the portal change and verify the production URLs before recording the
   documentation rollout as complete.

Candidate SDK wheels continue through this repository's installed-artifact gates;
do not label a checkout build as published documentation. A portal change alone
does not publish an SDK package.

## Publication boundary

The portal integration is [sdk-page PR #2](https://github.com/Zhivex/sdk-page/pull/2).
This repository does not maintain a parallel MkDocs site or documentation deployment
workflow. Do not dispatch the historical GitHub Pages publisher from an older
branch. Any earlier repository Pages configuration is unused by this flow.
