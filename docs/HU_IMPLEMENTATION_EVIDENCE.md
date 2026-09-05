# HU reconciliation and implementation — 2026-09-05

## Baseline and scope

Base: main `b129bb5a4741dc0bc235b3b7749d647195758f7b`, published 0.23.0. Recovered certification controls from local `ec840ce2531a791795c0aeb30e3af3e3b0c01c22`; that old feature branch is absent remotely. Current work combines HU06–09 with the remaining HU10 lint work and HU12 compatibility evidence. It does not implement HU11 or HU13–20 or declare 1.0 GA.

HU03 is already integrated through [PR36](https://github.com/Zhivex/zhivex-ai-sdk-py/pull/36); Notion was moved from Testing to Listo after revalidation. All five phase-0 HUs are delivered.

## HU06–08 certification

Current records use the published 0.23.0 wheel SHA256 `ccf02e9727edbfbd9bb4019efcfe646b21c31e97d384037ed24b723e1662a464`, not the candidate SHA in the old local release report. The installed venv was created from PyPI and the downloaded wheel hash was verified before calls. Each provider had a separate bounded run using synthetic markers and a nonce tool. No prompts, responses, credentials, or endpoints were retained.

- OpenAI Luna, Anthropic Fable 5.1, Qwen3.8-Max-0902, DeepSeek V4 Flash: four required operations passed locally.
- Meta Standard: five required operations passed locally, including portable retrieval.
- Azure and Vertex: credentials/deployment configuration unavailable.
- Kimi: credential unavailable.
- vLLM: no active reviewed deployment configured. Historical 0.22.0 Docker evidence remains historical.
- Gemini 3.8: a diagnostic run passed generation then encountered ReadTimeout; the bounded repeat with low reasoning produced PROVIDER_UNAVAILABLE. The retained target is blocked. No successful certification is claimed.
- Meta Contributor: normalized from retained protected [publication evidence](https://github.com/Zhivex/zhivex-ai-sdk-py/actions/runs/33983320649), with unchanged timestamp, wheel hash and operations. This is a separate Beta target and cannot certify Standard.

Aggregate: five integration-only Tier-1 targets, five blocked Tier-1 targets, one certified Beta Contributor canary. The current Tier-1 cohort has no newly protected certificates. The manual workflow must be integrated and its protected environment configured before local integrations can be replaced by independently retained workflow evidence.

The validator now binds the reviewed version/source/hash, expires local and protected successes after 30 days, writes reports before failing required-target gates, and canonicalizes redundant Pydantic Literal schema representation across supported versions. Historical 0.22.0 policy and records remain separate and unchanged.

## HU09 reproducibility

uv 0.12.4 generated the public lock. A fresh locked Python 3.11 environment passed check/release gates. Independent minimum-core, minimum-extras, and latest environments ran the full test suite; the core job pins httpx 0.27.0 and pydantic 2.8.0 exactly. Optional API tests explicitly skip only when their extra is absent; the other jobs execute them.

The initial minima run reproduced the old pytest-asyncio/pytest-9 collection incompatibility. Raising the tooling floor fixed it. Pydantic 2.8's redundant single-value enum is normalized without changing validation semantics. httpx2 removes the known Starlette TestClient warning; unrelated upstream deprecations remain visible. Wheel/sdist verification confirms the core excludes websockets and the realtime extra imports it successfully. See the dependency compatibility guide for commands and retained version reports.

## HU10 shared-provider analysis

PR37 already added all provider adapters to mypy and removed skipped provider imports. This change adds I/B/ASYNC/SIM101/SIM103/SIM114 checks for openai_compat, _payload and _url_security through make lint and CI. The only source changes are import ordering; no new ignores or public exports were added by this work. The stable contract remains unchanged relative to main b129bb5.

## HU12 persistence boundary

PR37 extracted memory/checkpoint persistence into _agent_persistence.py (605 lines), reducing agent.py by 587 net lines. Persistence was chosen as the single boundary; the initial story's example boundaries were tool execution/approvals/events. This choice serves the same durability/serialization objective and is documented explicitly in Notion.

The pre-extraction wheel 0.22.0 generated tests/fixtures/agent_persistence_0_22.json with synthetic memory and a checkpoint. The current code reads that data from legacy SQLite rows and roundtrips it without drift; raw_response remains redacted. Existing run/resume/stream/approval tests also pass. AST comparison confirms serializer bodies are unchanged; constructor changes defer imports to avoid circular dependencies and delegate the clock to the existing agent clock.

The release's separate HTTPTransport/aclose_default_clients additions are not attributed to the extraction. This HU adds no exports, store schema, serialized fields, or runtime changes beyond those already released in PR37.

## Delivery boundaries

Local validation is separate from the published artifact, PR CI, merge and protected certification. No new package version or release tag is created by this implementation. Notion remains in Testing for new delivery until remote validation and any applicable protected evidence are recorded. External provider blockers must never be converted into success.

## Validation results

- Final locked Python 3.11: 898 passed, 16 skipped, 161 subtests, 84.12% coverage; compile, scoped/global Ruff, mypy, schema and generated docs passed.
- Final local locked Python 3.14: 898 passed, 16 skipped, 161 subtests, 84.13% coverage. Optional Postgres live tests are not claimed locally; CI provisions PostgreSQL.
- Independent core-minimum run: 884 passed, 28 skipped (optional extras absent), 159 subtests; httpx 0.27.0 and pydantic 2.8.0.
- Independent compatible minimum-extras and latest runs: each 896 passed, 16 skipped, 161 subtests before adding the two legacy persistence characterization tests. These tests passed separately and in both final locked checks.
- release-check passed: wheel/sdist build and fresh installs, every extra including realtime, pip check, all-extras audit, local audit, and Twine. No known vulnerabilities were reported.
- The known Starlette/httpx fallback warning is gone. Remaining warnings come from upstream a2a/protobuf and other compatibility deprecations and are not suppressed.
