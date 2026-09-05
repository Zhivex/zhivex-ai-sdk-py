# Provider certification

Contract support, installed-wheel execution, protected live certification, and package GA are distinct.

## Reviewed release

The current policy is `docs/provider-certification-policy.json`. It requires the published 0.23.0 wheel:

- source: `b129bb5a4741dc0bc235b3b7749d647195758f7b`
- wheel SHA256: `ccf02e9727edbfbd9bb4019efcfe646b21c31e97d384037ed24b723e1662a464`
- filename: `zhivex_ai_sdk-0.23.0-py3-none-any.whl`

Each target records its model and required operations. The validator rejects a different version, source revision, or artifact hash. A successful local run is `integration-only`; only retained GitHub Actions evidence can be `certified`. Successful records older than 30 days become `stale`, including local integrations. Missing, failed, blocked, or unsupported evidence never certifies a provider.

The JSON Schema rejects unknown fields. Evidence excludes secrets, endpoints, prompts, and model responses; diagnostics use bounded machine codes. Evidence files are reviewed inputs: setting `platform=github-actions` is not cryptographic authentication. Maintainers must obtain records from the linked protected run and verify its conclusion and artifact before committing them.

## Current versus historical evidence

The 0.22.0 records are preserved unchanged. Evaluate them explicitly with `--policy docs/releases/0.22.0-certification-policy.json`; they are not inputs to the current 0.23.0 matrix. The current policy intentionally starts with fresh 0.23.0 results, so an older protected record cannot hide a newer release's missing certification.

The protected publication canaries have their own narrower policy, `docs/provider-release-canary-policy.json`: OpenAI Luna generation/agent-tool and Meta Contributor generation/streaming/structured-output/retrieval/agent-tool. They do not satisfy the full HU7 four-operation OpenAI cohort or certify Meta Standard. The normalized Contributor record was extracted from the retained [0.23.0 publication run](https://github.com/Zhivex/zhivex-ai-sdk-py/actions/runs/33983320649), preserving its timestamp and published wheel identity.

## Commands

```bash
make certification-check
uv run --no-sync python scripts/generate_support_matrix.py --write-docs
uv run --no-sync python scripts/verify_provider_certification.py --require-target meta-standard --report dist/meta-report.json
```

A required target returns nonzero unless current and certified. A valid blocked/integration report is still written before failure, so operators retain diagnostics. Schema validation alone passes when the schema and records are valid, even if targets remain blocked; it is not a GA gate.

## Protected cohorts

The manual `Tier-1 provider certification` workflow selects one of ten targets, verifies and installs the reviewed published wheel, then runs only bounded synthetic operations. Policies live at `docs/releases/0.23.0-hu7-*-smoke-policy.json` and `docs/releases/0.23.0-hu8-*-smoke-policy.json`.

- HU7: OpenAI, Anthropic, Azure OpenAI, Gemini, Vertex. Required: generation, streaming, structured output, agent tool loop.
- HU8: Qwen, Kimi, DeepSeek, Meta Standard, vLLM. Qwen/Kimi/DeepSeek mark portable retrieval unsupported; Meta/vLLM require retrieval in addition to the four operations.

Configure the `provider-certification` GitHub environment using the exact secret/variable names declared in the workflow. Azure and vLLM need reviewed deployment IDs; Vertex needs a token and project. A deployment model must match its policy. vLLM is deployment-specific: the historical Qwen2.5 1.5B fixture needs automatic tool choice and the Hermes parser; installing Docker alone is not certification.

After the workflow is integrated and the protected environment is configured:

```bash
gh workflow run provider-certification.yml -f provider=meta
```

Inspect the run conclusion; download and review the target artifact, replace only the matching current evidence record, then regenerate docs. Failed runs preserve blocker/failure evidence with 30-day retention. Local runs must never be relabeled as protected runs.

## Local integration

Install the exact published wheel into a fresh environment. Set `ZHIVEX_SMOKE_USE_INSTALLED=1`, `ZHIVEX_SMOKE_STRICT=1`, `ZHIVEX_SMOKE_AGENTS=1`, `ZHIVEX_SMOKE_PORTABLE_CERTIFICATION=1`, `ZHIVEX_SMOKE_META_CERTIFICATION=1`, and `ZHIVEX_SMOKE_SANITIZED_DIAGNOSTICS=1`. Select one provider and its model with `ZHIVEX_SMOKE_PROVIDERS` and the matching `ZHIVEX_SMOKE_*_MODEL`. Point `ZHIVEX_RELEASE_SMOKE_POLICY` at its reviewed policy, `ZHIVEX_SMOKE_ARTIFACT_PATH` at the wheel, and `ZHIVEX_SMOKE_EVIDENCE_PATH` at its evidence file.

Run `scripts/run_live_smoke.py` with the fresh environment's Python. Source tests and candidate wheels are separate from the installed published wheel. Gemini certification excludes native token-count/media calls and uses low reasoning for 3.8; these native operations are outside the portable cohort contract. Provider timeouts/unavailability remain explicit blockers rather than successes.
