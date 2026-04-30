# Support Policy

Zhivex AI SDK is currently published as a `Beta` package.

Related documents:

- [README.md](./README.md)
- [STABILITY.md](./STABILITY.md)
- [VERSIONING.md](./VERSIONING.md)
- [CHANGELOG.md](./CHANGELOG.md)

## Support expectations

- The latest beta release line is the primary target for fixes, documentation updates, and examples.
- The previous beta minor release may receive critical fixes when the change is low risk and clearly scoped.
- Stable APIs are the main compatibility contract for production integrations.
- Stable agent integrations include the core runtime, session helpers, portable agent skills, MCP helper path, MCP-backed registries, and Postgres-backed memory/checkpoint stores documented in [STABILITY.md](./STABILITY.md).
- Beta APIs are supported for early adoption, but they may still evolve between minor releases with changelog coverage.
- Beta provider capability metadata describes current provider agent ergonomics, but it is not a stable behavioral guarantee yet.
- Beta hosted-tool definitions and `provider-data` control parts describe the preferred native tool-registration path, but provider-specific execution semantics may still evolve between minor releases.
- Beta provider-managed approval flows currently cover OpenAI and Azure OpenAI only, including typed `provider-data` payload parsing and agent-runtime approval-policy integration.
- Beta response-reference helpers and `provider-data` UI chunks are supported for OpenAI/Azure continuation workflows and observability, but their exact ergonomics may still evolve between minor releases.
- Beta agent platform helpers cover durable run stores, native subagent tools, replay/evaluation reports, hierarchical trace artifacts, run-tree cancellation, redaction policies, and budget guards.
- Beta workflow agents cover declarative sequential, parallel, and loop orchestration with shared `session.state`; CLI/UI/deploy automation is intentionally outside this beta surface.
- Beta Google native media/job clients cover Gemini/Vertex image, video, music/audio, batch, and interaction workflows where the official Google endpoints expose them. Preview Google models remain subject to Google availability, quota, and deprecation windows.
- Beta Kimi/Moonshot native support covers Chat Completions, Files, Batch, token estimation, and official Formulas tools according to the current Kimi Open Platform docs. Kimi remains a compatibility provider rather than a tier-1 portable provider.
- The README support matrix is generated from runtime metadata and reflects the current provider capability story, but its `Agent Capabilities` section should still be read as beta guidance rather than a stable behavioral guarantee.
- Experimental APIs are available for evaluation and feedback, but they do not carry support or compatibility guarantees.

## What qualifies for patch releases

Patch releases are intended for focused changes such as:

- bug fixes in the stable surface
- low-risk regressions in tier-1 providers
- documentation corrections that unblock adoption
- packaging, build, or release metadata fixes

Patch releases should not introduce silent breaking changes to the documented stable surface.

## Provider support scope

The current tier-1 providers for the stable production API story are:

- OpenAI
- Anthropic
- Azure OpenAI
- Gemini
- Vertex

Anthropic is tier-1 for the portable text-generation surface in this repository. Embeddings, transcription, and speech remain unavailable on the Anthropic provider path here today.

Gemini and Vertex are tier-1 for the portable production surface. Google-specific media generation, Batch API, Interactions API, Deep Research, and Veo operation workflows are exposed through native provider clients rather than the portable contract.

Kimi/Moonshot is supported as a compatibility provider through `provider.native`. Its native text generation uses the official Chat Completions API, with Files, Batch, token estimation, and Formulas exposed as beta native clients.

Other providers remain available, but they should be treated according to the support matrix and the stability level of the specific feature area.

## Upgrade expectations

- Every user-visible change should appear in [CHANGELOG.md](./CHANGELOG.md).
- Changes to stable APIs require migration guidance.
- Deprecations should be documented before removal.
