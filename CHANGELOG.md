# Changelog

All notable changes to Zhivex AI SDK will be documented in this file.

The format follows grouped release notes with these sections when relevant:

- Added
- Changed
- Fixed
- Deprecated
- Removed

Related documents:

- [README.md](./README.md)
- [STABILITY.md](./STABILITY.md)
- [SUPPORT.md](./SUPPORT.md)
- [VERSIONING.md](./VERSIONING.md)

## Unreleased

### Added

- Stability, versioning, support, and changelog documentation for the documented public surface.
- Production API guidance with FastAPI integration examples for direct, streaming, and gateway-backed APIs.
- Observability guidance and examples for telemetry, request correlation, and gateway attempt hooks.
- Contract coverage for the stable surface, provider support matrix, tier-1 provider assertions, and public package status.
- Dedicated Ollama provider coverage for native text generation, streaming, structured output, tool calling, embeddings, and local smoke validation.

### Changed

- Promoted the package maturity signal from `Alpha` to `Beta`.
- Promoted Anthropic to the tier-1 portable provider set for text-generation API paths.
- Documented tier-1 providers for the stable production API story: OpenAI, Anthropic, Azure OpenAI, Gemini, and Vertex.
- Enforced CI quality gates for linting, type checking, coverage, build validation, and a minimum coverage floor of `80%`.
- Expanded `mypy` coverage over core API-facing modules including `generate_text`, `generate_object`, `middleware`, and `transport`.
- Documented the recommended local Ollama path with `provider.native.*`, the default compatibility token, and optional smoke-run configuration.
- Added async context manager support to `ToolRegistry` and updated MCP guidance to close registries cleanly after use.
- Promoted MCP helpers, MCP-backed registries, and Postgres-backed agent stores into the documented stable surface for production integrations.

### Fixed

- Fixed file-cache serialization so cached generate results round-trip correctly through the on-disk cache.
- Fixed SSE response serialization for dataclass-backed UI message chunks.
- Fixed request snapshotting in `generate_text()` so recorded step requests do not drift as later messages are appended.
- Fixed `stream_agent()` so output guardrails can block streamed assistant text before it is emitted, while preserving live non-text agent events.
- Fixed Postgres-backed agent stores to reject invalid `table_prefix` values early with a clear validation error.
- Fixed agent tool-callable inspection to fall back gracefully when `inspect.signature()` is unavailable.
- Fixed realtime/live voice adapters to distinguish turn completion from true session shutdown, align OpenAI and Azure browser bootstrap with `realtime/client_secrets`, support current OpenAI output-audio events, and accept Gemini ephemeral `access_token` connections.

### Deprecated

- None.

### Removed

- None.

## 0.4.0

### Added

- Initial public release line for the current package version.

### Changed

- No additional entries recorded.

### Fixed

- No additional entries recorded.

### Deprecated

- None.

### Removed

- None.
