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

### Changed

- Promoted the package maturity signal from `Alpha` to `Beta`.
- Promoted Anthropic to the tier-1 portable provider set for text-generation API paths.
- Documented tier-1 providers for the stable production API story: OpenAI, Anthropic, Azure OpenAI, Gemini, and Vertex.
- Enforced CI quality gates for linting, type checking, coverage, build validation, and a minimum coverage floor of `80%`.
- Expanded `mypy` coverage over core API-facing modules including `generate_text`, `generate_object`, `middleware`, and `transport`.

### Fixed

- Fixed file-cache serialization so cached generate results round-trip correctly through the on-disk cache.
- Fixed SSE response serialization for dataclass-backed UI message chunks.
- Fixed request snapshotting in `generate_text()` so recorded step requests do not drift as later messages are appended.

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
