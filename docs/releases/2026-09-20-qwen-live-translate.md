# Qwen3.8 LiveTranslate local integration evidence — September 20, 2026

Implemented the Beta native WebSocket integration for exactly
`qwen3.8-livetranslate-flash-realtime`. This is local candidate evidence, not a
published release or protected release certification. The wheel uses the current
checkout version `0.27.0`; its digest, rather than that version alone, identifies
the candidate.

## Validation

`UV_CACHE_DIR=/tmp/zhivex-qwen-uv-cache make check` passed: 1,237 tests,
356 subtests, 16 credentialed/optional tests skipped; total coverage 85.51%.
Compilation, Ruff, mypy, lock consistency, public stub, provider certification
schema and generated support-matrix checks passed. The package built successfully,
passed Twine metadata validation and loaded its native factory/focused namespace
from a separately installed wheel.

The [installed-wheel live report](2026-09-20-qwen-translate-final-installed.json)
records successful Singapore checks for:

- Audio input to Spanish text, with source transcription and bilingual correlation.
- Audio input to Spanish text and PCM speech output.
- JPEG context alongside audio translation.
- Dynamic voice-clone configuration with translated audio output.
- Two synthetic voices producing separate completed source/translation segments.
  This verifies multi-segment delivery, not speaker attribution accuracy.

Every operation required the final `session.finished` acknowledgement. Synthetic
input was used, and no output audio, transcripts, keys, authenticated URLs or raw
error messages were saved. The runner verified installed SDK bytes against the
wheel before connecting. Artifact SHA-256:

`d63afc19256cb95d6d22c6aa1585aa33e079ec67433dcdeb36c3e1e1c235dcb2`

The first live attempt exposed an upstream default: the server selected legacy
voice `Chelsie` and rejected it. The adapter now explicitly selects documented
voice `Tina` unless a voice or cloning mode is configured; subsequent text and
speech checks passed. This regression is covered by the handshake payload test.

## Boundaries

Live evidence applies to the Singapore DashScope route and the recorded candidate
wheel. Beijing/workspace-specific endpoints have offline routing coverage but
were not authenticated in this run. Voice-clone acceptance and output do not
certify perceptual similarity, and JPEG acceptance does not certify visual
reasoning. Speaker metadata is preserved as provider-owned data, without claiming
a Stable speaker schema or measuring diarization accuracy. Stored voice
provisioning, WebRTC and AOQ are separate provider services/protocols.

See [the integration guide](../providers/qwen-live-translate.md),
[example](../../examples/realtime/qwen_live_translate.py), and
[synthetic fixture reproduction](../../tests/fixtures/realtime/README.md).

Earlier candidate runs are retained for diagnosis in the
[initial installed report](2026-09-20-qwen-translate-installed.json) and
[two-voice report](2026-09-20-qwen-translate-multi-speaker.json). Their artifact
digest differs from the final candidate above.
