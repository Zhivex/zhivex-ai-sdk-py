# Qwen3.8 LiveTranslate (Beta)

The native integration targets exactly `qwen3.8-livetranslate-flash-realtime` over
WebSocket. Use the Stable `create_qwen` factory and `zhivex_ai.live` configuration
and events. The Qwen transport, native methods and provider settings are Beta;
the upstream model's Stable designation is separate from SDK release certification.

## Connect

Install `zhivex-ai-sdk[realtime]`. Configure `QWEN_API_KEY` or `DASHSCOPE_API_KEY`
and optionally `DASHSCOPE_WORKSPACE_ID`. Credentials must belong to the selected
region/workspace. Never put the account key in a browser.

```python
from zhivex_ai import create_qwen
from zhivex_ai.live import RealtimeSessionConfig

provider = create_qwen(region="intl", workspace_id="your-workspace")
session = await provider.native.realtime_model(
    "qwen3.8-livetranslate-flash-realtime"
).connect(RealtimeSessionConfig(
    translation_target_language_code="es",
    provider_options={"output_modalities": ["text", "audio"]},
))
```

`region="intl"` uses Singapore; `"cn"` uses Beijing. The workspace endpoint is
`wss://<workspace>.<region>.maas.aliyuncs.com/api-ws/v1/realtime?model=...`.
Existing official DashScope Beijing/Singapore base URLs remain usable without a
workspace ID. A workspace-specific `base_url` is also recognized. US has no
implicit LiveTranslate route. For a trusted gateway, set an explicit
`realtime_url="wss://..."`; that endpoint receives the account credential.
`realtime_connection_factory` allows an application-owned transport. Opening
waits for both `session.created` and `session.updated`. `RealtimeConnectOptions`
sets bounded connection/setup/update/finish timeouts (30 seconds by default).
There are no automatic retries/replay after audio has been sent.

## Audio, images and lifecycle

Send `AudioFrame` with `media_type="audio/pcm"`: signed little-endian 16-bit,
mono, 16000 Hz. Bytes and base64 strings are accepted; WAV headers are rejected.
The SDK does not transcode audio. Output is PCM16, mono, 24000 Hz. Read events
concurrently with input, and pace audio at its actual sample rate. Use
`output_modalities=["text"]` for text-only translation. The default is text and
audio; languages documented as text-only reject audio output before connecting.

`await session.aclose()` sends `session.finish`, drains final output until
`session.finished`, then releases the connection. Missing acknowledgement,
provider failure, malformed events and timeouts fail explicitly. A final
`AudioFrame(is_final=True)` does the same; it ends the entire translation session,
not just one utterance. Concurrent/repeated finish calls share one completion.
Cancellation closes the transport and does not claim successful final translation.

Additional native methods are typed through the focused Beta namespace:

```python
from typing import cast
from zhivex_ai.experimental.qwen import QwenLiveTranslateSession

native = cast(QwenLiveTranslateSession, session)
await native.send_image(jpeg_bytes, media_type="image/jpeg")
await native.clear_audio()  # Only clears uncommitted audio.
await native.update(translation_target_language_code="fr")
await native.finish()  # Equivalent to graceful aclose().
```

Send audio before images. Use JPEG frames up to 500000 bytes, maximum 1080p,
no faster than two per second; the application extracts frames from video.
The native method rejects excess size/rate and incorrect formats. No tools,
text prompts, manual `response.create`, browser tokens, WebRTC or AOQ are exposed.
Those protocols are separate from the documented 3.8 WebSocket integration.

## Translation settings and voice cloning

`provider_options` is the session body, with `output_modalities`, `translation`,
`audio`, `enable_voice_clone`, and `voice_clone_options`. Unknown/legacy 3.5
settings are rejected, as are conflicts with typed configuration. Updates wait
for the server acknowledgement; a timeout closes the session because its remote
configuration is uncertain. `provider_options` updates merge nested settings with the acknowledged configuration.
A new `corpus.phrases` map replaces the old glossary (use an empty map to clear it).
Set `enable_voice_clone=False` and a preset voice to disable cloning.

```python
config = RealtimeSessionConfig(
    translation_target_language_code="es",
    voice="default",
    provider_options={
        "output_modalities": ["text", "audio"],
        "translation": {"corpus": {"phrases": {"orbit seven": "órbita siete"}}},
        "enable_voice_clone": True,
        "voice_clone_options": {"frequency": "always"},
    },
)
```

`once` clones once; `always` follows changes in the input voice; both use
`audio.output.voice="default"`. `never` requires an existing
`qwen-translate-vc-...` voice ID. Provisioning a stored voice is a separate
provider service, not part of this realtime session client. Speaker detection
uses `audio.input.turn_detection.type="speaker_detection"`. Typed `voice`,
`turn_detection` and sample-rate settings map to this nested audio contract.
Source transcription is always enabled in 3.8; source language detection is
server-owned. The adapter explicitly selects `Tina` when no voice is supplied: a
Singapore live check returned an unsupported legacy `Chelsie` server default.
`same_language_skip_options`, 3.5 `modalities`, and Gemini's
`translation_echo_target_language` are not supported.

## Event semantics

- Source ASR deltas become cumulative `RealtimeTranscriptEvent(role="user",
  is_final=False)` snapshots; the completed event carries the final transcript.
- Translation deltas become `RealtimeTextDeltaEvent`. Append them once. Final
  assistant transcripts are snapshots, not extra deltas to append.
- Audio chunks carry item/response IDs, sample rate and channels.
- Translation metadata includes `previous_item_id` from `conversation.item.created`
  to pair source and translation. Raw speaker fields are preserved without
  inventing a cross-provider speaker schema.
- `RealtimeResponseCompletedEvent.provider_metadata` preserves response status and
  usage. Failed/incomplete responses and failed ASR end with an error.
- `native_event_stream()` preserves every server event, including unknown events,
  speaker/voice metadata, buffer acknowledgements and response details. Native
  payloads contain user content: do not log them wholesale.

Both streams retain at most 1000 events; lagging consumers receive an explicit
overrun error. Consume during the session and persist only application-approved
content. This is not a resumable event log or a guarantee of perfect diarization,
voice fidelity or translation accuracy.

## Example and verification

```sh
.venv/bin/python examples/realtime/qwen_live_translate.py \
  tests/fixtures/realtime/orbit-seven-16k.wav --language es --output /tmp/translation.wav
.venv/bin/python -m pytest tests/test_qwen_realtime.py -q
.venv/bin/python scripts/smoke_qwen_translate.py --report /tmp/qwen-translate.json
```

The example loads the local `.env`; the smoke expects exported credentials.
The smoke uses synthetic speech, checks original and translated markers, audio,
bilingual correlation and graceful completion, and writes only redacted evidence.
Add `--image <synthetic.jpg>`, `--voice-clone`, and
`--multi-speaker-fixture <synthetic.wav>` for additional scenarios.
See [synthetic fixture reproduction](../../tests/fixtures/realtime/README.md).
For exact-wheel integration set `ZHIVEX_SMOKE_USE_INSTALLED=1` and pass `--wheel`;
the verifier rejects checkout imports and compares installed package bytes.
A skipped, rejected or failed live operation is not a passing certification.
The runner checks clone configuration/output, not perceptual identity quality.

## Protocol sources reviewed September 20, 2026

- [Model guide and 3.8 example](https://help.aliyun.com/en/model-studio/qwen3-5-livetranslate-flash-realtime)
- [Client events](https://help.aliyun.com/en/model-studio/live-translator-client-events)
- [Server events](https://www.alibabacloud.com/help/en/model-studio/live-translator-server-events)

The upstream documentation mixes 3.5 and 3.8 examples. This integration uses the
3.8 fields/events; it does not silently route older model IDs through this parser.

Current [local installed-wheel evidence](../releases/2026-09-20-qwen-live-translate.md)
is separate from protected release certification.
