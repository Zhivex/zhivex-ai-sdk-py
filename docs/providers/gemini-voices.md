# Gemini Voices and 3.8 TTS (Beta)

`provider.native.voices()` exposes the Gemini Developer API Voices service. Its
client type is available from `zhivex_ai.experimental.gemini`, a focused Beta
namespace. This addition has offline transport tests; it does not certify project
access, voice consent verification or live synthesis. Vertex, Azure and Bedrock
are outside this integration.

```python
from zhivex_ai import create_gemini, generate_speech
from zhivex_ai.experimental.gemini import GeminiVoicesClient

provider = create_gemini()
voices: GeminiVoicesClient = provider.native.voices()
voice = await voices.design(
    prompt="A warm, calm narrator with a gentle British accent",
    model="gemini-3.8-flash-tts",
    display_name="Narrator",
)
result = await generate_speech(
    model=provider.speech_model("gemini-3.8-flash-tts"),
    input="Welcome to the next chapter.",
    voice=voice["id"],
)
```

Both `gemini-3.8-flash-tts` and `gemini-3.8-flash-lite-tts` use the documented
`voiceConfig.voice` field for prebuilt names, library IDs, custom IDs and voice
keys. Earlier Gemini models keep their previous prebuilt voice configuration.

`create(body)` accepts the REST request envelope: `store` plus a `voice` object
with `type="prompted"` and `prompted.input`, or `type="replicated"` and
`replicated.source_audio`/`replicated.consent_audio`. Audio objects contain
`mime_type` and base64 `data`. `design()` and `replicate()` are conveniences over
that single creation endpoint. Responses remain dictionaries, preserving usage,
samples, voice IDs, stateless keys and expiration metadata.

Replication requires reference and consent recordings from the same adult
speaker. Google verifies the speaker and required consent statement. Local
validation checks nonempty audio, base64 and MIME shape; it cannot verify spoken
consent. Supply `consent_audio` explicitly to `replicate()`; there is no bypass.
Follow the [replication guide](https://ai.google.dev/gemini-api/docs/voice-replication)
for recording requirements and the exact supported consent statement.

The SDK explicitly defaults creation to `store=True`, matching the guide's
stateful examples. Prompted voices require that value. Replication may use
`store=False`; its response contains `key` instead of a stored ID. Pass the key
to synthesis directly and keep it private. Respect the returned `expire_time`;
the SDK does not renew voices or persist voice keys. Stored voices can be
retrieved with `get("voice_...")` or `get("voices/voice_...")` and deleted with
`delete(...)`. Prebuilt names and stateless keys are rejected by these lifecycle
methods. Creation is not automatically retried, to avoid duplicate resources.

```python
page = await voices.list({"type": ["prompted"], "page_size": 20})
next_page = await voices.list({
    "type": ["prompted"],
    "page_size": 20,
    "page_token": page["next_page_token"],
})
# Or iterate every page while preserving the original filters:
async for item in voices.iter_voices({"language_code": ["en-US"]}):
    print(item["id"])
```

Keep filters identical when continuing a page token. The iterator preserves
filters and detects repeated tokens. Native options use `RetryOptions.timeout_ms`;
creation does not interpret retry counts. See the runnable
[design example](../../examples/audio/gemini_voices.py).

Sources: [REST reference](https://ai.google.dev/api/voices),
[voice design](https://ai.google.dev/gemini-api/docs/voice-design),
[3.8 speech schema](https://ai.google.dev/gemini-api/docs/generate-content/speech-generation).
