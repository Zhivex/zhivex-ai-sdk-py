"""Qwen3.8 LiveTranslate's native WebSocket protocol (Beta).

The 3.5 protocol is deliberately not selected by model-name heuristics.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import json
import re
from collections import OrderedDict
from collections.abc import AsyncIterable
from copy import deepcopy
from dataclasses import replace
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .._streaming import Broadcast
from ..errors import ConfigurationError, UnsupportedFeatureError, ValidationError
from ..realtime import RealtimeConnection, RealtimeConnectionFactory, open_websocket_connection
from ..types import (
    AudioFrame, ModelCapabilities, RealtimeAudioOutputEvent, RealtimeConnectOptions,
    RealtimeErrorEvent, RealtimeEvent, RealtimeResponseCompletedEvent,
    RealtimeSessionConfig, RealtimeSessionEndedEvent, RealtimeSessionStartedEvent,
    RealtimeTextDeltaEvent, RealtimeTokenResult, RealtimeTranscriptEvent, ToolExecutionResult,
)

MODEL_ID = "qwen3.8-livetranslate-flash-realtime"
AUDIO_LANGUAGES = frozenset("zh en ar de fr es pt id it ko ru th vi ja tr hi ms nl ur nb sv da he fi pl is cs fil fa".split())
TEXT_LANGUAGES = AUDIO_LANGUAGES | frozenset("yue el af ast be bg bn bs ca ceb et gl gu hr hu jv kk kn ky lv mk ml mr pa ro sk sl sw tg az uk".split())
CAPABILITIES = ModelCapabilities(
    streaming=True, tools=False, tool_choice=False, parallel_tool_calls=False,
    structured_output=False, json_mode=False, vision=True, files=False,
    audio_input=True, audio_output=True, embeddings=False, reasoning=False,
    web_search=False, realtime=True, realtime_audio_input=True, realtime_audio_output=True,
)
MAX_MESSAGE_BYTES = 1024 * 1024


def realtime_url(base_url: str, region: str, workspace_id: str | None, override: str | None) -> str:
    if override is not None:
        url = override
    elif workspace_id:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,62}", workspace_id):
            raise ConfigurationError("Invalid Qwen workspace ID.")
        location = {"intl": "ap-southeast-1", "cn": "cn-beijing"}.get(region)
        if location is None:
            raise ConfigurationError("Qwen LiveTranslate supports cn/intl; supply an explicit realtime_url for a gateway.")
        url = f"wss://{workspace_id}.{location}.maas.aliyuncs.com/api-ws/v1/realtime"
    else:
        parsed = urlsplit(base_url)
        host = parsed.hostname or ""
        if host in {"dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com"} or host.endswith((".cn-beijing.maas.aliyuncs.com", ".ap-southeast-1.maas.aliyuncs.com")):
            url = urlunsplit(("wss", parsed.netloc, "/api-ws/v1/realtime", "", ""))
        else:
            raise ConfigurationError("Set workspace_id or realtime_url for Qwen LiveTranslate on this endpoint.")
    parsed = urlsplit(url)
    try:
        valid_port = parsed.port is None or 1 <= parsed.port <= 65535
    except ValueError:
        valid_port = False
    if (parsed.scheme != "wss" or not parsed.hostname or parsed.username or parsed.password
            or parsed.fragment or not valid_port or any(ord(c) < 33 for c in url)):
        raise ConfigurationError("Qwen realtime_url must be a trusted wss URL without userinfo or fragment.")
    query = parse_qsl(parsed.query, keep_blank_values=True)
    if any(key == "model" and value != MODEL_ID for key, value in query):
        raise ConfigurationError("Qwen realtime_url contains a conflicting model.")
    query = [(key, value) for key, value in query if key != "model"]
    return urlunsplit(parsed._replace(query=urlencode([*query, ("model", MODEL_ID)])))


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"Qwen {name} must be an object.")
    return deepcopy(value)


def _merge(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        if key in target and target[key] != value:
            raise ValidationError(f"Conflicting Qwen {key} settings.")
        target[key] = value


def _merge_options(previous: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(previous)
    for key, value in changes.items():
        if key != "phrases" and isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_options(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    if changes.get("enable_voice_clone") is False:
        merged.pop("voice_clone_options", None)
    return merged


def session_payload(config: RealtimeSessionConfig) -> dict[str, Any]:
    if config.instructions or config.tools or config.tool_choice is not None or not config.auto_response:
        raise UnsupportedFeatureError("Qwen LiveTranslate does not support instructions, tools, or manual responses.")
    if config.translation_echo_target_language is not None:
        raise UnsupportedFeatureError("Qwen 3.8 does not support translation_echo_target_language.")
    for media in (config.input_audio_media_type, config.output_audio_media_type):
        if media not in (None, "audio/pcm"):
            raise ValidationError("Qwen 3.8 LiveTranslate expects raw PCM16 audio/pcm.")
    if config.channels not in (None, 1):
        raise ValidationError("Qwen LiveTranslate expects mono audio.")
    payload = _object(config.provider_options or {}, "provider_options")
    allowed = {"output_modalities", "audio", "translation", "enable_voice_clone", "voice_clone_options"}
    if payload.keys() - allowed:
        raise UnsupportedFeatureError("Unsupported Qwen 3.8 session option; use the 3.8 nested audio contract.")
    modalities = payload.setdefault("output_modalities", ["text", "audio"])
    if modalities not in (["text"], ["text", "audio"]):
        raise ValidationError('Qwen output_modalities must be ["text"] or ["text", "audio"].')
    translation = _object(payload.get("translation", {}), "translation")
    if translation.keys() - {"language", "corpus"}:
        raise UnsupportedFeatureError("Unsupported Qwen translation option (same_language_skip_options is not supported in 3.8).")
    _merge(translation, "language", config.translation_target_language_code)
    language = translation.setdefault("language", "en")
    if not isinstance(language, str) or language not in TEXT_LANGUAGES:
        raise ValidationError("Unsupported Qwen translation target language code.")
    if "audio" in modalities and language not in AUDIO_LANGUAGES:
        raise ValidationError("This Qwen target language supports text output only.")
    if "corpus" in translation:
        corpus = _object(translation["corpus"], "translation.corpus")
        phrases = _object(corpus.get("phrases", {}), "translation.corpus.phrases")
        if corpus.keys() - {"phrases"} or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in phrases.items()):
            raise ValidationError("Qwen corpus.phrases must map nonempty source terms to translated terms.")
    payload["translation"] = translation
    audio = _object(payload.get("audio", {}), "audio")
    if audio.keys() - {"input", "output"}:
        raise ValidationError("Unsupported Qwen audio setting.")
    for direction, rate in (("input", config.input_sample_rate_hz), ("output", config.output_sample_rate_hz)):
        settings = _object(audio.get(direction, {}), f"audio.{direction}")
        if settings.keys() - ({"format", "turn_detection"} if direction == "input" else {"format", "voice"}):
            raise ValidationError(f"Unsupported Qwen audio.{direction} setting.")
        fmt = _object(settings.get("format", {}), "audio format")
        _merge(fmt, "sample_rate", rate)
        expected = 16000 if direction == "input" else 24000
        if fmt.keys() - {"type", "sample_rate"} or fmt.get("type", "pcm") != "pcm" or fmt.get("sample_rate", expected) != expected:
            raise ValidationError(f"Qwen {direction} must use PCM16 at {expected} Hz.")
        if fmt:
            settings["format"] = fmt
        if direction == "input":
            _merge(settings, "turn_detection", config.turn_detection)
            if "turn_detection" in settings:
                detection = _object(settings["turn_detection"], "turn_detection")
                if detection.keys() - {"type", "threshold", "silence_duration_ms"} or detection.get("type", "speaker_detection") != "speaker_detection":
                    raise ValidationError("Qwen 3.8 requires speaker_detection.")
                threshold = detection.get("threshold", 0.5)
                silence = detection.get("silence_duration_ms", 1000)
                if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
                    raise ValidationError("Qwen speaker detection threshold must be between 0 and 1.")
                if type(silence) is not int or not 200 <= silence <= 6000:
                    raise ValidationError("Qwen silence_duration_ms must be between 200 and 6000.")
        else:
            _merge(settings, "voice", config.voice)
            if "voice" in settings and (not isinstance(settings["voice"], str) or not settings["voice"]):
                raise ValidationError("Qwen voice must be a nonempty string.")
        if settings:
            audio[direction] = settings
    if audio:
        payload["audio"] = audio
    clone = payload.get("enable_voice_clone", False)
    if type(clone) is not bool:
        raise ValidationError("Qwen enable_voice_clone must be boolean.")
    if "voice_clone_options" in payload and not clone:
        raise ValidationError("Qwen voice_clone_options requires enable_voice_clone.")
    if clone:
        if "audio" not in modalities:
            raise ValidationError("Voice cloning requires audio output.")
        options = _object(payload.get("voice_clone_options", {}), "voice_clone_options")
        frequency = options.get("frequency", "once")
        if options.keys() - {"frequency"} or frequency not in ("once", "always", "never"):
            raise ValidationError("Qwen voice clone frequency must be once, always, or never.")
        output = audio.setdefault("output", {})
        voice = output.setdefault("voice", "default")
        if (frequency != "never" and voice != "default") or (frequency == "never" and not voice.startswith("qwen-translate-vc-")):
            raise ValidationError("Voice cloning requires default for once/always or a cloned voice ID for never.")
        payload["audio"] = audio
        payload["voice_clone_options"] = {"frequency": frequency}
    # Some deployments advertise an unsupported legacy voice in session.created.
    # Pin the documented 3.8 default instead of inheriting that server default.
    audio.setdefault("output", {}).setdefault("voice", "Tina")
    payload["audio"] = audio
    return payload


def _bytes(data: bytes | bytearray | memoryview | str) -> bytes:
    try:
        return base64.b64decode(data, validate=True) if isinstance(data, str) else bytes(data)
    except (ValueError, binascii.Error) as error:
        raise ValidationError("Expected raw bytes or valid base64 data.") from error


def _validate_jpeg(raw: bytes) -> None:
    # Read JPEG frame dimensions without adding a decoder/runtime dependency.
    if not raw.startswith(b"\xff\xd8") or not raw.endswith(b"\xff\xd9"):
        raise ValidationError("Qwen requires a complete JPEG image.")
    offset = 2
    while offset + 4 <= len(raw):
        if raw[offset] != 255:
            break
        while offset < len(raw) and raw[offset] == 255:
            offset += 1
        if offset >= len(raw):
            break
        marker = raw[offset]
        offset += 1
        if marker in {0xD9, 0xDA}:
            break
        if marker in {0x01, *range(0xD0, 0xD8)}:
            continue
        size = int.from_bytes(raw[offset:offset + 2], "big")
        if size < 2 or offset + size > len(raw):
            break
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF} and size >= 8:
            height = int.from_bytes(raw[offset + 3:offset + 5], "big")
            width = int.from_bytes(raw[offset + 5:offset + 7], "big")
            if min(width, height) < 1 or min(width, height) > 1080 or max(width, height) > 1920:
                raise ValidationError("Qwen JPEG dimensions must not exceed 1080p.")
            return
        offset += size
    raise ValidationError("Qwen image has no valid JPEG frame dimensions.")


class QwenLiveTranslateSession:
    """Native session. Consume streams concurrently; each retains 1,000 events.

    `finish`, `aclose`, and a final AudioFrame drain the final translation before
    disconnecting. Native events preserve speaker and item metadata verbatim.
    """

    provider = "qwen"
    model_id = MODEL_ID
    capabilities = CAPABILITIES

    def __init__(self, connection: RealtimeConnection, config: RealtimeSessionConfig, *, timeout_ms: int = 30_000) -> None:
        self.config = deepcopy(config)
        self._connection = connection
        self._timeout = timeout_ms / 1000
        self._events: Broadcast[RealtimeEvent] = Broadcast(max_events=1000)
        self._raw: Broadcast[dict[str, Any]] = Broadcast(max_events=1000)
        self._receiver: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._update_ack: asyncio.Future[None] | None = None
        self._lock = asyncio.Lock()
        self._finishing = False
        self._closed = False
        self._finished = False
        self._failure: Exception | None = None
        self._audio_sent = False
        self._last_image = float("-inf")
        self._items: OrderedDict[str, str | None] = OrderedDict()
        self._transcripts: dict[str, str] = {}

    async def _read(self) -> dict[str, Any]:
        payload = await self._connection.recv_json()
        if not isinstance(payload, dict) or not isinstance(payload.get("type"), str):
            raise ConnectionError("Qwen connection ended or returned an invalid event.")
        if len(json.dumps(payload).encode()) > MAX_MESSAGE_BYTES:
            raise ValidationError("Qwen realtime event exceeds 1 MiB.")
        await self._raw.publish(payload)
        if payload["type"] == "error" or payload.get("code"):
            # Preserve details in native events, not exception strings.
            raise ConnectionError("Qwen rejected the realtime request; inspect the native error event.")
        return payload

    async def initialize(self) -> None:
        try:
            async with asyncio.timeout(self._timeout):
                created = await self._read()
                if created["type"] != "session.created":
                    raise ConnectionError("Expected Qwen session.created.")
                await self._connection.send_json({"type": "session.update", "session": session_payload(self.config)})
                updated = await self._read()
                if updated["type"] != "session.updated":
                    raise ConnectionError("Expected Qwen session.updated.")
                session = updated.get("session") or created.get("session") or {}
                await self._events.publish(RealtimeSessionStartedEvent(session_id=session.get("id"), provider_metadata=updated))
            self._receiver = asyncio.create_task(self._receive())
        except BaseException:
            await self._cleanup()
            raise

    def _ensure_open(self) -> None:
        if self._closed or self._finishing or self._failure or self._finished:
            raise ValidationError("Qwen realtime session is closed or finishing.")

    async def _send(self, payload: dict[str, Any]) -> None:
        try:
            async with asyncio.timeout(self._timeout):
                await self._connection.send_json(payload)
        except Exception as error:
            self._failure = error
            await self._cleanup()
            raise
        except asyncio.CancelledError:
            await self._cleanup()
            raise

    async def send_audio(self, frame: AudioFrame) -> None:
        if frame.media_type != "audio/pcm" or frame.sample_rate_hz not in (None, 16000) or frame.channels not in (None, 1):
            raise ValidationError("Qwen input must be mono PCM16 at 16000 Hz (audio/pcm).")
        data = _bytes(frame.data)
        if len(data) % 2 or len(data) > 512_000 or data.startswith(b"RIFF"):
            raise ValidationError("Send raw PCM16 chunks up to 512000 bytes, without WAV headers.")
        async with self._lock:
            self._ensure_open()
            if data:
                await self._send({"type": "input_audio_buffer.append", "audio": base64.b64encode(data).decode("ascii")})
                self._audio_sent = True
        if frame.is_final:
            await self.finish()

    async def send_image(self, data: bytes | bytearray | memoryview | str, *, media_type: str = "image/jpeg") -> None:
        raw = _bytes(data)
        if media_type != "image/jpeg" or not raw.startswith(b"\xff\xd8\xff") or len(raw) > 500_000:
            raise ValidationError("Qwen image input must be JPEG up to 500000 bytes.")
        _validate_jpeg(raw)
        async with self._lock:
            self._ensure_open()
            if not self._audio_sent:
                raise ValidationError("Send audio before the first Qwen image.")
            now = asyncio.get_running_loop().time()
            if now - self._last_image < 0.5:
                raise ValidationError("Qwen accepts at most two images per second.")
            await self._send({"type": "input_image_buffer.append", "image": base64.b64encode(raw).decode("ascii")})
            self._last_image = now

    async def clear_audio(self) -> None:
        async with self._lock:
            self._ensure_open()
            await self._send({"type": "input_audio_buffer.clear"})

    async def send_text(self, text: str) -> None:
        raise UnsupportedFeatureError("Qwen LiveTranslate accepts audio and optional images, not text prompts.")

    async def send_tool_result(self, result: ToolExecutionResult) -> None:
        raise UnsupportedFeatureError("Qwen LiveTranslate does not execute tools.")

    async def update(self, *, instructions: str | None = None, voice: str | None = None,
                     tools: dict[str, Any] | None = None, tool_choice: Any = None,
                     turn_detection: dict[str, Any] | None = None,
                     provider_options: dict[str, Any] | None = None,
                     translation_target_language_code: str | None = None) -> None:
        async with self._lock:
            self._ensure_open()
            changes = {key: value for key, value in {
                "instructions": instructions, "voice": voice, "tools": tools,
                "tool_choice": tool_choice, "turn_detection": turn_detection,
                "provider_options": provider_options,
                "translation_target_language_code": translation_target_language_code,
            }.items() if value is not None}
            if provider_options is not None:
                changes["provider_options"] = _merge_options(
                    self.config.provider_options or {}, _object(provider_options, "provider_options"),
                )
            next_config = replace(self.config, **changes)
            payload = session_payload(next_config)
            self._update_ack = asyncio.get_running_loop().create_future()
            try:
                async with asyncio.timeout(self._timeout):
                    await self._connection.send_json({"type": "session.update", "session": payload})
                    await self._update_ack
                self.config = deepcopy(next_config)
            except BaseException:
                # A timed out update has unknown remote state. Do not continue.
                if self._update_ack is not None and not self._update_ack.done():
                    self._update_ack.cancel()
                await self._cleanup()
                raise
            finally:
                self._update_ack = None

    def event_stream(self) -> AsyncIterable[RealtimeEvent]:
        return self._events.stream()

    def native_event_stream(self) -> AsyncIterable[dict[str, Any]]:
        return self._raw.stream()

    def _metadata(self, payload: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(payload)
        item_id = str(payload.get("item_id") or "")
        if item_id in self._items:
            metadata.setdefault("previous_item_id", self._items[item_id])
        return metadata

    async def _parse(self, payload: dict[str, Any]) -> None:
        kind = payload["type"]
        item_id, response_id = payload.get("item_id"), payload.get("response_id")
        metadata = self._metadata(payload)
        if kind == "session.updated":
            if self._update_ack is not None and not self._update_ack.done():
                self._update_ack.set_result(None)
        elif kind == "conversation.item.created":
            item = payload.get("item", {})
            key = str(item.get("id") or "")
            self._items[key] = payload.get("previous_item_id")
            if len(self._items) > 1000:
                self._items.popitem(last=False)
        elif kind in {"response.text.delta", "response.audio_transcript.delta"}:
            await self._events.publish(RealtimeTextDeltaEvent(text_delta=str(payload.get("delta") or ""), item_id=item_id, response_id=response_id, provider_metadata=metadata))
        elif kind == "conversation.item.input_audio_transcription.delta":
            key = str(item_id or "")
            text = self._transcripts.get(key, "") + str(payload.get("delta") or "")
            if len(text) > 262144 or len(self._transcripts) >= 1000 and key not in self._transcripts:
                raise ValidationError("Qwen source transcript retention limit exceeded.")
            self._transcripts[key] = text
            await self._events.publish(RealtimeTranscriptEvent(text=text, role="user", item_id=item_id, provider_metadata=metadata))
        elif kind in {"conversation.item.input_audio_transcription.completed", "response.text.done", "response.audio_transcript.done"}:
            source = kind.startswith("conversation.")
            self._transcripts.pop(str(item_id or ""), None)
            await self._events.publish(RealtimeTranscriptEvent(text=str(payload.get("transcript", payload.get("text", ""))), role="user" if source else "assistant", is_final=True, item_id=item_id, response_id=response_id, provider_metadata=metadata))
        elif kind == "response.audio.delta":
            data = _bytes(payload.get("delta", ""))
            if len(data) % 2:
                raise ValidationError("Invalid Qwen PCM16 output.")
            await self._events.publish(RealtimeAudioOutputEvent(audio=data, media_type="audio/pcm", sample_rate_hz=24000, channels=1, item_id=item_id, response_id=response_id, provider_metadata=metadata))
        elif kind == "conversation.item.input_audio_transcription.failed":
            raise ConnectionError("Qwen source transcription failed; inspect native events.")
        elif kind == "response.done":
            status = payload.get("response", {}).get("status")
            if status != "completed":
                raise ConnectionError("Qwen translation response did not complete successfully.")
            await self._events.publish(RealtimeResponseCompletedEvent(reason=status, provider_metadata=payload))
        elif kind == "session.finished":
            self._finished = True
            await self._events.publish(RealtimeSessionEndedEvent(reason="session.finished", provider_metadata=payload))

    async def _receive(self) -> None:
        try:
            while not self._finished:
                await self._parse(await self._read())
        except Exception as error:
            self._failure = error
            await self._events.publish(RealtimeErrorEvent(error=error, message=str(error)))
            await self._events.publish(RealtimeSessionEndedEvent(reason="error"))
        finally:
            if self._update_ack is not None and not self._update_ack.done():
                self._update_ack.set_exception(ConnectionError("Qwen session ended during update."))
            await self._events.close()
            await self._raw.close()
            # The receiver owns transport cleanup even when the user only consumes.
            try:
                async with asyncio.timeout(5):
                    await self._connection.close()
            except Exception:
                pass
            self._closed = True

    async def _cleanup(self) -> None:
        self._closed = True
        if self._receiver is not None and not self._receiver.done():
            self._receiver.cancel()
            await asyncio.gather(self._receiver, return_exceptions=True)
        try:
            async with asyncio.timeout(5):
                await self._connection.close()
        finally:
            await self._events.close()
            await self._raw.close()

    async def _finish(self) -> None:
        try:
            async with asyncio.timeout(self._timeout):
                async with self._lock:
                    if not self._closed and not self._finished:
                        self._finishing = True
                        await self._connection.send_json({"type": "session.finish"})
                if self._receiver is not None and not self._receiver.done():
                    await asyncio.shield(self._receiver)
                if self._failure is not None:
                    raise self._failure
                if not self._finished:
                    raise ConnectionError("Qwen closed without session.finished.")
        except BaseException:
            await self._cleanup()
            raise

    async def finish(self) -> None:
        if self._close_task is None:
            # Reject new writes as soon as finish is requested, even if the
            # finisher must wait for an already pending update or audio send.
            self._finishing = True
            self._close_task = asyncio.create_task(self._finish())
        try:
            await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            self._close_task.cancel()
            await asyncio.gather(self._close_task, return_exceptions=True)
            raise

    async def aclose(self) -> None:
        await self.finish()


class QwenLiveTranslateModel:
    provider = "qwen"
    capabilities = CAPABILITIES

    def __init__(self, model_id: str, *, api_key: str, base_url: str, region: str,
                 workspace_id: str | None = None, realtime_endpoint: str | None = None,
                 connection_factory: RealtimeConnectionFactory | None = None) -> None:
        if model_id != MODEL_ID:
            raise UnsupportedFeatureError(f"Qwen realtime currently supports only {MODEL_ID}.")
        self.model_id = model_id
        self._api_key = api_key
        self._url = realtime_url(base_url, region, workspace_id, realtime_endpoint)
        self._factory = connection_factory

    async def connect(self, config: RealtimeSessionConfig | None = None,
                      options: RealtimeConnectOptions | None = None) -> QwenLiveTranslateSession:
        config = deepcopy(config) if config is not None else RealtimeSessionConfig()
        session_payload(config)
        if options and options.browser_client:
            raise UnsupportedFeatureError("Qwen LiveTranslate uses server-side Bearer authentication; browser tokens are not supported.")
        timeout_ms = options.timeout_ms if options and options.timeout_ms is not None else 30_000
        if type(timeout_ms) is not int or timeout_ms <= 0:
            raise ValidationError("Qwen realtime timeout_ms must be a positive integer.")
        headers = {"Authorization": f"Bearer {self._api_key}"}
        async with asyncio.timeout(timeout_ms / 1000):
            connection = (await self._factory(self._url, headers, options) if self._factory
                          else await open_websocket_connection(self._url, headers=headers, options=options))
        session = QwenLiveTranslateSession(connection, config, timeout_ms=timeout_ms)
        await session.initialize()
        return session

    async def create_browser_token(self, config: RealtimeSessionConfig | None = None,
                                   options: RealtimeConnectOptions | None = None) -> RealtimeTokenResult:
        raise UnsupportedFeatureError("Qwen LiveTranslate does not expose browser tokens.")
