from __future__ import annotations

import base64
import json
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (
    FilePart,
    ImagePart,
    MCPServerConfig,
    MCPToolConfig,
    ToolChoiceName,
    create_gemini,
    create_vertex,
    embed_content,
    gemini_code_execution_tool,
    gemini_computer_use_tool,
    gemini_file_search_tool,
    gemini_google_search_tool,
    generate_grounded_text,
    generate_speech,
    generate_text,
    hosted_tool,
    stream_text,
    tool,
    vertex_external_search_tool,
    vertex_google_maps_tool,
    vertex_google_search_tool,
    vertex_vertex_ai_search_tool,
)
from zhivex_ai import UnsupportedFeatureError
from zhivex_ai.types import ModelGenerateInput, ModelMessage, StructuredOutputConfig, TextPart
from zhivex_ai.errors import ProviderHTTPError
from zhivex_ai.errors import ValidationError


@dataclass
class FakeResponse:
    status_code: int
    payload: Any = None
    body_text: str = ""
    headers: dict[str, str] | None = None

    async def json(self) -> Any:
        return self.payload

    async def text(self) -> str:
        return self.body_text or json.dumps(self.payload)

    async def iter_lines(self):
        for line in self.body_text.splitlines():
            yield line


class GeminiProviderTests(IsolatedAsyncioTestCase):
    async def test_gemini_image_generation_client_uses_generate_content(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append({"url": url, "json": json_body})
            return FakeResponse(
                status_code=200,
                payload={
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "done"},
                                    {"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="}},
                                ]
                            }
                        }
                    ]
                },
            )

        provider = create_gemini(api_key="test", fetch=fetch)
        result = await provider.images().generate(prompt="draw a transit map", model="gemini-3.1-flash-image-preview")

        self.assertEqual(result.images[0].b64_json, "iVBORw0KGgo=")
        self.assertEqual(result.images[0].media_type, "image/png")
        self.assertIn("/models/gemini-3.1-flash-image-preview:generateContent?key=test", requests[0]["url"])
        self.assertEqual(requests[0]["json"]["generationConfig"]["responseModalities"], ["IMAGE"])

    async def test_gemini_imagen_generation_client_uses_predict(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append({"url": url, "json": json_body})
            return FakeResponse(
                status_code=200,
                payload={
                    "predictions": [
                        {"bytesBase64Encoded": "img-b64", "mimeType": "image/png", "revisedPrompt": "better prompt"}
                    ]
                },
            )

        provider = create_gemini(api_key="test", fetch=fetch)
        result = await provider.images().generate(
            prompt="Robot holding a red skateboard",
            model="imagen-4.0-generate-001",
            size="2K",
            extra_body={"parameters": {"sampleCount": 2, "aspectRatio": "16:9"}},
        )

        self.assertEqual(result.images[0].b64_json, "img-b64")
        self.assertEqual(result.images[0].revised_prompt, "better prompt")
        self.assertIn("/models/imagen-4.0-generate-001:predict?key=test", requests[0]["url"])
        self.assertEqual(requests[0]["json"]["parameters"]["sampleCount"], 2)
        self.assertEqual(requests[0]["json"]["parameters"]["aspectRatio"], "16:9")
        self.assertEqual(requests[0]["json"]["parameters"]["imageSize"], "2K")

    async def test_gemini_embedding_2_accepts_multimodal_content(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append({"url": url, "json": json_body})
            return FakeResponse(status_code=200, payload={"embedding": {"values": [0.1, 0.2, 0.3]}})

        provider = create_gemini(api_key="test", fetch=fetch)
        result = await embed_content(
            model=provider.native.embedding_model("gemini-embedding-2"),
            value=[
                TextPart(text="Find similar visual docs"),
                FilePart(data="JVBERi0xLjQK", media_type="application/pdf", filename="doc.pdf"),
                ImagePart(image="data:image/png;base64,aGVsbG8="),
            ],
        )

        self.assertEqual(result.embeddings, [[0.1, 0.2, 0.3]])
        parts = requests[0]["json"]["content"]["parts"]
        self.assertEqual(parts[0]["text"], "Find similar visual docs")
        self.assertEqual(parts[1]["inlineData"]["mimeType"], "application/pdf")
        self.assertEqual(parts[2]["inlineData"]["mimeType"], "image/png")
        self.assertIn("/models/gemini-embedding-2:embedContent?key=test", requests[0]["url"])

    async def test_gemini_video_client_creates_and_polls_operation(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str,
            *,
            headers: dict[str, str],
            json_body: dict[str, Any] | None = None,
            timeout_ms: int | None,
            stream: bool = False,
            method: str = "POST",
            body: Any = None,
        ):
            requests.append({"url": url, "method": method, "json": json_body})
            if method == "GET":
                return FakeResponse(
                    status_code=200,
                    payload={
                        "name": "operations/video-1",
                        "done": True,
                        "response": {
                            "generateVideoResponse": {
                                "generatedSamples": [{"video": {"uri": "https://download.example.com/video.mp4"}}]
                            }
                        },
                    },
                )
            return FakeResponse(status_code=200, payload={"name": "operations/video-1", "done": False})

        provider = create_gemini(api_key="test", fetch=fetch)
        operation = await provider.videos().generate(
            model="veo-3.1-generate-preview",
            prompt="A cinematic lion shot.",
            config={"aspectRatio": "16:9"},
        )
        waited = await provider.videos().wait_operation(operation.name, poll_interval_ms=1, timeout_ms=100)

        self.assertEqual(operation.name, "operations/video-1")
        self.assertTrue(waited.done)
        self.assertIn(":predictLongRunning?key=test", requests[0]["url"])
        self.assertEqual(requests[0]["json"]["parameters"]["aspectRatio"], "16:9")
        self.assertEqual(waited.raw_response["generated_media"][0].url, "https://download.example.com/video.mp4")

    async def test_gemini_media_client_generates_lyria_audio(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append({"url": url, "json": json_body})
            return FakeResponse(
                status_code=200,
                payload={
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "lyrics"},
                                    {"inlineData": {"mimeType": "audio/mpeg", "data": "mp3-b64"}},
                                ]
                            }
                        }
                    ]
                },
            )

        provider = create_gemini(api_key="test", fetch=fetch)
        result = await provider.media().generate_music(prompt="Create a short folk song.")

        self.assertEqual(result.text, "lyrics")
        self.assertEqual(result.media[0].b64_data, "mp3-b64")
        self.assertEqual(result.media[0].media_type, "audio/mpeg")
        self.assertIn("/models/lyria-3-clip-preview:generateContent?key=test", requests[0]["url"])

    async def test_gemini_batches_and_interactions_clients(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str,
            *,
            headers: dict[str, str],
            json_body: dict[str, Any] | None = None,
            timeout_ms: int | None,
            stream: bool = False,
            method: str = "POST",
            body: Any = None,
        ):
            requests.append({"url": url, "method": method, "json": json_body, "stream": stream})
            if "batchEmbedContents" in url:
                return FakeResponse(status_code=200, payload={"name": "batches/embed-1", "done": False})
            if "/interactions?" in url and method == "POST":
                return FakeResponse(status_code=200, payload={"id": "int-1", "status": "in_progress"})
            if "/interactions/int-1?" in url:
                return FakeResponse(status_code=200, payload={"id": "int-1", "status": "completed"})
            raise AssertionError(f"Unexpected request: {method} {url}")

        provider = create_gemini(api_key="test", fetch=fetch)
        batch = await provider.batches().create_embeddings(
            {"model": "gemini-embedding-001", "batch": {"displayName": "embeddings"}}
        )
        interaction = await provider.interactions().create(
            {"input": "Research TPUs", "agent": "deep-research-pro-preview-12-2025"}
        )
        waited = await provider.interactions().wait("int-1", poll_interval_ms=1, timeout_ms=100)

        self.assertEqual(batch["name"], "batches/embed-1")
        self.assertEqual(interaction["id"], "int-1")
        self.assertEqual(waited["status"], "completed")
        self.assertTrue(requests[1]["json"]["background"])
        self.assertTrue(requests[1]["json"]["store"])

    async def test_vertex_media_clients_use_bearer_and_publisher_routes(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str,
            *,
            headers: dict[str, str],
            json_body: dict[str, Any] | None = None,
            timeout_ms: int | None,
            stream: bool = False,
            method: str = "POST",
            body: Any = None,
        ):
            requests.append({"url": url, "method": method, "headers": headers, "json": json_body})
            if "predictLongRunning" in url:
                return FakeResponse(status_code=200, payload={"name": "operations/vertex-video", "done": False})
            return FakeResponse(
                status_code=200,
                payload={
                    "candidates": [
                        {"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": "vertex-img"}}]}}
                    ]
                },
            )

        provider = create_vertex(access_token="token", project_id="proj", fetch=fetch)
        image = await provider.images().generate(prompt="draw", model="gemini-3.1-flash-image-preview")
        video = await provider.videos().generate(model="veo-3.1-generate-001", prompt="video")

        self.assertEqual(image.images[0].b64_json, "vertex-img")
        self.assertEqual(video.name, "operations/vertex-video")
        self.assertEqual(requests[0]["headers"]["authorization"], "Bearer token")
        self.assertIn("/publishers/google/models/gemini-3.1-flash-image-preview:generateContent", requests[0]["url"])
        self.assertIn("/publishers/google/models/veo-3.1-generate-001:predictLongRunning", requests[1]["url"])

    async def test_gemini_generates_speech(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append({"url": url, "json": json_body})
            return FakeResponse(
                status_code=200,
                payload={
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "inlineData": {
                                            "mimeType": "audio/pcm",
                                            "data": base64.b64encode(b"voice-bytes").decode("ascii"),
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                },
            )

        provider = create_gemini(api_key="test", fetch=fetch)
        result = await generate_speech(
            model=provider.speech_model("gemini-2.5-flash-preview-tts"),
            input="hello",
        )

        self.assertEqual(result.audio, b"voice-bytes")
        self.assertEqual(result.media_type, "audio/pcm")
        self.assertEqual(
            requests[0]["json"]["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"],
            "Kore",
        )
        self.assertEqual(requests[0]["json"]["generationConfig"]["responseModalities"], ["AUDIO"])

    async def test_gemini_creates_browser_token(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str,
            *,
            headers: dict[str, str],
            json_body: dict[str, Any] | None,
            timeout_ms: int | None,
            stream: bool = False,
            method: str = "POST",
            body: Any = None,
        ):
            requests.append({"url": url, "method": method, "headers": headers, "json": json_body})
            return FakeResponse(
                status_code=200,
                payload={
                    "name": "ephemeral-token",
                    "expireTime": "2026-04-12T00:00:00Z",
                },
            )

        provider = create_gemini(api_key="test", fetch=fetch)
        token = await provider.realtime_model("gemini-live-2.5-flash").create_browser_token()

        self.assertEqual(token.value, "ephemeral-token")
        self.assertIsNotNone(token.expires_at_ms)
        self.assertEqual(requests[0]["method"], "POST")
        self.assertIn("/v1alpha/authTokens?key=test", requests[0]["url"])

    async def test_vertex_realtime_disables_browser_token_capability(self) -> None:
        provider = create_vertex(access_token="token", project_id="proj", fetch=lambda **_: None)  # type: ignore[arg-type]
        model = provider.realtime_model("gemini-live-2.5-flash")
        self.assertFalse(model.capabilities.realtime_browser_tokens)

    async def test_vertex_generates_speech(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str,
            *,
            headers: dict[str, str],
            json_body: dict[str, Any],
            timeout_ms: int | None,
            stream: bool = False,
            method: str = "POST",
            body: Any = None,
        ):
            requests.append({"url": url, "headers": headers, "json": json_body})
            return FakeResponse(
                status_code=200,
                payload={
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "inlineData": {
                                            "mimeType": "audio/pcm",
                                            "data": base64.b64encode(b"vertex-voice").decode("ascii"),
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                },
            )

        provider = create_vertex(access_token="token", project_id="proj", fetch=fetch)
        result = await generate_speech(
            model=provider.speech_model("gemini-2.5-flash-tts"),
            input="hello",
            voice="Puck",
        )

        self.assertEqual(result.audio, b"vertex-voice")
        self.assertEqual(result.media_type, "audio/pcm")
        self.assertIn("/publishers/google/models/gemini-2.5-flash-tts:generateContent", requests[0]["url"])
        self.assertEqual(requests[0]["headers"]["authorization"], "Bearer token")
        self.assertEqual(
            requests[0]["json"]["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"],
            "Puck",
        )

    async def test_gemini_maps_tool_choice_and_usage(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            return FakeResponse(
                status_code=200,
                payload={
                    "candidates": [{"content": {"parts": [{"text": "sunny"}]}, "finishReason": "STOP"}],
                    "usageMetadata": {
                        "promptTokenCount": 11,
                        "candidatesTokenCount": 7,
                        "totalTokenCount": 18,
                    },
                },
            )

        provider = create_gemini(api_key="test", fetch=fetch)
        result = await generate_text(
            model=provider.native.language_model("gemini-2.5-flash"),
            prompt="weather",
            tools={"weather": tool(name="weather", schema=dict[str, str], execute=lambda input: {"ok": True})},
            tool_choice=ToolChoiceName(tool_name="weather"),
            structured_output=StructuredOutputConfig(schema=dict[str, str], mode="native"),
        )

        self.assertEqual(result.text, "sunny")
        self.assertEqual(result.usage.total_tokens, 18)
        self.assertEqual(
            requests[0]["toolConfig"]["functionCallingConfig"],
            {"mode": "ANY", "allowedFunctionNames": ["weather"]},
        )
        self.assertEqual(requests[0]["generationConfig"]["responseMimeType"], "application/json")
        self.assertIn("responseJsonSchema", requests[0]["generationConfig"])

    async def test_gemini_omits_null_fields_and_strips_data_url_prefix(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            return FakeResponse(
                status_code=200,
                payload={
                    "candidates": [{"content": {"parts": [{"text": "done"}]}, "finishReason": "STOP"}],
                },
            )

        provider = create_gemini(api_key="test", fetch=fetch)
        model = provider("gemini-2.5-flash")
        await model.generate(
            ModelGenerateInput(
                messages=[
                    ModelMessage(
                        role="user",
                        parts=[
                            ImagePart(image="data:image/png;base64,aGVsbG8="),
                            TextPart(text="describe"),
                        ],
                    )
                ]
            )
        )

        inline_data = requests[0]["contents"][0]["parts"][0]["inlineData"]
        self.assertEqual(inline_data["mimeType"], "image/png")
        self.assertEqual(inline_data["data"], "aGVsbG8=")
        self.assertNotIn("toolConfig", requests[0])
        self.assertNotIn("generationConfig", requests[0])

    async def test_gemini_maps_inline_pdf_file_input(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            return FakeResponse(
                status_code=200,
                payload={"candidates": [{"content": {"parts": [{"text": "done"}]}, "finishReason": "STOP"}]},
            )

        provider = create_gemini(api_key="test", fetch=fetch)
        await generate_text(
            model=provider("gemini-2.5-flash"),
            messages=[ModelMessage(role="user", parts=[FilePart(data="JVBERi0xLjQK", media_type="application/pdf", filename="stub.pdf")])],
        )

        inline_data = requests[0]["contents"][0]["parts"][0]["inlineData"]
        self.assertEqual(inline_data["mimeType"], "application/pdf")
        self.assertEqual(inline_data["data"], "JVBERi0xLjQK")

    async def test_gemini_maps_file_uri_pdf_input(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            return FakeResponse(
                status_code=200,
                payload={"candidates": [{"content": {"parts": [{"text": "done"}]}, "finishReason": "STOP"}]},
            )

        provider = create_gemini(api_key="test", fetch=fetch)
        await generate_text(
            model=provider("gemini-2.5-flash"),
            messages=[ModelMessage(role="user", parts=[FilePart(file_uri="files/123")])],
        )

        file_data = requests[0]["contents"][0]["parts"][0]["fileData"]
        self.assertEqual(file_data["fileUri"], "files/123")
        self.assertNotIn("mimeType", file_data)

    async def test_gemini_rejects_file_id_pdf_input(self) -> None:
        provider = create_gemini(api_key="test", fetch=lambda **_: None)  # type: ignore[arg-type]
        with self.assertRaises(ValidationError):
            await generate_text(
                model=provider("gemini-2.5-flash"),
                messages=[ModelMessage(role="user", parts=[FilePart(file_id="file_123")])],
            )

    async def test_gemini_maps_google_search_provider_option(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            return FakeResponse(
                status_code=200,
                payload={
                    "candidates": [{"content": {"parts": [{"text": "fresh answer"}]}, "finishReason": "STOP"}],
                },
            )

        provider = create_gemini(api_key="test", fetch=fetch)
        result = await generate_text(
            model=provider.native.language_model("gemini-2.5-flash"),
            prompt="latest news",
            provider_options={"google_search": True},
        )

        self.assertEqual(result.text, "fresh answer")
        self.assertEqual(requests[0]["tools"], [{"googleSearch": {}}])
        self.assertNotIn("google_search", requests[0])

    async def test_gemini_maps_builtin_tools_and_strips_provider_options(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            return FakeResponse(
                status_code=200,
                payload={"candidates": [{"content": {"parts": [{"text": "done"}]}, "finishReason": "STOP"}]},
            )

        provider = create_gemini(api_key="test", fetch=fetch)
        await generate_text(
            model=provider.native.language_model("gemini-3.1-flash-preview"),
            prompt="Research this",
            provider_options={
                "google_search": {"excludeDomains": ["example.com"]},
                "code_execution": True,
                "built_in_tools": [{"url_context": {}}],
            },
        )

        self.assertEqual(
            requests[0]["tools"],
            [
                {"googleSearch": {"excludeDomains": ["example.com"]}},
                {"codeExecution": {}},
                {"urlContext": {}},
            ],
        )
        self.assertNotIn("google_search", requests[0])
        self.assertNotIn("code_execution", requests[0])
        self.assertNotIn("built_in_tools", requests[0])

    async def test_gemini_hosted_tool_builder_is_exported(self) -> None:
        tool = gemini_google_search_tool()
        self.assertEqual(tool.type, "google_search")
        self.assertEqual(tool.tool_class, "web-search")
        self.assertEqual(tool.config, {})

    async def test_gemini_maps_hosted_tools_from_tools_set(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            return FakeResponse(
                status_code=200,
                payload={
                    "candidates": [{"content": {"parts": [{"text": "done"}]}, "finishReason": "STOP"}],
                },
            )

        provider = create_gemini(api_key="test", fetch=fetch)
        await generate_text(
            model=provider.native.language_model("gemini-2.5-flash"),
            prompt="hello",
            tools={
                "weather": tool(name="weather", schema=dict[str, str], execute=lambda input: {"ok": True}),
                "search": gemini_google_search_tool(),
                "code": gemini_code_execution_tool(),
            },
        )

        self.assertEqual(requests[0]["tools"][0]["functionDeclarations"][0]["name"], "weather")
        self.assertEqual(requests[0]["tools"][1], {"googleSearch": {}})
        self.assertEqual(requests[0]["tools"][2], {"codeExecution": {}})

    async def test_gemini_maps_file_search_hosted_tool_from_tools_set(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            return FakeResponse(
                status_code=200,
                payload={
                    "candidates": [{"content": {"parts": [{"text": "done"}]}, "finishReason": "STOP"}],
                },
            )

        provider = create_gemini(api_key="test", fetch=fetch)
        await generate_text(
            model=provider.native.language_model("gemini-2.5-flash"),
            prompt="hello",
            tools={
                "files": gemini_file_search_tool(file_search_store_names=["fileSearchStores/alpha"]),
            },
        )

        self.assertEqual(
            requests[0]["tools"][0],
            {"fileSearch": {"fileSearchStoreNames": ["fileSearchStores/alpha"]}},
        )

    async def test_gemini_rejects_combining_file_search_with_function_tools(self) -> None:
        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            raise AssertionError("request should not be dispatched")

        provider = create_gemini(api_key="test", fetch=fetch)
        with self.assertRaises(UnsupportedFeatureError) as context:
            await generate_text(
                model=provider.native.language_model("gemini-2.5-flash"),
                prompt="hello",
                tools={
                    "lookup": tool(name="lookup", schema=dict[str, str], execute=lambda input: input),
                    "files": gemini_file_search_tool(file_search_store_names=["fileSearchStores/alpha"]),
                },
            )

        self.assertIn('combining "file_search" with other tools', str(context.exception))

    async def test_gemini_maps_computer_use_hosted_tool_from_tools_set(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            return FakeResponse(
                status_code=200,
                payload={
                    "candidates": [{"content": {"parts": [{"text": "done"}]}, "finishReason": "STOP"}],
                },
            )

        provider = create_gemini(api_key="test", fetch=fetch)
        await generate_text(
            model=provider.native.language_model("gemini-2.5-flash"),
            prompt="hello",
            tools={
                "computer": gemini_computer_use_tool(display_name="Browser"),
            },
        )

        self.assertEqual(requests[0]["tools"][0], {"computerUse": {"display_name": "Browser"}})

    async def test_gemini_rejects_combining_computer_use_with_other_tools(self) -> None:
        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            raise AssertionError("request should not be dispatched")

        provider = create_gemini(api_key="test", fetch=fetch)
        with self.assertRaises(UnsupportedFeatureError) as context:
            await generate_text(
                model=provider.native.language_model("gemini-2.5-flash"),
                prompt="hello",
                tools={
                    "search": gemini_google_search_tool(),
                    "computer": gemini_computer_use_tool(display_name="Browser"),
                },
            )

        self.assertIn('combining "computer_use" with other tools', str(context.exception))

    async def test_vertex_rejects_duplicate_computer_use_tool_declarations(self) -> None:
        async def fetch(
            url: str,
            *,
            headers: dict[str, str],
            json_body: dict[str, Any],
            timeout_ms: int | None,
            stream: bool = False,
            method: str = "POST",
            body: Any = None,
        ):
            raise AssertionError("request should not be dispatched")

        provider = create_vertex(access_token="token", project_id="proj", fetch=fetch)
        with self.assertRaises(UnsupportedFeatureError) as context:
            await generate_text(
                model=provider.native.language_model("gemini-2.5-flash"),
                prompt="hello",
                tools={
                    "computer_a": hosted_tool(
                        name="computer_a",
                        provider="vertex",
                        type="computer_use",
                        tool_class="computer-use",
                        config={"display_name": "Browser A"},
                    ),
                    "computer_b": hosted_tool(
                        name="computer_b",
                        provider="vertex",
                        type="computer_use",
                        tool_class="computer-use",
                        config={"display_name": "Browser B"},
                    ),
                },
            )

        self.assertIn('declaring "computerUse" more than once', str(context.exception))

    async def test_gemini_grounded_language_model_returns_sources(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            return FakeResponse(
                status_code=200,
                payload={
                    "candidates": [
                        {
                            "content": {"parts": [{"text": "grounded answer"}]},
                            "finishReason": "STOP",
                            "groundingMetadata": {
                                "groundingChunks": [
                                    {"web": {"uri": "https://example.com/1", "title": "Example 1"}},
                                    {"web": {"uri": "https://example.com/2", "title": "Example 2", "text": "snippet"}},
                                ]
                            },
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 5,
                        "candidatesTokenCount": 3,
                        "totalTokenCount": 8,
                    },
                },
            )

        provider = create_gemini(api_key="test", fetch=fetch)
        result = await generate_grounded_text(
            model=provider.grounded_language_model("gemini-2.5-flash"),
            prompt="latest news",
        )

        self.assertEqual(result.text, "grounded answer")
        self.assertEqual(result.sources[0].url, "https://example.com/1")
        self.assertEqual(result.sources[1].snippet, "snippet")
        self.assertEqual(result.queries, [])
        self.assertEqual(requests[0]["tools"], [{"googleSearch": {}}])

    async def test_vertex_grounded_language_model_returns_sources(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str,
            *,
            headers: dict[str, str],
            json_body: dict[str, Any],
            timeout_ms: int | None,
            stream: bool = False,
            method: str = "POST",
            body: Any = None,
        ):
            requests.append({"url": url, "headers": headers, "json": json_body})
            return FakeResponse(
                status_code=200,
                payload={
                    "candidates": [
                        {
                            "content": {"parts": [{"text": "vertex grounded"}]},
                            "finishReason": "STOP",
                            "groundingMetadata": {
                                "webSearchQueries": ["latest ai infra"],
                                "groundingChunks": [
                                    {"web": {"uri": "https://example.com/vertex", "title": "Vertex Source", "text": "snippet"}}
                                ],
                                "groundingSupports": [
                                    {
                                        "segment": {"startIndex": 0, "endIndex": 14, "text": "vertex grounded"},
                                        "groundingChunkIndices": [0],
                                    }
                                ],
                            },
                        }
                    ]
                },
            )

        provider = create_vertex(access_token="token", project_id="proj", fetch=fetch)
        result = await generate_grounded_text(
            model=provider.grounded_language_model("gemini-2.5-flash"),
            prompt="latest news",
        )

        self.assertEqual(result.text, "vertex grounded")
        self.assertEqual(result.sources[0].url, "https://example.com/vertex")
        self.assertEqual(result.queries, ["latest ai infra"])
        self.assertEqual(result.supports[0].source_indices, [0])
        self.assertIn("/publishers/google/models/gemini-2.5-flash:generateContent", requests[0]["url"])
        self.assertEqual(requests[0]["headers"]["authorization"], "Bearer token")

    async def test_vertex_hosted_tool_builders_are_exported(self) -> None:
        google_search = vertex_google_search_tool()
        google_maps = vertex_google_maps_tool(enable_widget=False)
        self.assertEqual(google_search.type, "google_search")
        self.assertEqual(google_maps.type, "google_maps")
        self.assertEqual(google_maps.config, {"enable_widget": False})
        self.assertEqual(
            vertex_vertex_ai_search_tool(datastore="projects/p/locations/global/collections/default_collection/dataStores/docs"),
            {
                "retrieval": {
                    "vertexAiSearch": {
                        "datastore": "projects/p/locations/global/collections/default_collection/dataStores/docs"
                    }
                }
            },
        )
        self.assertEqual(
            vertex_external_search_tool(endpoint="https://search.example.com", api_key="secret"),
            {
                "retrieval": {
                    "externalApi": {
                        "apiSpec": "SIMPLE_SEARCH",
                        "endpoint": "https://search.example.com",
                        "apiAuth": {"apiKeyConfig": {"apiKeyString": "secret"}},
                    }
                }
            },
        )

    async def test_vertex_maps_native_grounding_aliases_to_tools(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str,
            *,
            headers: dict[str, str],
            json_body: dict[str, Any],
            timeout_ms: int | None,
            stream: bool = False,
            method: str = "POST",
            body: Any = None,
        ):
            requests.append({"url": url, "headers": headers, "json": json_body})
            return FakeResponse(
                status_code=200,
                payload={"candidates": [{"content": {"parts": [{"text": "done"}]}, "finishReason": "STOP"}]},
            )

        provider = create_vertex(access_token="token", project_id="proj", fetch=fetch)
        await generate_text(
            model=provider.native.language_model("gemini-2.5-flash"),
            prompt="Ground this",
            provider_options={
                "google_maps": {"enable_widget": False},
                "vertex_ai_search": {
                    "datastore": "projects/p/locations/global/collections/default_collection/dataStores/docs"
                },
                "external_search": {
                    "endpoint": "https://search.example.com",
                    "api_key": "secret",
                },
            },
        )

        self.assertEqual(
            requests[0]["json"]["tools"],
            [
                {"googleMaps": {"enable_widget": False}},
                {
                    "retrieval": {
                        "vertexAiSearch": {
                            "datastore": "projects/p/locations/global/collections/default_collection/dataStores/docs"
                        }
                    }
                },
                {
                    "retrieval": {
                        "externalApi": {
                            "apiSpec": "SIMPLE_SEARCH",
                            "endpoint": "https://search.example.com",
                            "apiAuth": {"apiKeyConfig": {"apiKeyString": "secret"}},
                        }
                    }
                },
            ],
        )
        self.assertNotIn("vertex_ai_search", requests[0]["json"])
        self.assertNotIn("external_search", requests[0]["json"])

    async def test_vertex_grounded_language_model_uses_explicit_grounding_tools_without_forcing_google_search(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str,
            *,
            headers: dict[str, str],
            json_body: dict[str, Any],
            timeout_ms: int | None,
            stream: bool = False,
            method: str = "POST",
            body: Any = None,
        ):
            requests.append({"url": url, "headers": headers, "json": json_body})
            return FakeResponse(
                status_code=200,
                payload={
                    "candidates": [{"content": {"parts": [{"text": "grounded"}]}, "finishReason": "STOP"}],
                },
            )

        provider = create_vertex(access_token="token", project_id="proj", fetch=fetch)
        result = await generate_grounded_text(
            model=provider.native.grounded_language_model("gemini-2.5-flash"),
            prompt="Search my docs",
            provider_options={
                "vertex_ai_search": {
                    "datastore": "projects/p/locations/global/collections/default_collection/dataStores/docs"
                }
            },
        )

        self.assertEqual(result.text, "grounded")
        self.assertEqual(
            requests[0]["json"]["tools"],
            [
                {
                    "retrieval": {
                        "vertexAiSearch": {
                            "datastore": "projects/p/locations/global/collections/default_collection/dataStores/docs"
                        }
                    }
                }
            ],
        )

    async def test_vertex_rejects_invalid_grounding_alias_shape(self) -> None:
        provider = create_vertex(access_token="token", project_id="proj", fetch=lambda **_: None)  # type: ignore[arg-type]
        with self.assertRaises(ValidationError):
            await generate_text(
                model=provider.native.language_model("gemini-2.5-flash"),
                prompt="hello",
                provider_options={"vertex_ai_search": True},
            )

    async def test_gemini_counts_tokens(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str,
            *,
            headers: dict[str, str],
            json_body: dict[str, Any] | None = None,
            timeout_ms: int | None,
            stream: bool = False,
            method: str = "POST",
            body: Any = None,
        ):
            requests.append({"url": url, "method": method, "headers": headers, "json": json_body})
            return FakeResponse(
                status_code=200,
                payload={
                    "totalTokens": 42,
                    "cachedContentTokenCount": 3,
                    "totalBillableCharacters": 128,
                    "promptTokensDetails": [
                        {"modality": "TEXT", "tokenCount": 42, "billableCharacters": 128}
                    ],
                },
            )

        provider = create_gemini(api_key="test", fetch=fetch)
        result = await provider.tokens().count(
            model_id="gemini-2.5-flash",
            prompt="Count me",
            provider_options={"google_search": True},
        )

        self.assertEqual(result.total_tokens, 42)
        self.assertEqual(result.cached_content_token_count, 3)
        self.assertEqual(result.total_billable_characters, 128)
        self.assertEqual(result.details[0].modality, "TEXT")
        self.assertEqual(requests[0]["method"], "POST")
        self.assertIn(":countTokens?key=test", requests[0]["url"])
        self.assertEqual(
            requests[0]["json"]["generateContentRequest"]["tools"],
            [{"googleSearch": {}}],
        )
        self.assertNotIn("google_search", requests[0]["json"]["generateContentRequest"])

    async def test_vertex_counts_tokens(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str,
            *,
            headers: dict[str, str],
            json_body: dict[str, Any] | None = None,
            timeout_ms: int | None,
            stream: bool = False,
            method: str = "POST",
            body: Any = None,
        ):
            requests.append({"url": url, "method": method, "headers": headers, "json": json_body})
            return FakeResponse(
                status_code=200,
                payload={
                    "totalTokens": 27,
                    "promptTokensDetails": [
                        {"modality": "TEXT", "tokenCount": 27}
                    ],
                },
            )

        provider = create_vertex(access_token="token", project_id="proj", fetch=fetch)
        result = await provider.tokens().count(
            model_id="gemini-2.5-flash",
            prompt="Count me too",
            provider_options={"google_search": True},
        )

        self.assertEqual(result.total_tokens, 27)
        self.assertEqual(result.details[0].token_count, 27)
        self.assertEqual(requests[0]["method"], "POST")
        self.assertIn("/publishers/google/models/gemini-2.5-flash:countTokens", requests[0]["url"])
        self.assertEqual(requests[0]["headers"]["authorization"], "Bearer token")
        self.assertEqual(requests[0]["json"]["tools"], [{"googleSearch": {}}])
        self.assertNotIn("google_search", requests[0]["json"])

    async def test_gemini_file_search_store_client_crud_and_operations(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str,
            *,
            headers: dict[str, str],
            json_body: dict[str, Any] | None = None,
            timeout_ms: int | None = None,
            stream: bool = False,
            method: str = "POST",
            body: Any = None,
        ):
            requests.append({"url": url, "method": method, "headers": headers, "json": json_body, "body": body})
            if method == "POST" and "/fileSearchStores?key=" in url:
                return FakeResponse(status_code=200, payload={"name": "fileSearchStores/alpha", "displayName": "Alpha"})
            if method == "GET" and "/fileSearchStores?key=" in url and "pageSize=10" in url:
                return FakeResponse(
                    status_code=200,
                    payload={
                        "fileSearchStores": [{"name": "fileSearchStores/alpha", "displayName": "Alpha"}],
                        "nextPageToken": "next-store",
                    },
                )
            if method == "GET" and "/fileSearchStores/alpha?key=" in url:
                return FakeResponse(status_code=200, payload={"name": "fileSearchStores/alpha", "displayName": "Alpha"})
            if method == "DELETE" and "/fileSearchStores/alpha?key=" in url:
                return FakeResponse(status_code=200, payload={})
            if "upload/v1beta/fileSearchStores/alpha:uploadToFileSearchStore" in url:
                return FakeResponse(status_code=200, payload={}, headers={"x-goog-upload-url": "https://upload.example.com/store"})
            if url == "https://upload.example.com/store":
                return FakeResponse(status_code=200, payload={"name": "operations/upload-1", "done": False})
            if method == "POST" and "/fileSearchStores/alpha:importFile?key=" in url:
                return FakeResponse(status_code=200, payload={"name": "operations/import-1", "done": False})
            if method == "GET" and "/fileSearchStores/alpha/documents?key=" in url:
                return FakeResponse(
                    status_code=200,
                    payload={
                        "documents": [
                            {
                                "name": "fileSearchStores/alpha/documents/doc-1",
                                "displayName": "manual.pdf",
                                "mimeType": "application/pdf",
                                "state": "ACTIVE",
                            }
                        ],
                        "nextPageToken": "next-doc",
                    },
                )
            if method == "GET" and "/fileSearchStores/alpha/documents/doc-1?key=" in url:
                return FakeResponse(
                    status_code=200,
                    payload={
                        "name": "fileSearchStores/alpha/documents/doc-1",
                        "displayName": "manual.pdf",
                        "mimeType": "application/pdf",
                        "state": "ACTIVE",
                    },
                )
            if method == "DELETE" and "/fileSearchStores/alpha/documents/doc-1?key=" in url:
                return FakeResponse(status_code=200, payload={})
            if method == "GET" and "/operations/upload-1?key=" in url:
                poll_count = sum(1 for request in requests if request["method"] == "GET" and "/operations/upload-1?key=" in request["url"])
                return FakeResponse(
                    status_code=200,
                    payload={"name": "operations/upload-1", "done": poll_count >= 2, "response": {"done": True}},
                )
            raise AssertionError(f"Unexpected request: {method} {url}")

        provider = create_gemini(api_key="test", fetch=fetch)
        stores = provider.file_search_stores()

        created = await stores.create(display_name="Alpha")
        listed = await stores.list(page_size=10)
        fetched = await stores.get("fileSearchStores/alpha")
        uploaded = await stores.upload(
            file_search_store_name="fileSearchStores/alpha",
            data=b"%PDF-1.4",
            filename="manual.pdf",
            media_type="application/pdf",
        )
        imported = await stores.import_file(
            file_search_store_name="fileSearchStores/alpha",
            file_name="files/123",
        )
        documents = await stores.list_documents(file_search_store_name="fileSearchStores/alpha")
        document = await stores.get_document("fileSearchStores/alpha/documents/doc-1")
        waited = await stores.wait_operation("operations/upload-1", poll_interval_ms=1, timeout_ms=50)
        deleted_document = await stores.delete_document("fileSearchStores/alpha/documents/doc-1")
        deleted_store = await stores.delete("fileSearchStores/alpha")

        self.assertEqual(created.name, "fileSearchStores/alpha")
        self.assertEqual(listed.stores[0].display_name, "Alpha")
        self.assertEqual(listed.next_page_token, "next-store")
        self.assertEqual(fetched.display_name, "Alpha")
        self.assertEqual(uploaded.name, "operations/upload-1")
        self.assertFalse(uploaded.done)
        self.assertEqual(imported.name, "operations/import-1")
        self.assertEqual(documents.documents[0].media_type, "application/pdf")
        self.assertEqual(documents.next_page_token, "next-doc")
        self.assertEqual(document.state, "ACTIVE")
        self.assertTrue(waited.done)
        self.assertTrue(deleted_document)
        self.assertTrue(deleted_store)

    async def test_gemini_files_client_crud(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str,
            *,
            headers: dict[str, str],
            json_body: dict[str, Any] | None = None,
            timeout_ms: int | None,
            stream: bool = False,
            method: str = "POST",
            body: Any = None,
        ):
            requests.append({"url": url, "method": method, "headers": headers, "json": json_body, "body": body})
            if "upload/v1beta/files" in url:
                return FakeResponse(status_code=200, payload={}, headers={"x-goog-upload-url": "https://upload.example.com/resumable"})
            if url == "https://upload.example.com/resumable":
                return FakeResponse(status_code=200, payload={"file": {"name": "files/123", "displayName": "stub.pdf", "mimeType": "application/pdf", "sizeBytes": 12, "state": "ACTIVE", "uri": "gs://files/123"}})
            if method == "GET" and "/files?" in url:
                return FakeResponse(status_code=200, payload={"files": [{"name": "files/123", "displayName": "stub.pdf", "mimeType": "application/pdf", "sizeBytes": 12, "state": "ACTIVE", "uri": "gs://files/123"}]})
            if method == "GET":
                return FakeResponse(status_code=200, payload={"name": "files/123", "displayName": "stub.pdf", "mimeType": "application/pdf", "sizeBytes": 12, "state": "ACTIVE", "uri": "gs://files/123"})
            if method == "DELETE":
                return FakeResponse(status_code=200, payload={})
            raise AssertionError(f"Unexpected request: {method} {url}")

        provider = create_gemini(api_key="test", fetch=fetch)
        files = provider.files()
        created = await files.upload(data=b"%PDF-1.4", filename="stub.pdf")
        listed = await files.list()
        fetched = await files.get("files/123")
        deleted = await files.delete("files/123")

        self.assertEqual(created.id, "files/123")
        self.assertEqual(created.file_uri, "gs://files/123")
        self.assertEqual(listed[0].status, "ACTIVE")
        self.assertEqual(fetched.media_type, "application/pdf")
        self.assertTrue(deleted)
        self.assertEqual(requests[0]["headers"]["x-goog-upload-header-content-type"], "application/pdf")

    async def test_gemini_files_client_supports_audio_uploads(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str,
            *,
            headers: dict[str, str],
            json_body: dict[str, Any] | None = None,
            timeout_ms: int | None,
            stream: bool = False,
            method: str = "POST",
            body: Any = None,
        ):
            requests.append({"url": url, "method": method, "headers": headers, "json": json_body, "body": body})
            if "upload/v1beta/files" in url:
                return FakeResponse(status_code=200, payload={}, headers={"x-goog-upload-url": "https://upload.example.com/audio"})
            if url == "https://upload.example.com/audio":
                return FakeResponse(
                    status_code=200,
                    payload={"file": {"name": "files/audio-1", "displayName": "sample.mp3", "mimeType": "audio/mpeg", "uri": "files/audio-1"}},
                )
            raise AssertionError(f"Unexpected request: {method} {url}")

        provider = create_gemini(api_key="test", fetch=fetch)
        created = await provider.files().upload(data=b"mp3-bytes", filename="sample.mp3", media_type="audio/mpeg")

        self.assertEqual(created.media_type, "audio/mpeg")
        self.assertEqual(requests[0]["headers"]["x-goog-upload-header-content-type"], "audio/mpeg")

    async def test_gemini_reports_builtin_search_tool_without_opt_in(self) -> None:
        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            return FakeResponse(
                status_code=200,
                payload={
                    "candidates": [
                        {
                            "content": {"parts": [{"functionCall": {"name": "search", "args": {"query": "Apollo"}}}]},
                            "finishReason": "STOP",
                        }
                    ],
                },
            )

        provider = create_gemini(api_key="test", fetch=fetch)
        with self.assertRaises(UnsupportedFeatureError) as context:
            await generate_text(model=provider("gemini-3.1-flash-preview"), prompt="Research Apollo.")

        self.assertIn("google_search", str(context.exception))

    async def test_gemini_normalizes_mcp_tool_schema(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            return FakeResponse(
                status_code=200,
                payload={
                    "candidates": [{"content": {"parts": [{"text": "done"}]}, "finishReason": "STOP"}],
                },
            )

        provider = create_gemini(api_key="test", fetch=fetch)
        await generate_text(
            model=provider("gemini-2.5-flash"),
            prompt="hello",
            tools={
                "fs_read_file": tool(
                    name="fs_read_file",
                    schema={
                        "type": "object",
                        "title": "Read File Input",
                        "properties": {
                            "path": {"type": "string", "title": "Path"},
                            "head": {"type": "integer", "default": 20},
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                    source="mcp",
                    mcp_config=MCPToolConfig(
                        server=MCPServerConfig(transport="stdio", name="fs", command="npx"),
                        tool_name="read_file",
                    ),
                )
            },
        )

        parameters = requests[0]["tools"][0]["functionDeclarations"][0]["parameters"]
        self.assertEqual(parameters["type"], "object")
        self.assertEqual(parameters["required"], ["path"])
        self.assertNotIn("title", parameters)
        self.assertNotIn("additionalProperties", parameters)
        self.assertNotIn("default", parameters["properties"]["head"])
        self.assertEqual(parameters["properties"]["head"]["type"], "integer")

    async def test_gemini_http_error_includes_response_body(self) -> None:
        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            return FakeResponse(status_code=400, body_text='{"error":{"message":"Bad schema: additionalProperties"}}')

        provider = create_gemini(api_key="test", fetch=fetch)
        with self.assertRaises(ProviderHTTPError) as context:
            await generate_text(model=provider("gemini-2.5-flash"), prompt="hello")

        self.assertIn("Response body:", str(context.exception))
        self.assertIn("Bad schema", str(context.exception))

    async def test_gemini_preserves_thought_signature_across_tool_loop(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            if len(requests) == 1:
                return FakeResponse(
                    status_code=200,
                    payload={
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "functionCall": {"name": "weather", "args": {"city": "Madrid"}},
                                            "thoughtSignature": "sig-123",
                                        }
                                    ]
                                },
                                "finishReason": "STOP",
                            }
                        ]
                    },
                )
            return FakeResponse(
                status_code=200,
                payload={
                    "candidates": [{"content": {"parts": [{"text": "sunny"}]}, "finishReason": "STOP"}],
                },
            )

        provider = create_gemini(api_key="test", fetch=fetch)
        result = await generate_text(
            model=provider("gemini-2.5-flash"),
            prompt="weather",
            max_steps=2,
            tools={
                "weather": tool(
                    name="weather",
                    schema={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
                    execute=lambda input: {"forecast": "sunny"},
                )
            },
        )

        self.assertEqual(result.text, "sunny")
        second_request_parts = requests[1]["contents"][1]["parts"]
        function_call_part = next(part for part in second_request_parts if "functionCall" in part)
        self.assertEqual(function_call_part["thoughtSignature"], "sig-123")

    async def test_gemini_stream_preserves_snake_case_thought_signature(self) -> None:
        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            return FakeResponse(
                status_code=200,
                body_text=(
                    'data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"weather","args":{"city":"Madrid"}},"thought_signature":"sig-456"}]},"finishReason":"STOP"}]}\n\n'
                ),
            )

        provider = create_gemini(api_key="test", fetch=fetch)
        stream = stream_text(model=provider("gemini-3-flash-preview"), prompt="weather")
        events = [event async for event in stream.event_stream()]

        tool_call = next(event.tool_call for event in events if event.type == "tool-call")
        self.assertEqual(tool_call.provider_metadata["thought_signature"], "sig-456")
