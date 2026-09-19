from __future__ import annotations

import asyncio
import os
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from zhivex_ai import create_vertex, generate_text
from zhivex_ai.errors import ConfigurationError, ValidationError
from tests.test_gemini_provider import FakeResponse


class VertexProviderTests(IsolatedAsyncioTestCase):
    async def test_mistral_native_predict_uses_publisher_and_stream_action(self):
        seen = []
        payload = {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}
        response = FakeResponse(200, payload)

        async def fetch(url, **kwargs):
            seen.append((url, kwargs))
            return response

        garden = create_vertex(access_token="synthetic", project_id="p", location="us-central1", fetch=fetch).native.model_garden()
        for stream in (False, True):
            body = {"model": "mistral-medium-3", "stream": stream,
                    "messages": [{"role": "user", "content": "Synthetic"}], "max_tokens": 128}
            result = await garden.raw_predict(publisher="mistralai", model="mistral-medium-3", body=body, stream=stream)
            action = "streamRawPredict" if stream else "rawPredict"
            self.assertEqual(seen[-1][0], f"https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/publishers/mistralai/models/mistral-medium-3:{action}")
            self.assertEqual(seen[-1][1]["json_body"], body)
            self.assertEqual(seen[-1][1].get("stream", False), stream)
            if stream:
                self.assertIs(result, response)
            else:
                self.assertEqual(result, payload)

    async def test_partner_grounding_preserves_config_without_google_search(self):
        from copy import deepcopy
        from zhivex_ai import generate_grounded_text

        configs = [
            {"exaAiSearch": {"api_key": "synthetic-key", "customConfigs": {
                "includeDomains": ["example.com"], "numResults": 2,
            }}},
            {"parallelAiSearch": {"api_key": "synthetic-key", "customConfigs": {
                "source_policy": {"include_domains": ["example.com"]}, "max_results": 2,
            }}},
            {"parallelAiSearch": {"enable_zero_data_retention": True}},
        ]
        for tool in configs:
            with self.subTest(tool=next(iter(tool)), marketplace="api_key" not in next(iter(tool.values()))):
                options = {"tools": [deepcopy(tool)]}
                original = deepcopy(options)

                async def fetch(url, **kwargs):
                    body = kwargs["json_body"]
                    self.assertEqual(body["tools"], [tool])
                    body["tools"][0][next(iter(tool))]["mutation"] = True
                    return FakeResponse(200, {"candidates": [{
                        "content": {"parts": [{"text": "Grounded answer"}]},
                        "finishReason": "STOP",
                        "groundingMetadata": {
                            "groundingChunks": [{"web": {"uri": "https://example.com/source", "title": "Source"}}],
                            "groundingSupports": [{"segment": {"startIndex": 0, "endIndex": 15, "text": "Grounded answer"}, "groundingChunkIndices": [0]}],
                        },
                    }]})

                provider = create_vertex(access_token="synthetic", project_id="p", location="global", fetch=fetch)
                result = await generate_grounded_text(
                    model=provider.native.grounded_language_model("gemini-3.8-flash"),
                    prompt="Synthetic grounding check", provider_options=options,
                )
                self.assertEqual(options, original)
                self.assertEqual(result.sources[0].url, "https://example.com/source")
                self.assertEqual(result.supports[0].source_indices, [0])

    async def test_responses_stream_failures_and_early_close(self):
        import json
        from zhivex_ai.errors import ZhivexAIError
        from zhivex_ai.types import ModelGenerateInput

        for events in (
            [],
            ["[DONE]"],
            [{"type": "error", "message": "upstream failure"}],
            [{"type": "response.output_text.delta", "delta": "partial"}],
            [{"type": "response.completed", "response": {"status": "in_progress"}}],
        ):
            closed = []

            class Response(FakeResponse):
                async def iter_lines(self):
                    try:
                        for event in events:
                            yield "data: " + (event if isinstance(event, str) else json.dumps(event))
                            yield ""
                    finally:
                        closed.append(True)

            async def fetch(url, **kwargs):
                return Response(200)

            model = create_vertex(access_token="t", project_id="p", fetch=fetch).native.model_garden().responses_model("xai/grok-4.6")
            stream = await model.stream(ModelGenerateInput(messages=[]))
            with self.subTest(events=events), self.assertRaises(ZhivexAIError):
                async for _ in stream:
                    pass
            self.assertEqual(closed, [True])

        for status, reason, expected in (("incomplete", "max_output_tokens", "length"), ("failed", None, "error")):
            events = [{"type": "response." + status, "response": {"status": status, "incomplete_details": {"reason": reason}}}]
            closed = []
            stream = await model.stream(ModelGenerateInput(messages=[]))
            finishes = [event async for event in stream if event.type == "finish"]
            self.assertEqual(finishes[0].finish_reason, expected)
            self.assertEqual(closed, [True])

        events = [{"type": "response.output_text.delta", "delta": "partial"}]
        closed = []
        stream = await model.stream(ModelGenerateInput(messages=[]))
        await anext(stream)
        await stream.aclose()
        self.assertEqual(closed, [True])

    async def test_normalized_responses_uses_vertex_auth_and_text_stream_contract(self):
        import json
        from zhivex_ai import stream_text

        seen = []
        payload = {"id": "r", "status": "completed", "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hello"}]}], "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}}

        async def fetch(url, **kwargs):
            seen.append((url, kwargs))
            if kwargs.get("stream"):
                events = [{"type": "response.output_text.delta", "delta": "hello"}, {"type": "response.completed", "response": payload}]
                return FakeResponse(200, body_text="".join("data: " + json.dumps(event) + "\n\n" for event in events))
            return FakeResponse(200, payload)

        garden = create_vertex(access_token="vertex-token", project_id="p", location="global", fetch=fetch).native.model_garden()
        model = garden.responses_model("xai/grok-4.6")
        self.assertEqual(model.provider, "vertex")
        self.assertFalse(model.capabilities.tools)
        self.assertFalse(model.capabilities.structured_output)
        generated = await generate_text(model=model, prompt="hi", provider_options={"store": False})
        streamed = await stream_text(model=model, prompt="hi", provider_options={"store": False}).collect()
        self.assertEqual(generated.text, "hello")
        self.assertEqual(streamed.text, "hello")
        self.assertEqual(generated.usage.total_tokens, 3)
        self.assertEqual(streamed.usage.total_tokens, 3)
        for url, request in seen:
            self.assertEqual(url, "https://aiplatform.googleapis.com/v1/projects/p/locations/global/endpoints/openapi/responses")
            self.assertEqual(request["headers"]["authorization"], "Bearer vertex-token")
            self.assertEqual(request["json_body"]["model"], "xai/grok-4.6")
            self.assertFalse(request["json_body"]["store"])
        with self.assertRaises(ValidationError):
            garden.responses_model("xai/grok-4.6", endpoint="https://outside.example")

    async def test_model_garden_responses_routes_and_preserves_native_payload(self):
        from zhivex_ai.types import RetryOptions
        from zhivex_ai.errors import ProviderHTTPError

        seen = []
        response = FakeResponse(200, {"id": "resp_1", "output": []})

        async def fetch(url, **kwargs):
            seen.append((url, kwargs))
            return response

        garden = create_vertex(access_token="t", project_id="p", location="us", fetch=fetch).native.model_garden()
        body = {"model": "xai/grok-4.6", "input": "hello", "store": False, "max_output_tokens": 64}
        self.assertEqual((await garden.responses(body))["id"], "resp_1")
        self.assertEqual(seen[-1][0], "https://aiplatform.us.rep.googleapis.com/v1/projects/p/locations/us/endpoints/openapi/responses")
        self.assertEqual(seen[-1][1]["json_body"], body)
        streaming_body = {**body, "stream": True}
        stream = await garden.responses(streaming_body, options=RetryOptions(timeout_ms=1234))
        self.assertIs(stream, response)
        self.assertTrue(seen[-1][1]["stream"])
        self.assertEqual(seen[-1][1]["timeout_ms"], 1234)
        self.assertEqual(seen[-1][1]["json_body"], streaming_body)
        for endpoint in ("../other", "x?route=other", "https://outside.example"):
            with self.assertRaises(ValidationError):
                await garden.responses(body, endpoint=endpoint)
        self.assertEqual(len(seen), 2)
        response = FakeResponse(404, {"error": {"message": "unavailable"}})
        for payload in (body, streaming_body):
            with self.assertRaises(ProviderHTTPError) as error:
                await garden.responses(payload)
            self.assertEqual(error.exception.status, 404)

    async def test_cyber_rejects_grounding_and_batch_without_affecting_flash(self):
        from zhivex_ai.errors import UnsupportedFeatureError

        seen = []

        async def fetch(url, **kwargs):
            seen.append(kwargs["json_body"])
            return FakeResponse(200, {"name": "batchPredictionJobs/b"})

        provider = create_vertex(access_token="t", project_id="p", fetch=fetch)
        for factory in (provider.grounded_language_model, provider.native.grounded_language_model):
            with self.assertRaises(UnsupportedFeatureError):
                factory("gemini-3.8-flash-cyber")
            self.assertIsNotNone(factory("gemini-3.8-flash"))
        for model in ("gemini-3.8-flash-cyber", "publishers/google/models/gemini-3.8-flash-cyber", "projects/p/locations/global/publishers/google/models/gemini-3.8-flash-cyber"):
            with self.assertRaises(UnsupportedFeatureError):
                await provider.batches().create({"model": model})
        self.assertEqual(seen, [])
        body = {"model": "publishers/google/models/gemini-3.8-flash", "inputConfig": {"instancesFormat": "jsonl"}}
        await provider.batches().create(body)
        self.assertEqual(seen, [body])

    async def test_cyber_has_model_specific_capabilities_and_rejects_native_tools(self):
        from zhivex_ai import default_model_catalog, vertex_google_search_tool
        from zhivex_ai.errors import UnsupportedFeatureError
        from zhivex_ai.types import ModelGenerateInput

        async def fetch(url, **kwargs):
            self.fail("Unsupported Cyber tools must fail before HTTP")

        provider = create_vertex(access_token="t", project_id="p", location="global", fetch=fetch)
        model = provider.native.language_model("gemini-3.8-flash-cyber")
        self.assertFalse(model.capabilities.tools)
        self.assertFalse(provider("gemini-3.8-flash-cyber").capabilities.tool_choice)
        self.assertFalse(model.capabilities.agent_capabilities.hosted_web_search)
        self.assertFalse(model.capabilities.agent_capabilities.code_execution)
        self.assertTrue(model.capabilities.structured_output)
        self.assertTrue(provider("gemini-3.8-flash").capabilities.tools)
        entry = default_model_catalog.find("vertex", model.model_id)
        self.assertEqual(entry.availability, "limited")
        self.assertFalse(entry.capabilities.tools)
        self.assertNotEqual(entry.support_evidence, "live-smoke")
        for input in (
            ModelGenerateInput(messages=[], tools={"search": vertex_google_search_tool()}),
            ModelGenerateInput(messages=[], provider_options={"tools": [{"codeExecution": {}}]}),
            ModelGenerateInput(messages=[], provider_options={"tools": [{"functionDeclarations": [{"name": "lookup"}]}]}),
        ):
            for method in (model.generate, model.stream):
                with self.assertRaises(UnsupportedFeatureError):
                    await method(input)

    async def test_live_translate_uses_vertex_setup_and_has_no_tools(self):
        from tests.test_realtime import FakeRealtimeConnection
        from zhivex_ai import RealtimeSessionConfig
        from zhivex_ai.errors import UnsupportedFeatureError

        connection = FakeRealtimeConnection([])

        async def factory(url, headers, options):
            self.assertIn("aiplatform.googleapis.com", url)
            return connection

        model = create_vertex(access_token="t", project_id="p", location="global", realtime_connection_factory=factory).realtime_model("gemini-3.5-live-translate-preview")
        self.assertFalse(model.capabilities.realtime_tools)
        session = await model.connect(RealtimeSessionConfig(translation_target_language_code="es", translation_echo_target_language=False))
        setup = connection.sent[0]["setup"]
        self.assertIn("projects/p/locations/global/publishers/google/models/", setup["model"])
        self.assertEqual(setup["generationConfig"]["responseModalities"], ["AUDIO"])
        self.assertEqual(setup["generationConfig"]["translationConfig"], {"targetLanguageCode": "es", "echoTargetLanguage": False})
        self.assertEqual(setup["inputAudioTranscription"], {})
        self.assertEqual(setup["outputAudioTranscription"], {})
        with self.assertRaises(UnsupportedFeatureError):
            await session.send_text("hello")
        await session.aclose()

    async def test_live_transcribe_interim_and_final_utterances(self):
        from tests.test_realtime import FakeRealtimeConnection
        from zhivex_ai import RealtimeSessionConfig, RealtimeTranscriptEvent, AudioFrame

        connection = FakeRealtimeConnection([
            {"serverContent": {"interimInputTranscription": {"text": "hel"}}},
            {"serverContent": {"inputTranscription": {"text": "hello"}}},
        ])

        async def factory(url, headers, options):
            return connection

        model = create_vertex(access_token="t", project_id="p", location="global", realtime_connection_factory=factory).realtime_model("gemini-3.5-transcribe-live-preview")
        session = await model.connect(RealtimeSessionConfig())
        self.assertEqual(connection.sent[0]["setup"]["inputAudioTranscription"], {})
        self.assertEqual(connection.sent[0]["setup"]["generationConfig"]["responseModalities"], ["TEXT"])
        events = [e async for e in session.event_stream() if isinstance(e, RealtimeTranscriptEvent)]
        self.assertEqual([(e.text, e.is_final) for e in events], [("hel", False), ("hello", True)])
        self.assertFalse(model.capabilities.realtime_audio_output)
        with self.assertRaises(ValidationError):
            await model.connect(RealtimeSessionConfig(instructions="hello"))
        with self.assertRaises(ValidationError):
            await model.connect(RealtimeSessionConfig(output_audio_media_type="audio/pcm"))
        # Use a fresh connection after the synthetic receive stream has ended.
        connection = FakeRealtimeConnection([])
        session = await model.connect(RealtimeSessionConfig())
        with self.assertRaises(ValidationError):
            await session.send_text("hello")
        await session.send_audio(AudioFrame(data=b"12", media_type="audio/pcm;rate=16000", is_final=True))
        self.assertEqual(connection.sent[-1], {"realtimeInput": {"audioStreamEnd": True}})
        await session.aclose()

    async def test_transcribe_audio_only_config_and_metadata(self):
        from zhivex_ai import AudioInput

        seen = []
        metadata = {"text": "hello", "speakerLabel": "1", "words": [{"word": "hello", "startOffset": "0s"}]}
        payload = {"candidates": [{"content": {"parts": [{"audioTranscription": metadata}]}}]}

        async def fetch(url, **kwargs):
            seen.append((url, kwargs["json_body"]))
            return FakeResponse(200, payload)

        provider = create_vertex(access_token="t", project_id="p", location="global", fetch=fetch)
        model = provider.native.transcription_model("gemini-3.5-transcribe-preview")
        options = {"generationConfig": {"audioTranscriptionConfig": {"wordTimestamp": True, "customVocabulary": ["Zhivex"]}}}
        result = await model.transcribe(audio=AudioInput(data=b"abc", media_type="audio/wav"), language="en-US", provider_options=options)
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.raw_response, payload)
        self.assertNotIn("languageCodes", options["generationConfig"]["audioTranscriptionConfig"])
        self.assertTrue(seen[0][0].endswith("/gemini-3.5-transcribe-preview:generateContent"))
        self.assertEqual(seen[0][1]["contents"], [{"role": "user", "parts": [{"inlineData": {"mimeType": "audio/wav", "data": "YWJj"}}]}])
        self.assertEqual(seen[0][1]["generationConfig"]["audioTranscriptionConfig"]["languageCodes"], ["en-US"])
        for kwargs in ({"prompt": "hello"}, {"provider_options": {"contents": []}}, {"provider_options": {"generationConfig": []}}):
            with self.assertRaises(ValidationError):
                await model.transcribe(audio=AudioInput(data=b"abc", media_type="audio/wav"), **kwargs)
        with self.assertRaises(ValidationError):
            await provider.transcription_model("gemini-3.5-transcribe-live-preview").transcribe(audio=AudioInput(data=b"abc", media_type="audio/wav"))
        self.assertEqual(len(seen), 1)

    async def test_transcribe_does_not_report_empty_success(self):
        from zhivex_ai import AudioInput

        async def fetch(url, **kwargs):
            return FakeResponse(200, {"candidates": []})

        model = create_vertex(access_token="t", project_id="p", fetch=fetch).transcription_model("gemini-3.5-transcribe-preview")
        with self.assertRaises(ValidationError):
            await model.transcribe(audio=AudioInput(data=b"abc", media_type="audio/wav"))

    async def test_vertex_hosted_tools_pass_native_provider_validation(self):
        from zhivex_ai import (
            vertex_google_search_tool, vertex_google_maps_tool, vertex_url_context_tool,
            vertex_code_execution_tool, vertex_computer_use_tool,
        )

        seen = []

        async def fetch(url, **kwargs):
            seen.append(kwargs["json_body"])
            return FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": "OK"}]}, "finishReason": "STOP"}]})

        p = create_vertex(access_token="t", project_id="p", fetch=fetch)
        for factory, wire_key in (
            (vertex_google_search_tool, "googleSearch"),
            (vertex_google_maps_tool, "googleMaps"),
            (vertex_url_context_tool, "urlContext"),
            (vertex_code_execution_tool, "codeExecution"),
            (vertex_computer_use_tool, "computerUse"),
        ):
            hosted = factory()
            self.assertEqual(hosted.provider, "vertex")
            await generate_text(model=p.native.language_model("gemini-3.8-flash"), prompt="synthetic", tools={"hosted": hosted})
            self.assertIn(wire_key, seen[-1]["tools"][0])

    async def test_multimodal_predict_preserves_modality_and_input_order(self):
        from zhivex_ai import ImagePart, TextPart

        seen = []

        async def fetch(url, **kwargs):
            seen.append(kwargs["json_body"])
            instance = seen[-1]["instances"][0]
            field = "textEmbedding" if "text" in instance else "imageEmbedding"
            return FakeResponse(200, {"predictions": [{field: [len(seen), 0.5]}]})

        model = create_vertex(access_token="t", project_id="p", fetch=fetch).embedding_model("multimodalembedding@001")
        result = await model.embed(["red", ImagePart(image="data:image/png;base64,AQID")], {"output_dimensionality": 128})
        self.assertEqual(result.embeddings, [[1, 0.5], [2, 0.5]])
        self.assertEqual(seen, [
            {"instances": [{"text": "red"}], "parameters": {"dimension": 128}},
            {"instances": [{"image": {"bytesBase64Encoded": "AQID"}}], "parameters": {"dimension": 128}},
        ])
        with self.assertRaises(ValidationError):
            await model.embed([[TextPart(text="x"), ImagePart(image="data:image/png;base64,AQID")]])
        with self.assertRaises(ValidationError):
            await model.embed(["x"], {"task_type": "RETRIEVAL_QUERY"})
        self.assertEqual(len(seen), 2)

    async def test_multimodal_predict_rejects_missing_requested_modality(self):
        async def fetch(url, **kwargs):
            return FakeResponse(200, {"predictions": [{"imageEmbedding": [1, 2]}]})

        model = create_vertex(access_token="t", project_id="p", fetch=fetch).embedding_model("multimodalembedding@001")
        with self.assertRaises(ValidationError):
            await model.embed(["text requires textEmbedding"])

    async def test_jurisdictional_endpoints_and_custom_origin(self):
        seen = []

        async def fetch(url, **kwargs):
            seen.append(url)
            return FakeResponse(200, {"candidates": []})

        for location in ("us", "eu"):
            p = create_vertex(access_token="t", project_id="p", location=location, fetch=fetch)
            await generate_text(model=p("gemini-3.8-flash"), prompt="synthetic")
            self.assertTrue(seen[-1].startswith(f"https://aiplatform.{location}.rep.googleapis.com/"))
            self.assertIn(f"/locations/{location}/", seen[-1])
        custom = create_vertex(access_token="t", project_id="p", location="us", base_url="https://custom.test/v1/projects/p/locations/us", fetch=fetch)
        await generate_text(model=custom("gemini-3.8-flash"), prompt="synthetic")
        self.assertTrue(seen[-1].startswith("https://custom.test/"))

    async def test_embedding_two_uses_content_and_preserves_input_cardinality(self):
        from zhivex_ai import TextPart

        seen = []

        async def fetch(url, **kwargs):
            seen.append((url, kwargs))
            return FakeResponse(200, {"embedding": {"values": [len(seen), 0.5]}})

        p = create_vertex(access_token="t", project_id="p", location="us", fetch=fetch)
        result = await p.embedding_model("gemini-embedding-2").embed(
            ["first", [TextPart(text="second"), TextPart(text="third")]],
            {"output_dimensionality": 2, "task_types": ["RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT"]},
        )
        self.assertEqual(result.embeddings, [[1, 0.5], [2, 0.5]])
        self.assertEqual(len(seen), 2)
        self.assertTrue(seen[0][0].endswith("/gemini-embedding-2:embedContent"))
        self.assertEqual(seen[0][1]["json_body"]["taskType"], "RETRIEVAL_QUERY")
        self.assertEqual(seen[1][1]["json_body"]["content"]["parts"], [{"text": "second"}, {"text": "third"}])
        self.assertEqual(seen[1][1]["json_body"]["outputDimensionality"], 2)
        self.assertNotIn("instances", seen[0][1]["json_body"])
        with self.assertRaises(ValidationError):
            await p.embedding_model("gemini-embedding-2").embed(["x"], {"titles": []})
        self.assertEqual(len(seen), 2)

    async def test_native_embedding_config_preserves_response_and_auth(self):
        from copy import deepcopy
        from zhivex_ai.types import RetryOptions

        seen = []
        response = {"embedding": {"values": [0.1, 0.2]}, "truncated": False,
                    "usageMetadata": {"totalTokenCount": 3}}

        async def fetch(url, **kwargs):
            seen.append((url, kwargs))
            return FakeResponse(200, response)

        client = create_vertex(access_token="vertex-token", project_id="p", location="us", fetch=fetch).native.model_garden()
        body = {"content": {"parts": [{"fileData": {"fileUri": "gs://bucket/doc.pdf", "mimeType": "application/pdf"}}]},
                "embedContentConfig": {"outputDimensionality": 2, "documentOcr": True, "audioTrackExtraction": False}}
        before = deepcopy(body)
        result = await client.embed_content(model="gemini-embedding-2", body=body, options=RetryOptions(timeout_ms=1234))
        self.assertEqual(result, response)
        self.assertEqual(body, before)
        self.assertEqual(seen[0][1]["json_body"], before)
        self.assertEqual(seen[0][1]["timeout_ms"], 1234)
        self.assertEqual(seen[0][1]["headers"]["authorization"], "Bearer vertex-token")
        self.assertEqual(seen[0][0], "https://aiplatform.us.rep.googleapis.com/v1/projects/p/locations/us/publishers/google/models/gemini-embedding-2:embedContent")
        for model in ("", "../model", "model?query=1", "model#fragment"):
            with self.subTest(model=model), self.assertRaises(ValidationError):
                await client.embed_content(model=model, body=body)
        self.assertEqual(len(seen), 1)

    async def test_embedding_responses_fail_closed_for_invalid_vectors(self):
        payload = {}

        async def fetch(url, **kwargs):
            return FakeResponse(200, payload)

        p = create_vertex(access_token="t", project_id="p", fetch=fetch)
        for vector in ([], [float("nan")], [True], ["0.1"]):
            payload = {"embedding": {"values": vector}}
            with self.assertRaises(ValidationError):
                await p.embedding_model("gemini-embedding-2").embed(["x"])
        payload = {"predictions": []}
        with self.assertRaises(ValidationError):
            await p.embedding_model("text-embedding-005").embed(["x"])

    async def test_global_and_express_routes_and_auth(self):
        requests = []

        async def fetch(url, **kwargs):
            requests.append((url, kwargs))
            return FakeResponse(
                200, {"candidates": [{"content": {"parts": [{"text": "OK"}]}}]}
            )

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "ambient"}, clear=True):
            standard = create_vertex(
                access_token="explicit", project_id="p", location="global", fetch=fetch
            )
            await generate_text(model=standard("gemini-2.5-flash"), prompt="hello")
            express = create_vertex(api_key="express", fetch=fetch)
            await generate_text(model=express("gemini-2.5-flash"), prompt="hello")
        self.assertEqual(
            requests[0][0],
            "https://aiplatform.googleapis.com/v1/projects/p/locations/global/publishers/google/models/gemini-2.5-flash:generateContent",
        )
        self.assertEqual(requests[0][1]["headers"]["authorization"], "Bearer explicit")
        self.assertEqual(
            requests[1][0],
            "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-flash:generateContent",
        )
        self.assertNotIn("authorization", requests[1][1]["headers"])
        self.assertEqual(requests[1][1]["headers"]["x-goog-api-key"], "express")
        self.assertNotIn("express", requests[1][0])
        with self.assertRaises(AttributeError):
            express.batches()

    async def test_credentials_refresh_is_offloaded_and_used_on_every_request(self):
        import threading

        main_thread = threading.get_ident()

        class Credentials:
            calls = 0

            def before_request(self, request, method, url, headers):
                assert threading.get_ident() != main_thread
                self.calls += 1
                headers["authorization"] = f"Bearer refreshed-{self.calls}"
                headers["x-goog-user-project"] = "billing"

        credentials = Credentials()
        seen = []

        async def fetch(url, **kwargs):
            seen.append(kwargs["headers"])
            return FakeResponse(200, {"totalTokens": 1})

        provider = create_vertex(credentials=credentials, project_id="p", fetch=fetch)
        await asyncio.gather(
            *(
                provider.tokens().count(model_id="gemini-2.5-flash", prompt="hello")
                for _ in range(3)
            )
        )
        self.assertEqual(
            {h["authorization"] for h in seen},
            {"Bearer refreshed-1", "Bearer refreshed-2", "Bearer refreshed-3"},
        )
        self.assertTrue(all(h["x-goog-user-project"] == "billing" for h in seen))

    def test_ambiguous_auth_rejected(self):
        with self.assertRaises(ConfigurationError):
            create_vertex(api_key="key", access_token="token")
        with self.assertRaises(ConfigurationError):
            create_vertex(access_token="token", express_mode=True)

    async def test_native_resource_paths_and_payloads(self):
        seen = []

        async def fetch(url, **kwargs):
            seen.append((url, kwargs))
            return FakeResponse(
                200, {"name": "projects/p/locations/us-central1/cachedContents/c"}
            )

        provider = create_vertex(access_token="token", project_id="p", fetch=fetch)
        cache = await provider.caches().create(
            {"model": "gemini-2.5-flash", "ttl": "60s"}
        )
        await provider.caches().get(cache.name)
        await provider.batches().create(
            {
                "model": "publishers/google/models/gemini-2.5-flash",
                "inputConfig": {"instancesFormat": "jsonl"},
            }
        )
        await provider.batches().cancel(
            "projects/p/locations/us-central1/batchPredictionJobs/b"
        )
        await provider.interactions().create(
            {"model": "gemini-2.5-flash", "input": "hello"}
        )
        self.assertEqual(
            seen[0][1]["json_body"]["model"],
            "projects/p/locations/us-central1/publishers/google/models/gemini-2.5-flash",
        )
        self.assertTrue(
            seen[1][0].endswith("/v1/projects/p/locations/us-central1/cachedContents/c")
        )
        self.assertTrue(seen[2][0].endswith("/batchPredictionJobs"))
        self.assertIn("inputConfig", seen[2][1]["json_body"])
        self.assertTrue(seen[3][0].endswith("/batchPredictionJobs/b:cancel"))
        self.assertIn(
            "/v1beta1/projects/p/locations/us-central1/interactions", seen[4][0]
        )
        self.assertTrue(all("key=" not in url for url, _ in seen))
        for name in (
            "https://evil.invalid/x",
            "../x",
            "cachedContents/%2e%2e/x",
            "cachedContents/x?key=oops",
        ):
            with self.assertRaises(ValidationError):
                await provider.caches().get(name)
        self.assertEqual(len(seen), 5)

    async def test_batch_wait_has_wall_clock_deadline(self):
        async def fetch(url, **kwargs):
            await asyncio.sleep(10)
            return FakeResponse(200, {})

        provider = create_vertex(access_token="token", project_id="p", fetch=fetch)
        with self.assertRaises(TimeoutError):
            await provider.batches().wait("b", timeout_ms=5)

    async def test_live_setup_uses_vertex_resource_and_waits_for_ack(self):
        from tests.test_realtime import FakeRealtimeConnection
        from zhivex_ai import RealtimeSessionConfig

        connection = FakeRealtimeConnection([])
        seen = []

        async def factory(url, headers, options):
            seen.append((url, headers))
            return connection

        provider = create_vertex(
            access_token="token", project_id="p", realtime_connection_factory=factory
        )
        session = await provider.realtime_model(
            "gemini-live-2.5-flash-native-audio"
        ).connect(RealtimeSessionConfig())
        self.assertEqual(
            connection.sent[0]["setup"]["model"],
            "projects/p/locations/us-central1/publishers/google/models/gemini-live-2.5-flash-native-audio",
        )
        self.assertEqual(seen[0][1]["authorization"], "Bearer token")
        self.assertEqual(
            seen[0][0],
            "wss://us-central1-aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent",
        )
        await session.aclose()

    async def test_platform_sessions_memory_and_operation_routes(self):
        seen = []

        async def fetch(url, **kwargs):
            seen.append((url, kwargs))
            return FakeResponse(200, {"done": True})

        p = create_vertex(access_token="t", project_id="p", fetch=fetch)
        platform = p.native.agent_platform()
        await platform.agents().query("a", {"input": {"text": "hi"}})
        await platform.sessions("a").create({"userId": "u"})
        await platform.sessions("a").append_event("s", {"event": {"author": "user"}})
        await platform.memory_bank("a").generate(
            {"directContentsSource": {"events": []}}
        )
        await platform.wait_operation("projects/p/locations/us-central1/operations/op")
        self.assertTrue(seen[0][0].endswith("/reasoningEngines/a:query"))
        self.assertTrue(seen[1][0].endswith("/reasoningEngines/a/sessions"))
        self.assertTrue(
            seen[2][0].endswith("/reasoningEngines/a/sessions/s:appendEvent")
        )
        self.assertIn(
            "/v1beta1/projects/p/locations/us-central1/reasoningEngines/a/memories:generate",
            seen[3][0],
        )
        self.assertTrue(
            seen[4][0].endswith("/v1/projects/p/locations/us-central1/operations/op")
        )

    async def test_veo_polling_and_download_does_not_leak_auth(self):
        seen = []

        async def fetch(url, **kwargs):
            seen.append((url, kwargs))
            return FakeResponse(200, {"done": True})

        p = create_vertex(access_token="secret", project_id="p", fetch=fetch)
        name = "projects/p/locations/us-central1/publishers/google/models/veo-3.1-generate-001/operations/op"
        await p.videos().get_operation(name)
        await p.videos().download("https://storage.googleapis.com/bucket/signed-video")
        self.assertTrue(
            seen[0][0].endswith("/models/veo-3.1-generate-001:fetchPredictOperation")
        )
        self.assertEqual(seen[0][1]["json_body"], {"operationName": name})
        self.assertNotIn("authorization", seen[1][1]["headers"])

    async def test_deployed_endpoint_prediction_json_binary_stream_and_errors(self):
        from zhivex_ai._http import BufferedResponse
        from zhivex_ai.errors import ProviderHTTPError
        from zhivex_ai.types import RetryOptions

        seen = []
        response = BufferedResponse(200, b'{"predictions":[{"value":7}]}')

        async def fetch(url, **kwargs):
            seen.append((url, kwargs))
            return response

        garden = create_vertex(access_token="token", project_id="p", location="us-central1", fetch=fetch).native.model_garden()
        body = {"instances": [{"input": "synthetic"}], "parameters": {"temperature": 0}}
        result = await garden.endpoint_predict(endpoint="123", body=body)
        self.assertEqual(result, {"predictions": [{"value": 7}]})
        self.assertEqual(seen[-1][0], "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/endpoints/123:predict")
        self.assertEqual(seen[-1][1]["json_body"], body)
        response = BufferedResponse(200, b"\x00\xffbinary", {"x-vertex-ai-deployed-model-id": "456"})
        for stream in (False, True):
            raw = await garden.endpoint_raw_predict(endpoint="123", body=b"\x00\xffinput", content_type="application/octet-stream", stream=stream, options=RetryOptions(timeout_ms=1234))
            self.assertIs(raw, response)
            self.assertEqual(await raw.read(), b"\x00\xffbinary")
            self.assertEqual(raw.headers["x-vertex-ai-deployed-model-id"], "456")
            self.assertEqual(seen[-1][1]["body"], b"\x00\xffinput")
            self.assertEqual(seen[-1][1]["headers"]["authorization"], "Bearer token")
            self.assertEqual(seen[-1][1]["timeout_ms"], 1234)
            self.assertEqual(seen[-1][1]["stream"], stream)
            self.assertTrue(seen[-1][0].endswith(":streamRawPredict" if stream else ":rawPredict"))
        count = len(seen)
        for endpoint in ("", "../123", "https://evil.test", "123:delete", "123?query", "projects/p/locations/us/endpoints/123"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValidationError):
                await garden.endpoint_predict(endpoint=endpoint, body=body)
        with self.assertRaises(ValidationError):
            await garden.endpoint_raw_predict(endpoint="123", body=b"x", content_type="text/plain\r\nx-injected: true")
        self.assertEqual(len(seen), count)
        response = BufferedResponse(403, b"denied")
        with self.assertRaises(ProviderHTTPError) as error:
            await garden.endpoint_raw_predict(endpoint="123", body=b"x")
        self.assertEqual(error.exception.status, 403)

    async def test_endpoint_discovery_preserves_pagination_and_encodes_query(self):
        from urllib.parse import parse_qs, urlsplit

        seen = []
        payload = {"endpoints": [{"name": "projects/p/locations/us-central1/endpoints/123", "deployedModels": [{"id": "m"}]}], "nextPageToken": "next&token"}

        async def fetch(url, **kwargs):
            seen.append((url, kwargs))
            return FakeResponse(200, payload)

        garden = create_vertex(access_token="t", project_id="p", location="us-central1", fetch=fetch).native.model_garden()
        result = await garden.list_endpoints(page_size=10, page_token="a+b&c", filter='displayName="my endpoint"', order_by="createTime desc", read_mask="name,deployedModels")
        self.assertEqual(result, payload)
        self.assertEqual(len(seen), 1)
        self.assertEqual(parse_qs(urlsplit(seen[0][0]).query), {"pageSize": ["10"], "pageToken": ["a+b&c"], "filter": ['displayName="my endpoint"'], "orderBy": ["createTime desc"], "readMask": ["name,deployedModels"]})
        self.assertEqual(seen[0][1]["method"], "GET")
        self.assertIsNone(seen[0][1]["json_body"])
        await garden.get_endpoint("123")
        self.assertTrue(seen[-1][0].endswith("/endpoints/123"))
        with self.assertRaises(ValidationError):
            await garden.get_endpoint("123:delete")

    async def test_model_garden_keeps_publisher_routes_and_native_bodies(self):
        seen = []

        async def fetch(url, **kwargs):
            seen.append((url, kwargs))
            return FakeResponse(200, {"content": []})

        p = create_vertex(
            access_token="token", project_id="p", location="global", fetch=fetch
        )
        garden = p.native.model_garden()
        body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 32}
        await garden.anthropic_messages(model="claude-sonnet-4-6", body=body)
        await garden.chat_completions(
            {"model": "google/gemini-2.5-flash", "messages": []}
        )
        await garden.list_models(publisher="meta")
        self.assertTrue(
            seen[0][0].endswith(
                "/publishers/anthropic/models/claude-sonnet-4-6:rawPredict"
            )
        )
        self.assertEqual(
            seen[0][1]["json_body"]["anthropic_version"], "vertex-2023-10-16"
        )
        self.assertNotIn("anthropic_version", body)
        self.assertTrue(seen[1][0].endswith("/endpoints/openapi/chat/completions"))
        self.assertEqual(
            seen[2][0],
            "https://aiplatform.googleapis.com/v1beta1/publishers/meta/models",
        )

    async def test_cloud_retry_and_non_retryable_error_are_preserved(self):
        from zhivex_ai.types import RetryOptions
        from zhivex_ai.errors import ProviderHTTPError

        responses = [
            FakeResponse(503, {}),
            FakeResponse(200, {"state": "JOB_STATE_SUCCEEDED"}),
        ]
        calls = []

        async def fetch(url, **kwargs):
            calls.append(kwargs)
            return responses.pop(0)

        p = create_vertex(access_token="t", project_id="p", fetch=fetch)
        result = await p.batches().retrieve(
            "b", RetryOptions(max_retries=1, retry_backoff_ms=0, timeout_ms=100)
        )
        self.assertEqual(result["state"], "JOB_STATE_SUCCEEDED")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["timeout_ms"], 100)
        responses.append(FakeResponse(403, {}))
        with self.assertRaises(ProviderHTTPError) as error:
            await p.batches().retrieve("b", RetryOptions(max_retries=2))
        self.assertEqual(error.exception.status, 403)
        self.assertEqual(len(calls), 3)

    async def test_adc_project_and_refresh_failure_do_not_expose_details(self):
        class Credentials:
            def before_request(self, request, method, url, headers):
                raise RuntimeError("secret-provider-diagnostic")

        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "zhivex_ai.providers.vertex.default_credentials",
                return_value=(Credentials(), "adc-project"),
            ),
        ):
            p = create_vertex()
        with self.assertRaises(ConfigurationError) as error:
            await p.tokens().count(model_id="gemini-2.5-flash", prompt="hi")
        self.assertNotIn("secret-provider-diagnostic", str(error.exception))
        self.assertIsNone(error.exception.__cause__)

    async def test_cloud_crud_queries_and_terminal_failures(self):
        seen = []

        async def fetch(url, **kwargs):
            seen.append((url, kwargs))
            if kwargs.get("method") == "DELETE":
                return FakeResponse(204, None)
            return FakeResponse(
                200,
                {
                    "done": True,
                    "error": {"code": 7},
                    "state": "JOB_STATE_FAILED",
                    "status": "requires_action",
                },
            )

        p = create_vertex(access_token="t", project_id="p", fetch=fetch)
        platform = p.native.agent_platform()
        sessions = platform.sessions("a")
        await sessions.get("s")
        await sessions.list(page_size=2, page_token="a+b", filter='userId="u"')
        await sessions.update("s", {"displayName": "name"}, update_mask="displayName")
        await sessions.events("s", page_size=3)
        await sessions.delete("s")
        self.assertIn("pageToken=a%2Bb", seen[1][0])
        self.assertEqual(seen[2][1]["method"], "PATCH")
        self.assertIn("updateMask=displayName", seen[2][0])
        self.assertTrue(seen[3][0].endswith("/sessions/s/events?pageSize=3"))
        self.assertEqual(
            (await platform.wait_operation("operations/op"))["error"]["code"], 7
        )
        self.assertEqual((await p.batches().wait("b"))["state"], "JOB_STATE_FAILED")
        self.assertEqual(
            (await p.interactions().wait("i"))["status"], "requires_action"
        )

    async def test_veo_cloud_storage_result_and_download(self):
        seen = []

        async def fetch(url, **kwargs):
            seen.append((url, kwargs))
            return FakeResponse(
                200,
                {
                    "done": True,
                    "response": {
                        "videos": [
                            {
                                "gcsUri": "gs://bucket/folder/video.mp4",
                                "mimeType": "video/mp4",
                            }
                        ]
                    },
                },
            )

        provider = create_vertex(access_token="token", project_id="p", fetch=fetch)
        op = await provider.videos().get_operation(
            "projects/p/locations/us-central1/publishers/google/models/veo/operations/op"
        )
        self.assertEqual(
            op.raw_response["generated_media"][0].file_uri,
            "gs://bucket/folder/video.mp4",
        )
        await provider.videos().download("gs://bucket/folder/video.mp4")
        self.assertEqual(
            seen[1][0],
            "https://storage.googleapis.com/storage/v1/b/bucket/o/folder%2Fvideo.mp4?alt=media",
        )
        self.assertEqual(seen[1][1]["headers"]["authorization"], "Bearer token")

    async def test_vertex_interactions_preserves_vertex_specific_fields(self):
        seen = []

        async def fetch(url, **kwargs):
            seen.append((url, kwargs))
            return FakeResponse(200, {"id": "i"})

        p = create_vertex(
            access_token="t", project_id="p", location="global", fetch=fetch
        )
        await p.interactions().create(
            {
                "model": "lyria-3-clip-preview",
                "input": "tone",
                "response_mime_type": "audio/mpeg",
            }
        )
        self.assertEqual(seen[0][1]["json_body"]["response_mime_type"], "audio/mpeg")
        await p.interactions().retrieve("i", stream=True)
        self.assertIn("stream=true", seen[1][0])

    async def test_cache_update_sends_field_mask(self):
        seen = []

        async def fetch(url, **kwargs):
            seen.append(url)
            return FakeResponse(200, {"name": "cachedContents/c"})

        p = create_vertex(access_token="t", project_id="p", fetch=fetch)
        await p.caches().update("cachedContents/c", {"ttl": "60s"})
        self.assertTrue(seen[0].endswith("cachedContents/c?updateMask=ttl"))

    async def test_lyria_uses_versioned_vertex_routes_and_normalizes_audio(self):
        seen = []

        async def fetch(url, **kwargs):
            seen.append((url, kwargs))
            if ":predict" in url:
                return FakeResponse(
                    200,
                    {
                        "predictions": [
                            {"audioContent": "bXVzaWM=", "mimeType": "audio/wav"}
                        ]
                    },
                )
            return FakeResponse(
                200,
                {
                    "steps": [
                        {
                            "content": [
                                {
                                    "type": "audio",
                                    "data": "bXVzaWM=",
                                    "mime_type": "audio/mpeg",
                                }
                            ]
                        }
                    ]
                },
            )

        p = create_vertex(
            access_token="t", project_id="p", location="global", fetch=fetch
        )
        clip = await p.media().generate_music(
            prompt="tone", model="lyria-3-clip-preview"
        )
        old = await p.media().generate_music(prompt="tone", model="lyria-002")
        self.assertTrue(
            seen[0][0].endswith("/v1beta1/projects/p/locations/global/interactions")
        )
        self.assertIs(seen[0][1]["json_body"]["store"], False)
        self.assertEqual(clip.media[0].media_type, "audio/mpeg")
        self.assertTrue(
            seen[1][0].endswith("/publishers/google/models/lyria-002:predict")
        )
        self.assertEqual(old.media[0].b64_data, "bXVzaWM=")

    async def test_lyria_two_accepts_live_prediction_bytes_shape(self):
        async def fetch(url, **kwargs):
            return FakeResponse(200, {"predictions": [{"bytesBase64Encoded": "bXVzaWM="}]})

        p = create_vertex(access_token="t", project_id="p", fetch=fetch)
        result = await p.media().generate_music(prompt="tone", model="lyria-002")
        self.assertEqual(len(result.media), 1)
        self.assertEqual(result.media[0].b64_data, "bXVzaWM=")
        self.assertEqual(result.media[0].media_type, "audio/wav")
