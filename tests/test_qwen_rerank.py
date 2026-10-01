from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zhivex_ai import ConfigurationError, ParseError, ProviderHTTPError, UnsupportedFeatureError, ValidationError, create_qwen
from zhivex_ai._http import BufferedResponse
from zhivex_ai.types import RetryOptions


def response(payload, status=200):
    return BufferedResponse(status, json.dumps(payload).encode(), {"x-request-id": "req-rerank"})


class QwenRerankTests(IsolatedAsyncioTestCase):
    async def test_latest_workspace_route_native_shape_and_result(self):
        payload = {"output": {"results": [{"index": 1, "relevance_score": 0.9}]},
                   "usage": {"prompt_tokens": 25, "total_tokens": 25}, "request_id": "req-rerank"}
        fetch = AsyncMock(return_value=response(payload))
        provider = create_qwen(api_key="test", region="cn", workspace_id="workspace-42", fetch=fetch)
        client = provider.native.rerank()
        self.assertIs(client, provider.native.rerank())
        self.assertTrue(provider.native_support.rerank)
        result = await client.create(query="weather", documents=["irrelevant", "sunny"], top_n=1,
                                     instruct="Retrieve semantically similar text.", options=RetryOptions(timeout_ms=1200))
        self.assertEqual(result, payload)
        args, kwargs = fetch.call_args
        self.assertEqual(args[0], "https://workspace-42.cn-beijing.maas.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank")
        self.assertEqual(kwargs["json_body"], {"model": "qwen3.7-text-rerank", "input": {"query": "weather", "documents": ["irrelevant", "sunny"]},
                                             "parameters": {"top_n": 1, "instruct": "Retrieve semantically similar text."}})
        self.assertEqual(kwargs["timeout_ms"], 1200)
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test")

    async def test_legacy_uses_flat_compatible_api_and_workspace_region(self):
        payload = {"results": [{"index": 0, "relevance_score": 0.5}], "usage": {"total_tokens": 3}, "id": "request"}
        fetch = AsyncMock(return_value=response(payload))
        with patch.dict("os.environ", {"DASHSCOPE_WORKSPACE_ID": "workspace-42"}):
            client = create_qwen(api_key="test", region="intl", fetch=fetch).native.rerank()
        result = await client.create(model="qwen3-rerank", query="query", documents=("document",), top_n=8)
        self.assertEqual(result, payload)
        self.assertEqual(fetch.call_args.args[0], "https://workspace-42.ap-southeast-1.maas.aliyuncs.com/compatible-api/v1/reranks")
        self.assertEqual(fetch.call_args.kwargs["json_body"], {"model": "qwen3-rerank", "query": "query", "documents": ["document"], "top_n": 8})

    async def test_routes_default_regions_and_custom_gateway_prefix(self):
        for region, host in (("cn", "dashscope.aliyuncs.com"), ("intl", "dashscope-intl.aliyuncs.com"), ("us", "dashscope-us.aliyuncs.com")):
            with self.subTest(region=region), patch.dict("os.environ", {"DASHSCOPE_WORKSPACE_ID": ""}):
                fetch = AsyncMock(return_value=response({"results": []}))
                await create_qwen(api_key="test", region=region, fetch=fetch).native.rerank().create(model="qwen3-rerank", query="q", documents=["d"])
                self.assertEqual(fetch.call_args.args[0], f"https://{host}/compatible-api/v1/reranks")
        fetch = AsyncMock(return_value=response({"output": {"results": []}}))
        await create_qwen(api_key="test", base_url="https://gateway.example/tenant/compatible-mode/v1", workspace_id="workspace-42", fetch=fetch).native.rerank().create(query="q", documents=["d"])
        self.assertEqual(fetch.call_args.args[0], "https://gateway.example/tenant/api/v1/services/rerank/text-rerank/text-rerank")

    async def test_invalid_input_fails_before_network(self):
        fetch = AsyncMock()
        client = create_qwen(api_key="test", region="cn", fetch=fetch).native.rerank()
        cases = [{"query": ""}, {"query": None}, {"documents": []}, {"documents": "document"},
                 {"documents": ["d"] * 501}, {"documents": [1]}, {"documents": [" "]},
                 {"top_n": 0}, {"top_n": True}, {"top_n": 1.5}, {"instruct": ""},
                 {"model": "qwen3-vl-rerank"}, {"model": "qwen3.7-max"}]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValidationError):
                await client.create(**{"query": "q", "documents": ["d"], **values})
        fetch.assert_not_called()

    async def test_new_model_rejects_unlisted_regions_before_network(self):
        for region in ("intl", "us"):
            for workspace in (None, "workspace-42"):
                with self.subTest(region=region, workspace=workspace):
                    fetch = AsyncMock()
                    with self.assertRaises(UnsupportedFeatureError):
                        await create_qwen(api_key="test", region=region, workspace_id=workspace, fetch=fetch).native.rerank().create(query="q", documents=["d"])
                    fetch.assert_not_called()

    async def test_invalid_json_raises_parse_error(self):
        fetch = AsyncMock(return_value=BufferedResponse(200, b"invalid json"))
        with self.assertRaises(ParseError):
            await create_qwen(api_key="test", region="cn", fetch=fetch).native.rerank().create(query="q", documents=["d"])

    async def test_response_indices_use_snapshot_of_submitted_documents(self):
        documents = ["one", "two"]
        async def fetch(url, **kwargs):
            documents.clear()
            return response({"output": {"results": [{"index": 1, "relevance_score": 0.8}]}})
        result = await create_qwen(api_key="test", region="cn", fetch=fetch).native.rerank().create(query="q", documents=documents)
        self.assertEqual(result["output"]["results"][0]["index"], 1)

    def test_workspace_and_url_validation(self):
        for workspace in ("wrong.label", "bad/path", "-bad", "x" * 64):
            with self.subTest(workspace=workspace), self.assertRaises(ConfigurationError):
                create_qwen(api_key="test", workspace_id=workspace).native.rerank()
        for url in ("http://example.com", "https://user:pass@example.com", "https://example.com?query=1"):
            with self.subTest(url=url), self.assertRaises(ConfigurationError):
                create_qwen(api_key="test", base_url=url).native.rerank()

    async def test_http_and_provider_error_are_not_retried(self):
        for status, payload in ((429, {"message": "limited"}), (200, {"code": "InvalidApiKey", "message": "invalid key"})):
            with self.subTest(status=status):
                fetch = AsyncMock(return_value=response(payload, status))
                with self.assertRaises(ProviderHTTPError) as caught:
                    await create_qwen(api_key="test", region="cn", fetch=fetch).native.rerank().create(query="q", documents=["d"], options=RetryOptions(max_retries=3))
                self.assertEqual(caught.exception.status, status)
                self.assertEqual(caught.exception.response_headers["x-request-id"], "req-rerank")
                self.assertEqual(fetch.await_count, 1)

    async def test_invalid_results_are_not_success(self):
        payloads = [[], {}, {"output": {}}, {"output": {"results": [1]}},
                    {"output": {"results": [{"index": 1, "relevance_score": 0.5}]}},
                    {"output": {"results": [{"index": True, "relevance_score": 0.5}]}},
                    {"output": {"results": [{"index": 0, "relevance_score": float("nan")}]}},
                    {"output": {"results": [{"index": 0, "relevance_score": 2}]}},
                    {"output": {"results": [{"index": 0, "relevance_score": 0.5}] * 2}}]
        for payload in payloads:
            with self.subTest(payload=payload), self.assertRaises(ParseError):
                fetch = AsyncMock(return_value=response(payload))
                await create_qwen(api_key="test", region="cn", fetch=fetch).native.rerank().create(query="q", documents=["d"])
