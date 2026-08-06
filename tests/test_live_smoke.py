from __future__ import annotations

import io
import os
from contextlib import redirect_stdout
from typing import Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from scripts import run_live_smoke
from zhivex_ai import create_mock_language_model, create_text_message
from zhivex_ai.types import GenerateResult, ModelMessage, ToolCall, ToolCallPart


class LiveSmokeControlTests(IsolatedAsyncioTestCase):
    def test_anthropic_agent_smoke_uses_compatible_sampling_and_budget(self) -> None:
        self.assertEqual(
            run_live_smoke._agent_smoke_generation_options("anthropic"),
            {"temperature": None, "max_tokens": 4096},
        )
        self.assertEqual(
            run_live_smoke._agent_smoke_generation_options("openai"),
            {"temperature": 0, "max_tokens": 80},
        )

    def test_gemini_agent_smoke_allows_reasoning_before_tool_output(self) -> None:
        self.assertEqual(
            run_live_smoke._agent_smoke_generation_options("gemini"),
            {"temperature": 0, "max_tokens": 512},
        )

    def test_selected_providers_normalizes_azure_alias(self) -> None:
        with patch.dict(os.environ, {"ZHIVEX_SMOKE_PROVIDERS": "openai,azure"}, clear=True):
            selected = run_live_smoke._selected_providers()

        self.assertEqual(selected, {"openai", "azure-openai"})

    def test_selected_providers_rejects_unknown_values(self) -> None:
        with patch.dict(os.environ, {"ZHIVEX_SMOKE_PROVIDERS": "openai,typo"}, clear=True):
            with self.assertRaisesRegex(ValueError, "Unknown live smoke provider selector"):
                run_live_smoke._selected_providers()

    async def test_main_returns_configuration_error_for_unknown_provider(self) -> None:
        with (
            patch.dict(os.environ, {"ZHIVEX_SMOKE_PROVIDERS": "not-a-provider"}, clear=True),
            patch.object(run_live_smoke, "_load_dotenv_if_available"),
        ):
            status = await run_live_smoke.main()

        self.assertEqual(status, 2)

    async def test_strict_mode_fails_when_every_selected_provider_is_skipped(self) -> None:
        with (
            patch.dict(
                os.environ,
                {
                    "ZHIVEX_SMOKE_PROVIDERS": "openai",
                    "ZHIVEX_SMOKE_STRICT": "1",
                },
                clear=True,
            ),
            patch.object(run_live_smoke, "_load_dotenv_if_available"),
        ):
            status = await run_live_smoke.main()

        self.assertEqual(status, 1)

    async def test_non_strict_mode_keeps_missing_credentials_as_documented_skip(self) -> None:
        with (
            patch.dict(os.environ, {"ZHIVEX_SMOKE_PROVIDERS": "openai"}, clear=True),
            patch.object(run_live_smoke, "_load_dotenv_if_available"),
        ):
            status = await run_live_smoke.main()

        self.assertEqual(status, 0)

    async def test_every_provider_reports_a_four_field_skip_without_configuration(self) -> None:
        runners = [
            run_live_smoke._run_openai,
            run_live_smoke._run_gemini,
            run_live_smoke._run_anthropic,
            run_live_smoke._run_azure_openai,
            run_live_smoke._run_vertex,
            run_live_smoke._run_ollama,
            run_live_smoke._run_qwen,
            run_live_smoke._run_kimi,
            run_live_smoke._run_deepseek,
            run_live_smoke._run_vllm,
        ]

        with patch.dict(os.environ, {}, clear=True):
            outcomes = [await runner() for runner in runners]

        self.assertTrue(all(len(outcome) == 4 for outcome in outcomes))
        self.assertTrue(all(outcome[1] is False for outcome in outcomes))
        self.assertTrue(all(outcome[3] is False for outcome in outcomes))

    async def test_strict_agent_mode_requires_agent_tool_execution(self) -> None:
        with (
            patch.dict(
                os.environ,
                {
                    "ZHIVEX_SMOKE_PROVIDERS": "openai",
                    "ZHIVEX_SMOKE_STRICT": "1",
                    "ZHIVEX_SMOKE_AGENTS": "1",
                },
                clear=True,
            ),
            patch.object(run_live_smoke, "_load_dotenv_if_available"),
            patch.object(
                run_live_smoke,
                "_run_openai",
                new=AsyncMock(return_value=("openai", True, "ok", False)),
            ),
        ):
            status = await run_live_smoke.main()

        self.assertEqual(status, 1)

    async def test_strict_agent_mode_accepts_agent_tool_execution(self) -> None:
        with (
            patch.dict(
                os.environ,
                {
                    "ZHIVEX_SMOKE_PROVIDERS": "openai",
                    "ZHIVEX_SMOKE_STRICT": "1",
                    "ZHIVEX_SMOKE_AGENTS": "1",
                },
                clear=True,
            ),
            patch.object(run_live_smoke, "_load_dotenv_if_available"),
            patch.object(
                run_live_smoke,
                "_run_openai",
                new=AsyncMock(return_value=("openai", True, "ok, agent-tool=ok", True)),
            ),
        ):
            status = await run_live_smoke.main()

        self.assertEqual(status, 0)

    async def test_agent_tool_smoke_runs_and_validates_local_tool_result(self) -> None:
        model = create_mock_language_model(
            responses=[
                GenerateResult(
                    messages=[
                        ModelMessage(
                            role="assistant",
                            parts=[
                                ToolCallPart(
                                    tool_call=ToolCall(
                                        id="call-agent-smoke",
                                        name="validate_agent_smoke",
                                        input={"nonce": "zhivex-agent-smoke"},
                                    )
                                )
                            ],
                        )
                    ]
                ),
                GenerateResult(
                    text="AGENT_SMOKE_OK",
                    messages=[create_text_message("assistant", "AGENT_SMOKE_OK")],
                    finish_reason="stop",
                ),
            ]
        )
        captured: dict[str, Any] = {}
        real_tool = run_live_smoke.tool

        def capture_tool(**kwargs: Any):
            captured["schema"] = kwargs["schema"]
            return real_tool(**kwargs)

        with patch.object(run_live_smoke, "tool", side_effect=capture_tool):
            await run_live_smoke._run_agent_tool_smoke(provider="test", model=model)

        schema = captured["schema"].model_json_schema()
        self.assertIs(schema["additionalProperties"], False)
        self.assertEqual(schema["required"], ["nonce"])

    async def test_qwen_smoke_accepts_exact_token_without_optional_period(self) -> None:
        provider = MagicMock()
        provider.return_value = object()
        response = GenerateResult(
            text="QWEN_SMOKE_OK",
            messages=[create_text_message("assistant", "QWEN_SMOKE_OK")],
            finish_reason="stop",
        )

        with (
            patch.dict(
                os.environ,
                {
                    "QWEN_API_KEY": "test-key",
                    "ZHIVEX_SMOKE_QWEN_MODEL": "qwen3.8-max",
                },
                clear=True,
            ),
            patch.object(run_live_smoke, "create_qwen", return_value=provider),
            patch.object(run_live_smoke, "generate_text", new=AsyncMock(return_value=response)) as generate,
        ):
            result = await run_live_smoke._run_qwen()

        self.assertEqual(result, ("qwen", True, "ok: qwen3.8-max, region=intl", False))
        self.assertEqual(generate.await_args.kwargs["reasoning"].effort, "none")

    async def test_deepseek_smoke_disables_thinking_for_exact_token_check(self) -> None:
        provider = MagicMock()
        provider.return_value = object()
        response = GenerateResult(
            text="DEEPSEEK_SMOKE_OK.",
            messages=[create_text_message("assistant", "DEEPSEEK_SMOKE_OK.")],
            finish_reason="stop",
        )

        with (
            patch.dict(
                os.environ,
                {
                    "DEEPSEEK_API_KEY": "test-key",
                    "ZHIVEX_SMOKE_DEEPSEEK_MODEL": "deepseek-v4-flash",
                },
                clear=True,
            ),
            patch.object(run_live_smoke, "create_deepseek", return_value=provider),
            patch.object(run_live_smoke, "generate_text", new=AsyncMock(return_value=response)) as generate,
        ):
            result = await run_live_smoke._run_deepseek()

        self.assertEqual(result, ("deepseek", True, "ok: deepseek-v4-flash", False))
        self.assertEqual(generate.await_args.kwargs["reasoning"].effort, "none")

    async def test_agent_tool_smoke_rejects_a_marker_embedded_in_extra_text(self) -> None:
        model = create_mock_language_model(
            responses=[
                GenerateResult(
                    messages=[
                        ModelMessage(
                            role="assistant",
                            parts=[
                                ToolCallPart(
                                    tool_call=ToolCall(
                                        id="call-agent-smoke",
                                        name="validate_agent_smoke",
                                        input={"nonce": "zhivex-agent-smoke"},
                                    )
                                )
                            ],
                        )
                    ]
                ),
                GenerateResult(
                    text="I was told to say AGENT_SMOKE_OK, but validation is inconclusive.",
                    messages=[
                        create_text_message(
                            "assistant",
                            "I was told to say AGENT_SMOKE_OK, but validation is inconclusive.",
                        )
                    ],
                    finish_reason="stop",
                ),
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "unexpected final text"):
            await run_live_smoke._run_agent_tool_smoke(provider="test", model=model)

    def test_safe_error_message_redacts_configured_secrets(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "top-secret-value"}, clear=True):
            message = run_live_smoke._safe_error_message(RuntimeError("failed: top-secret-value"))

        self.assertEqual(message, "failed: [REDACTED]")

    def test_safe_error_message_redacts_url_credentials_paths_and_queries(self) -> None:
        message = run_live_smoke._safe_error_message(
            RuntimeError("failed: https://user:password@example.test/private/token?api_key=secret")
        )

        self.assertEqual(message, "failed: https://example.test/[REDACTED]")

    async def test_main_redacts_unexpected_provider_failure(self) -> None:
        output = io.StringIO()
        with (
            patch.dict(
                os.environ,
                {"ZHIVEX_SMOKE_PROVIDERS": "openai", "OPENAI_API_KEY": "top-secret-value"},
                clear=True,
            ),
            patch.object(run_live_smoke, "_load_dotenv_if_available"),
            patch.object(
                run_live_smoke,
                "_run_openai",
                new=AsyncMock(side_effect=ValueError("unexpected top-secret-value")),
            ),
            redirect_stdout(output),
        ):
            status = await run_live_smoke.main()

        self.assertEqual(status, 1)
        self.assertNotIn("top-secret-value", output.getvalue())
        self.assertIn("[REDACTED]", output.getvalue())
