from __future__ import annotations

import io
import json
import os
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
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
            {"temperature": None, "max_tokens": 512},
        )
        self.assertEqual(
            run_live_smoke._agent_smoke_generation_options("meta"),
            {"temperature": None, "max_tokens": 1024},
        )

    def test_gemini_agent_smoke_allows_reasoning_before_tool_output(self) -> None:
        self.assertEqual(
            run_live_smoke._agent_smoke_generation_options("gemini"),
            {"temperature": None, "max_tokens": 512},
        )

    def test_openai_luna_smoke_disables_reasoning_for_cost_and_determinism(self) -> None:
        luna = run_live_smoke._openai_smoke_reasoning("gpt-5.6-luna")

        self.assertIsNotNone(luna)
        self.assertEqual(luna.effort, "none")
        self.assertIsNone(run_live_smoke._openai_smoke_reasoning("gpt-4.1-mini"))

    def test_selected_providers_normalizes_azure_alias(self) -> None:
        with patch.dict(os.environ, {"ZHIVEX_SMOKE_PROVIDERS": "openai,azure"}, clear=True):
            selected = run_live_smoke._selected_providers()

        self.assertEqual(selected, {"openai", "azure-openai"})

    def test_selected_providers_accepts_meta(self) -> None:
        with patch.dict(os.environ, {"ZHIVEX_SMOKE_PROVIDERS": "meta"}, clear=True):
            self.assertEqual(run_live_smoke._selected_providers(), {"meta"})

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

    async def test_strict_mode_requires_every_explicitly_selected_provider(self) -> None:
        output = io.StringIO()
        with (
            patch.dict(
                os.environ,
                {
                    "ZHIVEX_SMOKE_PROVIDERS": "openai,gemini",
                    "ZHIVEX_SMOKE_STRICT": "1",
                },
                clear=True,
            ),
            patch.object(run_live_smoke, "_load_dotenv_if_available"),
            patch.object(
                run_live_smoke,
                "_run_openai",
                new=AsyncMock(return_value=("openai", True, "ok", False)),
            ),
            patch.object(
                run_live_smoke,
                "_run_gemini",
                new=AsyncMock(return_value=("gemini", False, "skip: missing configuration", False)),
            ),
            redirect_stdout(output),
        ):
            status = await run_live_smoke.main()

        self.assertEqual(status, 1)
        self.assertIn("every explicitly selected provider", output.getvalue())
        self.assertIn("missing: gemini", output.getvalue())

    async def test_strict_mode_without_selector_keeps_at_least_one_provider_semantics(self) -> None:
        skipped = AsyncMock(return_value=("skipped", False, "skip: missing configuration", False))
        with (
            patch.dict(os.environ, {"ZHIVEX_SMOKE_STRICT": "1"}, clear=True),
            patch.object(run_live_smoke, "_load_dotenv_if_available"),
            patch.object(
                run_live_smoke,
                "_run_openai",
                new=AsyncMock(return_value=("openai", True, "ok", False)),
            ),
            patch.object(run_live_smoke, "_run_gemini", new=skipped),
            patch.object(run_live_smoke, "_run_anthropic", new=skipped),
            patch.object(run_live_smoke, "_run_azure_openai", new=skipped),
            patch.object(run_live_smoke, "_run_vertex", new=skipped),
            patch.object(run_live_smoke, "_run_ollama", new=skipped),
            patch.object(run_live_smoke, "_run_qwen", new=skipped),
            patch.object(run_live_smoke, "_run_kimi", new=skipped),
            patch.object(run_live_smoke, "_run_deepseek", new=skipped),
            patch.object(run_live_smoke, "_run_meta", new=skipped),
            patch.object(run_live_smoke, "_run_vllm", new=skipped),
        ):
            status = await run_live_smoke.main()

        self.assertEqual(status, 0)

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
            run_live_smoke._run_meta,
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

    async def test_strict_agent_mode_requires_agent_smoke_for_every_executed_selection(self) -> None:
        output = io.StringIO()
        with (
            patch.dict(
                os.environ,
                {
                    "ZHIVEX_SMOKE_PROVIDERS": "openai,gemini",
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
            patch.object(
                run_live_smoke,
                "_run_gemini",
                new=AsyncMock(return_value=("gemini", True, "ok", False)),
            ),
            redirect_stdout(output),
        ):
            status = await run_live_smoke.main()

        self.assertEqual(status, 1)
        self.assertIn("agent tool smoke for every selected provider", output.getvalue())
        self.assertIn("missing: gemini", output.getvalue())

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

    async def test_second_cohort_executes_portable_certification_instead_of_inferencing_it(self) -> None:
        cases = (
            (
                "qwen",
                run_live_smoke._run_qwen,
                "create_qwen",
                {"QWEN_API_KEY": "test-key", "ZHIVEX_SMOKE_QWEN_MODEL": "qwen3.8-max"},
                "QWEN_SMOKE_OK",
            ),
            (
                "kimi",
                run_live_smoke._run_kimi,
                "create_kimi",
                {"MOONSHOT_API_KEY": "test-key", "ZHIVEX_SMOKE_KIMI_MODEL": "kimi-k3"},
                "KIMI_SMOKE_OK",
            ),
            (
                "deepseek",
                run_live_smoke._run_deepseek,
                "create_deepseek",
                {
                    "DEEPSEEK_API_KEY": "test-key",
                    "ZHIVEX_SMOKE_DEEPSEEK_MODEL": "deepseek-v4-flash",
                },
                "DEEPSEEK_SMOKE_OK",
            ),
        )
        for provider_name, runner, factory_name, provider_env, token in cases:
            with self.subTest(provider=provider_name):
                provider = MagicMock()
                language_model = MagicMock(model_id=next(iter(provider_env.values())))
                provider.return_value = language_model
                response = GenerateResult(
                    text=token,
                    messages=[create_text_message("assistant", token)],
                    finish_reason="stop",
                )
                with (
                    patch.dict(
                        os.environ,
                        {**provider_env, "ZHIVEX_SMOKE_PORTABLE_CERTIFICATION": "1"},
                        clear=True,
                    ),
                    patch.object(run_live_smoke, factory_name, return_value=provider),
                    patch.object(
                        run_live_smoke,
                        "generate_text",
                        new=AsyncMock(return_value=response),
                    ),
                    patch.object(
                        run_live_smoke,
                        "_run_portable_certification",
                        new=AsyncMock(),
                    ) as certification,
                ):
                    result = await runner()

                self.assertIn("portable-certification=ok", result[2])
                certification.assert_awaited_once()
                self.assertEqual(certification.await_args.kwargs["provider"], provider_name)
                self.assertIs(certification.await_args.kwargs["model"], language_model)
                if provider_name == "qwen":
                    self.assertIsNone(
                        certification.await_args.kwargs["structured_max_tokens"]
                    )

    async def test_vllm_certification_executes_portable_and_retrieval_operations(self) -> None:
        provider = MagicMock()
        language_model = MagicMock(model_id="Qwen/Qwen2.5-0.5B-Instruct")
        provider.return_value = language_model
        response = GenerateResult(
            text="VLLM_SMOKE_OK",
            messages=[create_text_message("assistant", "VLLM_SMOKE_OK")],
            finish_reason="stop",
        )
        with (
            patch.dict(
                os.environ,
                {
                    "ZHIVEX_SMOKE_VLLM_MODEL": "Qwen/Qwen2.5-0.5B-Instruct",
                    "ZHIVEX_SMOKE_PORTABLE_CERTIFICATION": "1",
                },
                clear=True,
            ),
            patch.object(run_live_smoke, "create_vllm", return_value=provider),
            patch.object(
                run_live_smoke,
                "generate_text",
                new=AsyncMock(return_value=response),
            ),
            patch.object(
                run_live_smoke,
                "_run_portable_certification",
                new=AsyncMock(),
            ) as portable,
            patch.object(
                run_live_smoke,
                "_run_portable_retrieval_certification",
                new=AsyncMock(),
            ) as retrieval,
        ):
            result = await run_live_smoke._run_vllm()

        self.assertIn("portable-certification=ok", result[2])
        self.assertIn("portable-retrieval=ok", result[2])
        portable.assert_awaited_once()
        retrieval.assert_awaited_once()
        self.assertEqual(portable.await_args.kwargs["provider"], "vllm")
        self.assertIs(portable.await_args.kwargs["model"], language_model)
        self.assertIs(
            portable.await_args.kwargs["completed_operations"],
            retrieval.await_args.kwargs["completed_operations"],
        )

    async def test_vllm_failure_preserves_completed_operations_for_evidence(self) -> None:
        provider = MagicMock()
        language_model = MagicMock(model_id="Qwen/Qwen2.5-0.5B-Instruct")
        provider.return_value = language_model
        response = GenerateResult(
            text="VLLM_SMOKE_OK",
            messages=[create_text_message("assistant", "VLLM_SMOKE_OK")],
            finish_reason="stop",
        )

        async def complete_portable(**kwargs: Any) -> None:
            kwargs["completed_operations"].update({"streaming", "structured-output"})

        async def complete_retrieval(**kwargs: Any) -> None:
            kwargs["completed_operations"].add("portable-retrieval")

        with (
            patch.dict(
                os.environ,
                {
                    "ZHIVEX_SMOKE_VLLM_MODEL": "Qwen/Qwen2.5-0.5B-Instruct",
                    "ZHIVEX_SMOKE_PORTABLE_CERTIFICATION": "1",
                    "ZHIVEX_SMOKE_AGENTS": "1",
                },
                clear=True,
            ),
            patch.object(run_live_smoke, "create_vllm", return_value=provider),
            patch.object(
                run_live_smoke,
                "generate_text",
                new=AsyncMock(return_value=response),
            ),
            patch.object(
                run_live_smoke,
                "_run_portable_certification",
                new=AsyncMock(side_effect=complete_portable),
            ),
            patch.object(
                run_live_smoke,
                "_run_portable_retrieval_certification",
                new=AsyncMock(side_effect=complete_retrieval),
            ),
            patch.object(
                run_live_smoke,
                "_run_agent_tool_smoke",
                new=AsyncMock(side_effect=RuntimeError("unexpected final text")),
            ),
        ):
            with self.assertRaises(run_live_smoke._ProviderOperationError) as caught:
                await run_live_smoke._run_vllm()

        self.assertEqual(
            caught.exception.completed_operations,
            frozenset({"generation", "streaming", "structured-output", "portable-retrieval"}),
        )

    async def test_openai_luna_smoke_uses_cost_oriented_reasoning(self) -> None:
        provider = MagicMock()
        language_model = MagicMock(model_id="gpt-5.6-luna")
        provider.return_value = language_model
        response = GenerateResult(
            text="OPENAI_SMOKE_OK.",
            messages=[create_text_message("assistant", "OPENAI_SMOKE_OK.")],
            finish_reason="stop",
        )

        with (
            patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "test-key",
                    "ZHIVEX_SMOKE_OPENAI_MODEL": "gpt-5.6-luna",
                },
                clear=True,
            ),
            patch.object(run_live_smoke, "create_openai", return_value=provider),
            patch.object(
                run_live_smoke,
                "generate_text",
                new=AsyncMock(return_value=response),
            ) as generate,
        ):
            result = await run_live_smoke._run_openai()

        self.assertEqual(result, ("openai", True, "ok: gpt-5.6-luna", False))
        self.assertEqual(generate.await_args.kwargs["reasoning"].effort, "none")

    async def test_portable_certification_covers_stream_and_structured_output(self) -> None:
        language_model = MagicMock(model_id="gpt-5.6-luna")
        stream_result = MagicMock()
        stream_result.collect = AsyncMock(
            return_value=GenerateResult(
                text="ZHIVEX_OPENAI_STREAM_OK",
                messages=[create_text_message("assistant", "ZHIVEX_OPENAI_STREAM_OK")],
                finish_reason="stop",
            )
        )
        structured_result = MagicMock()
        structured_result.object.nonce = "zhivex-openai-structured-smoke"

        with (
            patch.object(run_live_smoke, "stream_text", return_value=stream_result) as stream,
            patch.object(
                run_live_smoke,
                "generate_object",
                new=AsyncMock(return_value=structured_result),
            ) as generate_structured,
        ):
            await run_live_smoke._run_portable_certification(
                provider="openai",
                model=language_model,
                reasoning=run_live_smoke.ReasoningConfig(effort="none"),
            )

        self.assertEqual(stream.call_args.kwargs["reasoning"].effort, "none")
        self.assertEqual(generate_structured.await_args.kwargs["reasoning"].effort, "none")
        self.assertEqual(
            generate_structured.await_args.kwargs["schema_name"],
            "openai_release_smoke",
        )

    def test_release_policy_requires_the_exact_artifact_hash(self) -> None:
        policy = {
            "schema_version": 1,
            "package_version": run_live_smoke._source_package_version(),
            "artifact_sha256": "a" * 64,
            "require_artifact_sha256": True,
            "required_providers": {"openai": {"operations": ["generation"]}},
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {
                    "ZHIVEX_SMOKE_PROVIDERS": "openai",
                    "ZHIVEX_SMOKE_STRICT": "1",
                    "ZHIVEX_SMOKE_ARTIFACT_PATH": str(
                        Path(directory) / "zhivex_ai_sdk-0.22.0-py3-none-any.whl"
                    ),
                },
                clear=True,
            ),
        ):
            artifact = Path(directory) / "zhivex_ai_sdk-0.22.0-py3-none-any.whl"
            artifact.write_bytes(b"different-wheel")
            with self.assertRaisesRegex(ValueError, "SHA256 does not match"):
                run_live_smoke._validate_release_smoke_configuration(
                    policy,
                    selected={"openai"},
                )

    async def test_meta_smoke_uses_standard_opt_in_configuration(self) -> None:
        provider = MagicMock()
        provider.return_value = object()
        response = GenerateResult(
            text="META_SMOKE_OK.",
            messages=[create_text_message("assistant", "META_SMOKE_OK.")],
            finish_reason="stop",
        )

        with (
            patch.dict(
                os.environ,
                {
                    "MODEL_API_KEY": "test-key",
                    "ZHIVEX_SMOKE_META_MODEL": "muse-spark-1.2",
                },
                clear=True,
            ),
            patch.object(run_live_smoke, "create_meta", return_value=provider) as create,
            patch.object(run_live_smoke, "generate_text", new=AsyncMock(return_value=response)) as generate,
        ):
            result = await run_live_smoke._run_meta()

        self.assertEqual(result, ("meta", True, "ok: muse-spark-1.2", False))
        create.assert_called_once_with(api_key="test-key")
        self.assertEqual(generate.await_args.kwargs["reasoning"].effort, "low")
        self.assertEqual(generate.await_args.kwargs["max_tokens"], 512)

    async def test_meta_certification_covers_stream_object_retrieval_and_agent_tool(self) -> None:
        provider = MagicMock()
        language_model = MagicMock(model_id="muse-spark-1.2-contributor")
        provider.return_value = language_model
        generation = GenerateResult(
            text="META_SMOKE_OK",
            messages=[create_text_message("assistant", "META_SMOKE_OK")],
            finish_reason="stop",
        )
        retrieval = GenerateResult(
            text="META_RETRIEVAL_SMOKE_OK",
            messages=[create_text_message("assistant", "META_RETRIEVAL_SMOKE_OK")],
            finish_reason="stop",
        )
        stream_result = MagicMock()
        stream_result.collect = AsyncMock(
            return_value=GenerateResult(
                text="META_STREAM_SMOKE_OK",
                messages=[create_text_message("assistant", "META_STREAM_SMOKE_OK")],
                finish_reason="stop",
            )
        )
        structured_result = MagicMock()
        structured_result.object.nonce = "meta-structured-smoke"

        with (
            patch.dict(
                os.environ,
                {
                    "MODEL_API_KEY": "test-key",
                    "ZHIVEX_SMOKE_META_MODEL": "muse-spark-1.2-contributor",
                    "ZHIVEX_SMOKE_META_CERTIFICATION": "1",
                    "ZHIVEX_SMOKE_AGENTS": "1",
                },
                clear=True,
            ),
            patch.object(run_live_smoke, "create_meta", return_value=provider),
            patch.object(
                run_live_smoke,
                "generate_text",
                new=AsyncMock(side_effect=[generation, retrieval]),
            ) as generate,
            patch.object(run_live_smoke, "stream_text", return_value=stream_result) as stream,
            patch.object(
                run_live_smoke,
                "generate_object",
                new=AsyncMock(return_value=structured_result),
            ) as generate_structured,
            patch.object(
                run_live_smoke,
                "_run_agent_tool_smoke",
                new=AsyncMock(),
            ) as agent_smoke,
        ):
            result = await run_live_smoke._run_meta()

        self.assertEqual(
            result,
            (
                "meta",
                True,
                "ok: muse-spark-1.2-contributor, portable-certification=ok, agent-tool=ok",
                True,
            ),
        )
        self.assertEqual(generate.await_count, 2)
        self.assertEqual(generate.await_args_list[0].kwargs["reasoning"].effort, "low")
        self.assertEqual(generate.await_args_list[1].kwargs["reasoning"].effort, "low")
        self.assertIsNotNone(generate.await_args_list[1].kwargs["retrieval"])
        generate_structured.assert_awaited_once()
        self.assertEqual(generate_structured.await_args.kwargs["reasoning"].effort, "low")
        self.assertEqual(generate_structured.await_args.kwargs["max_tokens"], 512)
        self.assertEqual(stream.call_args.kwargs["reasoning"].effort, "low")
        self.assertEqual(stream.call_args.kwargs["max_tokens"], 512)
        agent_smoke.assert_awaited_once_with(provider="meta", model=language_model)

    async def test_release_policy_rejects_an_omitted_required_provider(self) -> None:
        policy = {
            "schema_version": 1,
            "package_version": run_live_smoke._source_package_version(),
            "required_providers": {
                "openai": {"operations": ["generation"]},
                "meta": {"operations": ["generation"]},
            },
        }
        output = io.StringIO()
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
            patch.object(run_live_smoke, "_load_release_smoke_policy", return_value=policy),
            redirect_stdout(output),
        ):
            status = await run_live_smoke.main()

        self.assertEqual(status, 2)
        self.assertIn("missing: meta", output.getvalue())

    def test_release_evidence_records_artifact_hash_without_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "zhivex_ai_sdk-0.22.0-py3-none-any.whl"
            artifact.write_bytes(b"exact-wheel")
            evidence = Path(directory) / "smoke.json"
            with patch.dict(
                os.environ,
                {
                    "MODEL_API_KEY": "never-record-this-secret",
                    "ZHIVEX_SMOKE_ARTIFACT_PATH": str(artifact),
                    "ZHIVEX_SMOKE_EVIDENCE_PATH": str(evidence),
                    "ZHIVEX_SMOKE_META_MODEL": "muse-spark-1.2-contributor",
                    "GITHUB_SHA": "a" * 40,
                    "GITHUB_REPOSITORY": "Zhivex/zhivex-ai-sdk-py",
                    "GITHUB_RUN_ID": "123456",
                    "GITHUB_RUN_ATTEMPT": "1",
                    "GITHUB_WORKFLOW": "Publish to PyPI",
                },
                clear=True,
            ):
                run_live_smoke._write_release_smoke_evidence(
                    policy={"package_version": run_live_smoke._source_package_version()},
                    executed_operations={"meta": {"generation", "agent-tool"}},
                    failures=0,
                )
            payload = json.loads(evidence.read_text("utf-8"))
            serialized = evidence.read_text("utf-8")

        self.assertEqual(payload["run_status"], "passed")
        self.assertEqual(payload["artifact"]["source_revision"], "a" * 40)
        self.assertEqual(payload["artifact"]["filename"], artifact.name)
        self.assertEqual(len(payload["artifact"]["sha256"]), 64)
        self.assertEqual(
            payload["targets"][0]["model"],
            "muse-spark-1.2-contributor",
        )
        self.assertEqual(payload["targets"][0]["target_id"], "meta-contributor")
        self.assertEqual(payload["targets"][0]["surface"], "contributor")
        self.assertEqual(payload["workflow"]["platform"], "github-actions")
        self.assertNotIn("never-record-this-secret", serialized)

    def test_blocked_second_cohort_evidence_records_each_operation_without_false_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "zhivex_ai_sdk-0.22.0-py3-none-any.whl"
            artifact.write_bytes(b"exact-wheel")
            evidence = Path(directory) / "smoke.json"
            policy = {
                "package_version": run_live_smoke._source_package_version(),
                "required_providers": {
                    "qwen": {
                        "operations": [
                            "generation",
                            "streaming",
                            "structured-output",
                            "agent-tool",
                        ],
                        "unsupported_operations": ["portable-retrieval"],
                    }
                },
            }
            with patch.dict(
                os.environ,
                {
                    "ZHIVEX_SMOKE_ARTIFACT_PATH": str(artifact),
                    "ZHIVEX_SMOKE_EVIDENCE_PATH": str(evidence),
                    "ZHIVEX_SMOKE_QWEN_MODEL": "qwen3.8-max",
                    "GITHUB_SHA": "a" * 40,
                },
                clear=True,
            ):
                run_live_smoke._write_release_smoke_evidence(
                    policy=policy,
                    executed_operations={"qwen": {"generation"}},
                    blocked_providers={"qwen": "QWEN_CREDENTIALS_UNAVAILABLE"},
                    failures=1,
                )

            target = json.loads(evidence.read_text("utf-8"))["targets"][0]

        self.assertEqual(target["result"], "blocked")
        self.assertEqual(
            {operation["name"]: operation["status"] for operation in target["operations"]},
            {
                "agent-tool": "blocked",
                "generation": "passed",
                "portable-retrieval": "unsupported",
                "streaming": "blocked",
                "structured-output": "blocked",
            },
        )

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

    def test_certification_diagnostics_are_allowlisted_without_response_bodies(self) -> None:
        class ConnectError(RuntimeError):
            pass

        anthropic_error = RuntimeError("do not persist this body")
        anthropic_error.response_body = (  # type: ignore[attr-defined]
            '{"error":{"message":"anthropic-workspace-id is required"}}'
        )
        gemini_error = RuntimeError("do not persist this body")
        gemini_error.response_body = (  # type: ignore[attr-defined]
            '{"error":{"message":"API key not valid"}}'
        )

        self.assertEqual(
            run_live_smoke._failure_diagnostic_code("anthropic", anthropic_error),
            "ANTHROPIC_WORKSPACE_ID_REQUIRED",
        )
        self.assertEqual(
            run_live_smoke._failure_diagnostic_code("gemini", gemini_error),
            "GEMINI_API_KEY_INVALID",
        )
        self.assertEqual(
            run_live_smoke._blocked_diagnostic_code("azure-openai"),
            "AZURE_CREDENTIALS_UNAVAILABLE",
        )
        self.assertEqual(
            run_live_smoke._blocked_diagnostic_code("vertex"),
            "VERTEX_CREDENTIALS_UNAVAILABLE",
        )
        self.assertEqual(
            run_live_smoke._failure_diagnostic_code("vllm", ConnectError("offline")),
            "VLLM_DEPLOYMENT_UNAVAILABLE",
        )
        self.assertTrue(
            run_live_smoke._is_external_blocker("ANTHROPIC_WORKSPACE_ID_REQUIRED")
        )
        self.assertTrue(run_live_smoke._is_external_blocker("GEMINI_API_KEY_INVALID"))
        self.assertFalse(run_live_smoke._is_external_blocker("PROVIDER_EXECUTION_FAILED"))

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
