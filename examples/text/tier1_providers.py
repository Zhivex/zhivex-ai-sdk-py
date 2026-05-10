from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (  # noqa: E402
    create_anthropic,
    create_azure_openai,
    create_gemini,
    create_kimi,
    create_openai,
    create_qwen,
    create_vertex,
    create_vllm,
    generate_text,
)


def _configured_runs() -> list[tuple[str, Callable[[], Any], str]]:
    runs: list[tuple[str, Callable[[], Any], str]] = []
    if os.getenv("OPENAI_API_KEY") and os.getenv("ZHIVEX_EXAMPLE_OPENAI_MODEL"):
        runs.append(("openai", create_openai, os.environ["ZHIVEX_EXAMPLE_OPENAI_MODEL"]))
    if os.getenv("ANTHROPIC_API_KEY") and os.getenv("ZHIVEX_EXAMPLE_ANTHROPIC_MODEL"):
        runs.append(("anthropic", create_anthropic, os.environ["ZHIVEX_EXAMPLE_ANTHROPIC_MODEL"]))
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("ZHIVEX_EXAMPLE_AZURE_OPENAI_MODEL"):
        runs.append(
            (
                "azure-openai",
                lambda: create_azure_openai(api_key=os.environ["AZURE_OPENAI_API_KEY"], endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]),
                os.environ["ZHIVEX_EXAMPLE_AZURE_OPENAI_MODEL"],
            )
        )
    if os.getenv("GEMINI_API_KEY") and os.getenv("ZHIVEX_EXAMPLE_GEMINI_MODEL"):
        runs.append(("gemini", create_gemini, os.environ["ZHIVEX_EXAMPLE_GEMINI_MODEL"]))
    if (os.getenv("VERTEX_ACCESS_TOKEN") or os.getenv("GOOGLE_ACCESS_TOKEN")) and (
        os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
    ) and os.getenv("ZHIVEX_EXAMPLE_VERTEX_MODEL"):
        runs.append(("vertex", create_vertex, os.environ["ZHIVEX_EXAMPLE_VERTEX_MODEL"]))
    if (os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")) and os.getenv("ZHIVEX_EXAMPLE_QWEN_MODEL"):
        runs.append(("qwen", create_qwen, os.environ["ZHIVEX_EXAMPLE_QWEN_MODEL"]))
    if (os.getenv("MOONSHOT_API_KEY") or os.getenv("KIMI_API_KEY")) and os.getenv("ZHIVEX_EXAMPLE_KIMI_MODEL"):
        runs.append(("kimi", create_kimi, os.environ["ZHIVEX_EXAMPLE_KIMI_MODEL"]))
    if os.getenv("ZHIVEX_EXAMPLE_VLLM_MODEL"):
        runs.append(("vllm", create_vllm, os.environ["ZHIVEX_EXAMPLE_VLLM_MODEL"]))
    return runs


async def main() -> None:
    runs = _configured_runs()
    if not runs:
        print("Set one ZHIVEX_EXAMPLE_*_MODEL and matching provider credentials to run a tier-1 example.")
        return
    for provider_name, create_provider, model_id in runs:
        provider = create_provider()
        result = await generate_text(
            model=provider(model_id),
            prompt=f"Reply with one short sentence identifying the {provider_name} provider.",
        )
        print(f"[{provider_name}] {result.text}")


if __name__ == "__main__":
    asyncio.run(main())
