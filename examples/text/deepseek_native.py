from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from _bootstrap import load_dotenv_if_available

load_dotenv_if_available()

from zhivex_ai import ReasoningConfig, create_deepseek, generate_object, generate_text  # noqa: E402


class ProviderSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    supports_reasoning: bool
    supports_tools: bool


async def main() -> None:
    deepseek = create_deepseek()
    model_id = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    model = deepseek.native.language_model(model_id)

    reasoned = await generate_text(
        model=model,
        prompt="Explain DeepSeek support in Zhivex AI SDK in one sentence.",
        reasoning=ReasoningConfig(effort="high"),
    )
    print("reasoning:", reasoned.text)

    structured = await generate_object(
        model=model,
        schema=ProviderSummary,
        prompt="Return a JSON summary of DeepSeek support: provider DeepSeek, reasoning true, tools true.",
        reasoning=ReasoningConfig(effort="none"),
    )
    print("structured:", structured.object.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
