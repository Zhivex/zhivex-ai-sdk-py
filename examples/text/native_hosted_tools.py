from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from _bootstrap import load_dotenv_if_available

load_dotenv_if_available()

from zhivex_ai import (  # noqa: E402
    azure_openai_web_search_tool,
    create_azure_openai,
    create_openai,
    generate_text,
    openai_web_search_tool,
    tool,
)


class WeatherInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str


def _resolve_provider() -> tuple[str, Any, str, Any]:
    provider_name = os.getenv("PROVIDER", "openai").strip().lower()

    if provider_name == "openai":
        provider = create_openai()
        model_name = os.getenv("MODEL", "gpt-5.6-terra")
        hosted_search = openai_web_search_tool(search_context_size="high")
        return provider_name, provider, model_name, hosted_search

    if provider_name in {"azure", "azure-openai"}:
        provider = create_azure_openai()
        model_name = os.getenv("MODEL", "gpt-5.6-terra")
        hosted_search = azure_openai_web_search_tool(search_context_size="high")
        return "azure-openai", provider, model_name, hosted_search

    raise RuntimeError(
        f'Unsupported PROVIDER="{provider_name}" for this example. '
        'Use PROVIDER=openai or PROVIDER=azure-openai.'
    )


async def main() -> None:
    provider_name, provider, model_name, hosted_search = _resolve_provider()

    result = await generate_text(
        model=provider.native.language_model(model_name),
        prompt=(
            "Compare today's weather in Buenos Aires with one current news theme you can verify online. "
            "Use the local weather tool for the forecast and the hosted search tool for the news context."
        ),
        tools={
            "weather": tool(
                name="weather",
                description="Returns a tiny forecast snapshot for a city.",
                schema=WeatherInput,
                execute=lambda input: {"city": input.city, "forecast": "18C and cloudy"},
            ),
            "search": hosted_search,
        },
    )

    print(f"provider={provider_name} model={model_name}")
    print(result.text)
    if result.tool_results:
        print("local tool results:")
        for item in result.tool_results:
            print(f"- {item.tool_name}: {item.output}")


if __name__ == "__main__":
    asyncio.run(main())
