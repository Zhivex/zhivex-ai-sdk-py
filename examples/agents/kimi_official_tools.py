from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from _bootstrap import load_dotenv_if_available

load_dotenv_if_available()

from zhivex_ai import KIMI_OFFICIAL_TOOL_URIS, ReasoningConfig, create_kimi, generate_text  # noqa: E402


async def main() -> None:
    kimi = create_kimi()
    model = os.getenv("KIMI_MODEL", "kimi-k2.6")
    tools = await kimi.formulas().toolset(["moonshot/web-search:latest", "moonshot/date:latest"])

    result = await generate_text(
        model=kimi.native.language_model(model),
        prompt="Use official tools to answer: what is one current Kimi API capability developers should know about?",
        tools=tools,
        max_steps=4,
        reasoning=ReasoningConfig(effort="none"),
    )

    print("available official tool URIs:", ", ".join(KIMI_OFFICIAL_TOOL_URIS))
    print(result.text)
    for item in result.tool_results:
        print(f"{item.tool_name}: {item.output}")


if __name__ == "__main__":
    asyncio.run(main())
