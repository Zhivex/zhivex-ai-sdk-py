"""OpenAI hosted multi-agent Responses (Beta; requires model access)."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

from zhivex_ai import aclose_default_clients, create_openai, generate_text, stream_text, user  # noqa: E402


async def main() -> None:
    provider = create_openai()
    model = provider.native.language_model("gpt-6.1-sol")
    options = {"multi_agent": {"enabled": True, "max_concurrent_subagents": 3}, "store": False}
    try:
        async with stream_text(
            model=model,
            prompt="Compare SQLite and Postgres for a small agent service; delegate independent research if useful.",
            provider_options=options,
        ) as stream:
            result = await stream.collect()
        print(result.text)
        followup = await generate_text(
            model=model,
            messages=[*result.messages, user("Give me the first three implementation steps.")],
            provider_options=options,
        )
        print(followup.text)
    finally:
        await aclose_default_clients()


if __name__ == "__main__":
    asyncio.run(main())
