"""Claude 5.5 reasoning controls on the direct Anthropic API (requires a key)."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

from zhivex_ai import ReasoningConfig, aclose_default_clients, create_anthropic, generate_text  # noqa: E402


async def main() -> None:
    provider = create_anthropic()
    try:
        opus = await generate_text(
            model=provider("claude-opus-5-5"),
            prompt="Explain why idempotent operations simplify retries, in two sentences.",
            reasoning=ReasoningConfig(effort="medium"),
            max_tokens=1024,
        )
        print(opus.text)
        sonnet = await generate_text(
            model=provider("claude-sonnet-5-5"),
            prompt="Rewrite this concisely: We will try again if the operation fails.",
            reasoning=ReasoningConfig(effort="none"),
            max_tokens=1024,
        )
        print(sonnet.text)
    finally:
        await aclose_default_clients()


if __name__ == "__main__":
    asyncio.run(main())
