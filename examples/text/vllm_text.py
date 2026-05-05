from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import create_vllm, generate_text  # noqa: E402


async def main() -> None:
    provider = create_vllm(
        api_key=os.getenv("VLLM_API_KEY"),
        base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
    )
    model = os.getenv("VLLM_MODEL", "NousResearch/Meta-Llama-3-8B-Instruct")
    result = await generate_text(
        model=provider(model),
        prompt="Explain vLLM in one sentence.",
    )
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
