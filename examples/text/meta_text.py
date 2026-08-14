from __future__ import annotations

import asyncio
import os

from zhivex_ai import create_meta, generate_text


async def main() -> None:
    model_id = os.getenv("ZHIVEX_EXAMPLE_META_MODEL", "muse-spark-1.2")
    meta = create_meta()  # Reads MODEL_API_KEY; defaults to https://api.meta.ai/v1.
    result = await generate_text(
        model=meta(model_id),
        prompt="Explain in one sentence what makes a provider API portable.",
        max_tokens=120,
    )
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
