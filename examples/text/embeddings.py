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

from zhivex_ai import create_gemini, embed_many


async def main() -> None:
    provider = create_gemini(api_key=os.getenv("GOOGLE_API_KEY"))
    result = await embed_many(
        model=provider.embedding_model("gemini-embedding-001"),
        values=[
            "Zhivex AI SDK normalizes providers.",
            "Embeddings help semantic search.",
        ],
    )

    print("vectors:", len(result.embeddings))
    print("dimension:", len(result.embeddings[0]) if result.embeddings else 0)


if __name__ == "__main__":
    asyncio.run(main())
