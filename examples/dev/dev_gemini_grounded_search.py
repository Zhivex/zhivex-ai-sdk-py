from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from _bootstrap import load_dotenv_if_available

from zhivex_ai import create_gemini, generate_grounded_text

load_dotenv_if_available()


async def main() -> None:
    gemini = create_gemini(api_key=os.getenv("GOOGLE_API_KEY"))
    result = await generate_grounded_text(
        model=gemini.grounded_language_model("gemini-3-flash-preview"),
        prompt="Find one recent fact about AI infrastructure and cite the source.",
    )

    print(result.text)
    print()
    print("Sources:")
    for source in result.sources:
        print(f"- {source.title}: {source.url}")


if __name__ == "__main__":
    asyncio.run(main())
