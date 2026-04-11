import asyncio
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

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
