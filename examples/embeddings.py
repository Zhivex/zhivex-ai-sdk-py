import asyncio

from zhivex_ai import create_openai, embed_many


async def main() -> None:
    provider = create_openai()
    result = await embed_many(
        model=provider.embedding_model("text-embedding-3-small"),
        values=[
            "Zhivex AI SDK normalizes providers.",
            "Embeddings help semantic search.",
        ],
    )

    print("vectors:", len(result.embeddings))
    print("dimension:", len(result.embeddings[0]) if result.embeddings else 0)


if __name__ == "__main__":
    asyncio.run(main())
