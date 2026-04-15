import asyncio

from zhivex_ai import create_openai, stream_text


async def main() -> None:
    provider = create_openai()
    result = stream_text(
        model=provider("gpt-5.4-nano"),
        prompt="Reply in two short sentences about SDK portability.",
    )

    async for chunk in result.text_stream():
        print(chunk, end="")

    final = await result.collect()
    print("\nfinish:", final.finish_reason)


if __name__ == "__main__":
    asyncio.run(main())
