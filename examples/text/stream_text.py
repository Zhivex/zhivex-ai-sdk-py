import asyncio

from zhivex_ai import HTTPTransport, create_openai, stream_text


async def main() -> None:
    async with HTTPTransport() as transport:
        provider = create_openai(fetch=transport)
        async with stream_text(
            model=provider("gpt-6-astra"),
            prompt="Reply in two short sentences about SDK portability.",
            stream_buffer_size=4096,
        ) as result:
            async for chunk in result.text_stream():
                print(chunk, end="")
            final = await result.collect()
            print("\nfinish:", final.finish_reason)


if __name__ == "__main__":
    asyncio.run(main())
