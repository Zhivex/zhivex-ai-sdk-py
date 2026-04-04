import asyncio

from zhivex_ai import (
    create_openai,
    create_telemetry_middleware,
    generate_text,
    wrap_language_model,
)


async def main() -> None:
    provider = create_openai()
    model = wrap_language_model(
        provider("gpt-4o-mini"),
        [
            create_telemetry_middleware(
                on_event=lambda event: print("telemetry:", event["type"])
            )
        ],
    )

    result = await generate_text(
        model=model,
        prompt="Explain middleware in one sentence.",
    )
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
