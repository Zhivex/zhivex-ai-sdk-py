import asyncio

from zhivex_ai import create_openai, generate_text, tool, user


async def main() -> None:
    provider = create_openai()

    result = await generate_text(
        model=provider("gpt-5.6-terra"),
        messages=[user("What is the weather in Madrid? Use the tool.")],
        max_steps=2,
        tools={
            "weather": tool(
                name="weather",
                description="Returns a tiny weather summary for a city.",
                schema=dict[str, str],
                execute=lambda input: {
                    "city": input["city"],
                    "forecast": "sunny",
                    "temperature_c": "26",
                },
            )
        },
    )

    print(result.text)
    print(result.tool_results)


if __name__ == "__main__":
    asyncio.run(main())
