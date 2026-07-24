import asyncio

from zhivex_ai import create_openai, stream_text, to_text_stream_response, to_ui_message_stream_response


async def main() -> None:
    provider = create_openai()

    text_result = stream_text(
        model=provider("gpt-5.6-terra"),
        prompt="Stream a short answer.",
    )
    text_response = to_text_stream_response(text_result)
    print(text_response.headers["content-type"])

    ui_result = stream_text(
        model=provider("gpt-5.6-terra"),
        prompt="Stream another short answer.",
    )
    ui_response = to_ui_message_stream_response(ui_result)
    print(ui_response.headers["content-type"])


if __name__ == "__main__":
    asyncio.run(main())
