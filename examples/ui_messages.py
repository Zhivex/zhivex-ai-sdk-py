import asyncio

from zhivex_ai import (
    create_text_message,
    create_ui_message_json_response,
    create_ui_message_lines_response,
    from_ui_messages,
    serialize_ui_message,
    to_ui_messages,
)


async def main() -> None:
    source_messages = [
        create_text_message("user", "Hello"),
        create_text_message("assistant", "Hi there"),
    ]

    ui_messages = to_ui_messages(source_messages)
    print(serialize_ui_message(ui_messages[0]))

    model_messages = from_ui_messages(ui_messages)
    print(model_messages[1].parts[0].text)

    json_response = create_ui_message_json_response(ui_messages)
    lines_response = create_ui_message_lines_response(ui_messages)
    print(json_response.headers["content-type"])
    print(lines_response.headers["content-type"])


if __name__ == "__main__":
    asyncio.run(main())
