# Examples

This folder contains runnable Python examples for the main public surfaces of the Zhivex AI SDK.

## Run

From the repository root:

```bash
.venv/bin/python examples/openai_text.py
```

Most examples require provider credentials in environment variables. The files show which provider setup they expect.

Useful starting points:

```bash
.venv/bin/python examples/openai_text.py
.venv/bin/python examples/stream_text.py
.venv/bin/python examples/stream_object.py
.venv/bin/python examples/messages_and_tools.py
.venv/bin/python examples/embeddings.py
.venv/bin/python examples/grounded_text.py
.venv/bin/python examples/transcribe_audio.py
.venv/bin/python examples/generate_speech.py
.venv/bin/python examples/ui_messages.py
.venv/bin/python examples/http_responses.py
.venv/bin/python examples/gateway_fallback.py
```

## Notes

- OpenAI and Azure OpenAI currently expose the richest Python feature surface for audio and grounded text.
- Some providers do not support every capability. The examples follow the actual adapter capabilities in this repo.
- Structured output examples use `pydantic`.
