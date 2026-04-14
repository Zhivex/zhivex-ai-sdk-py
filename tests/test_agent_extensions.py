from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, skipUnless

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (  # noqa: E402
    Agent,
    AgentMemoryState,
    MCPServerConfig,
    StreamTextDeltaEvent,
    ToolExecutionContext,
    create_agent_session,
    create_mcp_tool_registry,
    create_sqlite_agent_memory_store,
    create_sqlite_checkpoint_store,
    discover_mcp_tools,
    mcp_http_server,
    mcp_stdio_server,
    remote_tool,
    resume_agent,
    run_agent,
    stream_agent,
    tool,
)
from zhivex_ai.errors import ValidationError  # noqa: E402
from zhivex_ai.agent import HTTPRemoteToolRuntime, MCPToolRuntime  # noqa: E402
from zhivex_ai.messages import create_text_message  # noqa: E402
from zhivex_ai.types import (  # noqa: E402
    GenerateResult,
    ModelCapabilities,
    ModelGenerateInput,
    ModelMessage,
    StreamFinishEvent,
    StreamToolCallEvent,
    TokenUsage,
    ToolCall,
    ToolCallPart,
)


BASE_CAPABILITIES = ModelCapabilities(
    streaming=True,
    tools=True,
    structured_output=True,
    json_mode=True,
    tool_choice=True,
    parallel_tool_calls=False,
    vision=False,
    files=False,
    audio_input=False,
    audio_output=False,
    embeddings=False,
    reasoning=False,
    web_search=False,
)


class IncrementalAgentModel:
    provider = "test"
    model_id = "incremental"
    capabilities = BASE_CAPABILITIES

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        return GenerateResult(messages=[create_text_message("assistant", "alpha beta")], text="alpha beta")

    async def stream(self, input: ModelGenerateInput):
        async def generator():
            yield StreamTextDeltaEvent(text_delta="alpha")
            yield StreamTextDeltaEvent(text_delta=" beta")
            yield StreamFinishEvent(
                finish_reason="stop",
                usage=TokenUsage(input_tokens=2, output_tokens=2, total_tokens=4),
            )

        return generator()


class StreamingToolAgentModel:
    provider = "test"
    model_id = "streaming-tool"
    capabilities = BASE_CAPABILITIES

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        self.calls += 1
        if self.calls == 1 and input.tools:
            return GenerateResult(
                messages=[
                    ModelMessage(
                        role="assistant",
                        parts=[ToolCallPart(tool_call=ToolCall(id="call_1", name="lookup", input={"item": "apollo"}))],
                    )
                ]
            )
        return GenerateResult(messages=[create_text_message("assistant", "done")], text="done")

    async def stream(self, input: ModelGenerateInput):
        self.calls += 1

        async def generator():
            if self.calls == 1 and input.tools:
                yield StreamToolCallEvent(tool_call=ToolCall(id="call_1", name="lookup", input={"item": "apollo"}))
                yield StreamFinishEvent(
                    finish_reason="tool-calls",
                    usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                )
                return
            yield StreamTextDeltaEvent(text_delta="done")
            yield StreamFinishEvent(
                finish_reason="stop",
                usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            )

        return generator()


class ToolLoopModel:
    provider = "test"
    model_id = "tool-loop"
    capabilities = BASE_CAPABILITIES

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        has_tool_message = any(message.role == "tool" for message in input.messages)
        if not has_tool_message and input.tools:
            return GenerateResult(
                messages=[
                    ModelMessage(
                        role="assistant",
                        parts=[ToolCallPart(tool_call=ToolCall(id="call_1", name="lookup", input={"item": "apollo"}))],
                    )
                ]
            )
        return GenerateResult(messages=[create_text_message("assistant", "done")], text="done")

    async def stream(self, input: ModelGenerateInput):
        async def generator():
            yield StreamTextDeltaEvent(text_delta="done")
            yield StreamFinishEvent(
                finish_reason="stop",
                usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            )

        return generator()


class MemoryAwareModel:
    provider = "test"
    model_id = "memory-aware"
    capabilities = BASE_CAPABILITIES

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        last_user = next((message for message in reversed(input.messages) if message.role == "user"), None)
        prompt = "".join(part.text for part in (last_user.parts if last_user else []) if part.type == "text")
        if "What project" in prompt:
            for message in input.messages:
                if message.role == "user":
                    text = "".join(part.text for part in message.parts if part.type == "text")
                    if "Remember that" in text:
                        project = text.removeprefix("Remember that ").removesuffix(" is important.")
                        return GenerateResult(
                            messages=[create_text_message("assistant", project)],
                            text=project,
                        )
        return GenerateResult(messages=[create_text_message("assistant", f"echo:{prompt}")], text=f"echo:{prompt}")

    async def stream(self, input: ModelGenerateInput):
        async def generator():
            yield StreamTextDeltaEvent(text_delta="unused")
            yield StreamFinishEvent(finish_reason="stop")

        return generator()


class _JSONResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self.payload = payload
        self.headers = {"content-type": "application/json"}

    async def json(self):
        return self.payload

    async def text(self) -> str:
        return str(self.payload)

    async def read(self) -> bytes:
        return str(self.payload).encode("utf-8")

    async def iter_lines(self):
        if False:
            yield ""


class AgentExtensionsTests(IsolatedAsyncioTestCase):
    async def test_stream_agent_emits_incremental_deltas_before_collect(self) -> None:
        stream = stream_agent(agent=Agent(name="assistant", instructions="Stream.", model=IncrementalAgentModel()), prompt="hello")
        chunks = []
        async for event in stream.event_stream():
            if event.type == "text-delta":
                chunks.append(event.text_delta)

        final = await stream.collect()
        self.assertEqual(chunks, ["alpha", " beta"])
        self.assertEqual(final.text, "alpha beta")

    async def test_stream_agent_preserves_tool_event_order(self) -> None:
        stream = stream_agent(
            agent=Agent(
                name="assistant",
                instructions="Use tools.",
                model=StreamingToolAgentModel(),
                tools={
                    "lookup": tool(
                        name="lookup",
                        schema=dict[str, str],
                        execute=lambda input: {"item": input["item"], "status": "ok"},
                        permissions=["project:read"],
                        requires_approval=True,
                    )
                },
            ),
            prompt="plan",
        )
        event_types = [event.type async for event in stream.event_stream()]
        await stream.collect()
        self.assertLess(event_types.index("tool-call"), event_types.index("tool-approval"))
        self.assertLess(event_types.index("tool-approval"), event_types.index("tool-result"))
        self.assertLess(event_types.index("tool-result"), event_types.index("text-delta"))

    async def test_sqlite_stores_persist_across_instances_and_resume_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "agent.sqlite3")
            memory_store = create_sqlite_agent_memory_store(path)
            checkpoint_store = create_sqlite_checkpoint_store(path)
            agent = Agent(
                name="assistant",
                instructions="Remember things.",
                model=MemoryAwareModel(),
                memory=memory_store,
                checkpoint_store=checkpoint_store,
            )
            session = create_agent_session()
            await run_agent(agent=agent, session=session, prompt="Remember that project Apollo is important.")

            restored_memory = create_sqlite_agent_memory_store(path)
            restored_state = await restored_memory.load(session.id)
            self.assertTrue(restored_state.messages)

            restored_checkpoints = create_sqlite_checkpoint_store(path)
            latest = await restored_checkpoints.get_latest(session_id=session.id)
            self.assertIsNotNone(latest)

            resumed_agent = Agent(
                name="assistant",
                instructions="Remember things.",
                model=MemoryAwareModel(),
                memory=restored_memory,
                checkpoint_store=restored_checkpoints,
            )
            resumed = await resume_agent(
                agent=resumed_agent,
                session_id=session.id,
                prompt="What project did I mention?",
            )
            self.assertEqual(resumed.text, "project Apollo")
            self.assertIsNotNone(resumed.resumed_from_checkpoint)

    async def test_http_remote_runtime_posts_json_contract_and_context(self) -> None:
        observed: dict[str, object] = {}

        async def fake_fetch(url: str, *, method: str = "POST", headers: dict[str, str], json_body=None, body=None, timeout_ms=None, stream=False):
            observed["url"] = url
            observed["method"] = method
            observed["headers"] = headers
            observed["json_body"] = json_body
            observed["timeout_ms"] = timeout_ms
            return _JSONResponse(200, {"output": {"ok": True}})

        runtime = HTTPRemoteToolRuntime(fetch=fake_fetch)
        definition = remote_tool(
            name="lookup",
            url="https://example.com/tools/lookup",
            schema=dict[str, str],
            headers={"x-api-key": "secret"},
            timeout_ms=321,
        )
        result = await runtime.execute(
            definition,
            {"item": "apollo"},
            ToolExecutionContext(tool_name="lookup", tool_call_id="call_1", run_id="run_1"),
        )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(observed["url"], "https://example.com/tools/lookup")
        self.assertEqual(observed["timeout_ms"], 321)
        self.assertEqual(observed["json_body"]["tool"], "lookup")
        self.assertEqual(observed["json_body"]["context"]["tool_call_id"], "call_1")

    async def test_http_remote_runtime_raises_on_invalid_payload(self) -> None:
        async def fake_fetch(url: str, *, method: str = "POST", headers: dict[str, str], json_body=None, body=None, timeout_ms=None, stream=False):
            return _JSONResponse(200, {"unexpected": True})

        runtime = HTTPRemoteToolRuntime(fetch=fake_fetch)
        definition = remote_tool(name="lookup", url="https://example.com/tools/lookup", schema=dict[str, str])
        with self.assertRaises(RuntimeError):
            await runtime.execute(definition, {"item": "apollo"}, ToolExecutionContext(tool_name="lookup"))

    async def test_discover_mcp_tools_and_runtime_execution(self) -> None:
        stdio_calls: dict[str, int] = {"closed": 0}

        class FakeTool:
            def __init__(self, name: str, description: str) -> None:
                self.name = name
                self.description = description
                self.inputSchema = {"type": "object", "properties": {"item": {"type": "string"}}}

        class FakeResult:
            def __init__(self, tools=None, structured=None) -> None:
                self.tools = tools or []
                self.structuredContent = structured

        class FakeClientSession:
            def __init__(self, read_stream, write_stream) -> None:
                self.read_stream = read_stream
                self.write_stream = write_stream

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def initialize(self):
                return None

            async def list_tools(self):
                return FakeResult(tools=[FakeTool("lookup", "Lookup an item")])

            async def call_tool(self, name: str, arguments: dict[str, str]):
                return FakeResult(structured={"tool": name, "arguments": arguments})

        class FakeTransportContext:
            async def __aenter__(self):
                return ("read", "write")

            async def __aexit__(self, exc_type, exc, tb):
                stdio_calls["closed"] += 1
                return None

        class FakeParams:
            def __init__(self, command: str, args: list[str], env=None) -> None:
                self.command = command
                self.args = args
                self.env = env

        fake_mcp = types.ModuleType("mcp")
        fake_mcp.ClientSession = FakeClientSession
        fake_mcp_client = types.ModuleType("mcp.client")
        fake_stdio = types.ModuleType("mcp.client.stdio")
        fake_stdio.StdioServerParameters = FakeParams
        fake_stdio.stdio_client = lambda params: FakeTransportContext()
        fake_http = types.ModuleType("mcp.client.streamable_http")
        fake_http.streamable_http_client = lambda url, headers=None, timeout=None: FakeTransportContext()

        previous_modules = {
            name: sys.modules.get(name)
            for name in ("mcp", "mcp.client", "mcp.client.stdio", "mcp.client.streamable_http")
        }
        sys.modules["mcp"] = fake_mcp
        sys.modules["mcp.client"] = fake_mcp_client
        sys.modules["mcp.client.stdio"] = fake_stdio
        sys.modules["mcp.client.streamable_http"] = fake_http
        try:
            server = MCPServerConfig(transport="stdio", name="demo", command="demo-server")
            tools = await discover_mcp_tools(server, prefix="mcp_")
            self.assertIn("mcp_lookup", tools)
            self.assertEqual(tools["mcp_lookup"].metadata["mcp_server"], "demo")
            self.assertEqual(tools["mcp_lookup"].metadata["mcp_tool_name"], "lookup")
            runtime = MCPToolRuntime()
            result = await runtime.execute(
                tools["mcp_lookup"],
                {"item": "apollo"},
                ToolExecutionContext(tool_name="mcp_lookup"),
            )
            self.assertEqual(result["tool"], "lookup")
            self.assertEqual(result["arguments"]["item"], "apollo")
            await runtime.aclose()
            self.assertGreaterEqual(stdio_calls["closed"], 1)
        finally:
            for name, module in previous_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    async def test_create_mcp_tool_registry_uses_server_name_prefix_and_sanitizes_names(self) -> None:
        class FakeTool:
            def __init__(self, name: str, description: str) -> None:
                self.name = name
                self.description = description
                self.inputSchema = {"type": "object"}

        class FakeResult:
            def __init__(self, tools=None, structured=None) -> None:
                self.tools = tools or []
                self.structuredContent = structured

        class FakeClientSession:
            def __init__(self, read_stream, write_stream) -> None:
                return None

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def initialize(self):
                return None

            async def list_tools(self):
                return FakeResult(tools=[FakeTool("lookup-value", "Lookup"), FakeTool("read.file", "Read file")])

            async def call_tool(self, name: str, arguments: dict[str, str]):
                return FakeResult(structured={"tool": name, "arguments": arguments})

        class FakeTransportContext:
            async def __aenter__(self):
                return ("read", "write")

            async def __aexit__(self, exc_type, exc, tb):
                return None

        class FakeParams:
            def __init__(self, command: str, args: list[str], env=None) -> None:
                self.command = command
                self.args = args
                self.env = env

        fake_mcp = types.ModuleType("mcp")
        fake_mcp.ClientSession = FakeClientSession
        fake_mcp_client = types.ModuleType("mcp.client")
        fake_stdio = types.ModuleType("mcp.client.stdio")
        fake_stdio.StdioServerParameters = FakeParams
        fake_stdio.stdio_client = lambda params: FakeTransportContext()
        fake_http = types.ModuleType("mcp.client.streamable_http")
        fake_http.streamable_http_client = lambda url, headers=None, timeout=None: FakeTransportContext()

        previous_modules = {
            name: sys.modules.get(name)
            for name in ("mcp", "mcp.client", "mcp.client.stdio", "mcp.client.streamable_http")
        }
        sys.modules["mcp"] = fake_mcp
        sys.modules["mcp.client"] = fake_mcp_client
        sys.modules["mcp.client.stdio"] = fake_stdio
        sys.modules["mcp.client.streamable_http"] = fake_http
        try:
            registry = await create_mcp_tool_registry(
                mcp_stdio_server(name="file-system", command="demo-server"),
            )
            lookup = registry.get("file_system_lookup_value")
            reader = registry.get("file_system_read_file")
            self.assertIsNotNone(lookup)
            self.assertIsNotNone(reader)
            result = await registry.execute(
                lookup,
                {"item": "apollo"},
                ToolExecutionContext(tool_name="file_system_lookup_value"),
            )
            self.assertEqual(result["tool"], "lookup-value")
            self.assertEqual(result["arguments"]["item"], "apollo")
            await registry.aclose()
        finally:
            for name, module in previous_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    async def test_create_mcp_tool_registry_raises_on_name_collisions(self) -> None:
        class FakeTool:
            def __init__(self, name: str) -> None:
                self.name = name
                self.description = name
                self.inputSchema = {"type": "object"}

        class FakeResult:
            def __init__(self, tools=None) -> None:
                self.tools = tools or []

        class FakeClientSession:
            def __init__(self, read_stream, write_stream) -> None:
                return None

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def initialize(self):
                return None

            async def list_tools(self):
                return FakeResult(tools=[FakeTool("read-file"), FakeTool("read.file")])

        class FakeTransportContext:
            async def __aenter__(self):
                return ("read", "write")

            async def __aexit__(self, exc_type, exc, tb):
                return None

        class FakeParams:
            def __init__(self, command: str, args: list[str], env=None) -> None:
                self.command = command
                self.args = args
                self.env = env

        fake_mcp = types.ModuleType("mcp")
        fake_mcp.ClientSession = FakeClientSession
        fake_mcp_client = types.ModuleType("mcp.client")
        fake_stdio = types.ModuleType("mcp.client.stdio")
        fake_stdio.StdioServerParameters = FakeParams
        fake_stdio.stdio_client = lambda params: FakeTransportContext()
        fake_http = types.ModuleType("mcp.client.streamable_http")
        fake_http.streamable_http_client = lambda url, headers=None, timeout=None: FakeTransportContext()

        previous_modules = {
            name: sys.modules.get(name)
            for name in ("mcp", "mcp.client", "mcp.client.stdio", "mcp.client.streamable_http")
        }
        sys.modules["mcp"] = fake_mcp
        sys.modules["mcp.client"] = fake_mcp_client
        sys.modules["mcp.client.stdio"] = fake_stdio
        sys.modules["mcp.client.streamable_http"] = fake_http
        try:
            with self.assertRaises(ValidationError):
                await create_mcp_tool_registry(
                    mcp_stdio_server(name="filesystem", command="demo-server"),
                )
        finally:
            for name, module in previous_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_mcp_server_helpers_build_expected_configs(self) -> None:
        stdio = mcp_stdio_server(name="filesystem", command="npx", args=["-y", "server"], env={"ROOT": "."}, timeout_ms=500)
        http = mcp_http_server(name="search", url="https://example.com/mcp", headers={"authorization": "Bearer token"}, timeout_ms=700)
        self.assertEqual(stdio.transport, "stdio")
        self.assertEqual(stdio.command, "npx")
        self.assertEqual(stdio.args, ["-y", "server"])
        self.assertEqual(stdio.env["ROOT"], ".")
        self.assertEqual(stdio.timeout_ms, 500)
        self.assertEqual(http.transport, "streamable-http")
        self.assertEqual(http.url, "https://example.com/mcp")
        self.assertEqual(http.headers["authorization"], "Bearer token")
        self.assertEqual(http.timeout_ms, 700)


class FakeAsyncPGConnection:
    store: dict[str, dict[str, object]] = {"memory": {}, "checkpoints": []}

    async def execute(self, sql: str, *args):
        lowered = " ".join(sql.split()).lower()
        if "insert into zhivex_ai_agent_memory" in lowered:
            self.store["memory"][args[0]] = {"state_json": args[1], "updated_at_ms": args[2]}
        elif "insert into zhivex_ai_agent_checkpoints" in lowered:
            self.store["checkpoints"].append(
                {
                    "run_id": args[0],
                    "session_id": args[1],
                    "agent_name": args[2],
                    "step_index": args[3],
                    "saved_at_ms": args[4],
                    "is_final": args[5],
                    "checkpoint_json": args[6],
                }
            )
        return "OK"

    async def fetchrow(self, sql: str, *args):
        lowered = " ".join(sql.split()).lower()
        if "from zhivex_ai_agent_memory" in lowered:
            state = self.store["memory"].get(args[0])
            return {"state_json": state["state_json"]} if state else None
        items = list(self.store["checkpoints"])
        if "where session_id = $1" in lowered:
            items = [item for item in items if item["session_id"] == args[0]]
        if "where run_id = $1" in lowered:
            items = [item for item in items if item["run_id"] == args[0]]
        if not items:
            return None
        items.sort(key=lambda item: (item["saved_at_ms"], item["step_index"]), reverse=True)
        return {"checkpoint_json": items[0]["checkpoint_json"]}

    async def fetch(self, sql: str, *args):
        items = list(self.store["checkpoints"])
        lowered = " ".join(sql.split()).lower()
        if "where session_id = $1 and run_id = $2" in lowered:
            items = [item for item in items if item["session_id"] == args[0] and item["run_id"] == args[1]]
        elif "where session_id = $1" in lowered:
            items = [item for item in items if item["session_id"] == args[0]]
        elif "where run_id = $1" in lowered:
            items = [item for item in items if item["run_id"] == args[0]]
        items.sort(key=lambda item: (item["saved_at_ms"], item["step_index"]))
        return [{"checkpoint_json": item["checkpoint_json"]} for item in items]

    async def close(self):
        return None


class PostgresStoreTests(IsolatedAsyncioTestCase):
    async def test_postgres_stores_work_with_asyncpg_driver(self) -> None:
        FakeAsyncPGConnection.store = {"memory": {}, "checkpoints": []}

        async def connect(dsn: str):
            return FakeAsyncPGConnection()

        fake_asyncpg = types.SimpleNamespace(connect=connect)
        previous = sys.modules.get("asyncpg")
        sys.modules["asyncpg"] = fake_asyncpg
        try:
            from zhivex_ai import create_postgres_agent_memory_store, create_postgres_checkpoint_store

            memory = create_postgres_agent_memory_store("postgres://example")
            checkpoint_store = create_postgres_checkpoint_store("postgres://example")
            agent = Agent(
                name="assistant",
                instructions="Use tools.",
                model=ToolLoopModel(),
                memory=memory,
                checkpoint_store=checkpoint_store,
                tools={
                    "lookup": tool(
                        name="lookup",
                        schema=dict[str, str],
                        execute=lambda input: {"item": input["item"], "status": "ok"},
                    )
                },
            )
            session = create_agent_session()
            await run_agent(agent=agent, session=session, prompt="plan", max_steps=2)
            restored = await memory.load(session.id)
            latest = await checkpoint_store.get_latest(session_id=session.id)
            self.assertTrue(restored.messages)
            self.assertIsNotNone(latest)
        finally:
            if previous is None:
                sys.modules.pop("asyncpg", None)
            else:
                sys.modules["asyncpg"] = previous


@skipUnless(os.getenv("ZHIVEX_TEST_POSTGRES_DSN"), "requires ZHIVEX_TEST_POSTGRES_DSN")
class PostgresIntegrationTests(IsolatedAsyncioTestCase):
    async def test_postgres_memory_store_roundtrip(self) -> None:
        from zhivex_ai import create_postgres_agent_memory_store

        store = create_postgres_agent_memory_store(os.environ["ZHIVEX_TEST_POSTGRES_DSN"])
        session_id = "integration-session"
        await store.save(
            session_id,
            AgentMemoryState(messages=[create_text_message("user", "hello")], summary="hello", metadata={"test": True}),
        )
        restored = await store.load(session_id)
        self.assertEqual(restored.summary, "hello")
