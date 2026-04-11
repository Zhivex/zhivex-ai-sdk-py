from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeAlias, TypeVar

JsonPrimitive = str | int | float | bool | None
JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
PartialJsonValue = JsonPrimitive | list["PartialJsonValue"] | dict[str, "PartialJsonValue | None"]

MessageRole = Literal["system", "user", "assistant", "tool"]
FinishReason = Literal["stop", "length", "tool-calls", "content-filter", "error", "unknown"]
StructuredOutputMode = Literal["auto", "native", "prompted"]
ToolChoiceMode = Literal["none", "auto", "required"]


@dataclass(slots=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(slots=True)
class AudioInput:
    data: bytes | bytearray | memoryview | str
    media_type: str
    filename: str | None = None


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    input: JsonValue
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolExecutionError:
    message: str


@dataclass(slots=True)
class ToolExecutionResult:
    tool_call_id: str
    tool_name: str
    output: JsonValue | None = None
    error: ToolExecutionError | None = None
    is_error: bool = False


@dataclass(slots=True)
class TextPart:
    type: Literal["text"] = "text"
    text: str = ""


@dataclass(slots=True)
class ImagePart:
    type: Literal["image"] = "image"
    image: str = ""
    media_type: str | None = None


@dataclass(slots=True)
class FilePart:
    type: Literal["file"] = "file"
    data: str = ""
    media_type: str = ""
    filename: str | None = None


@dataclass(slots=True)
class ToolCallPart:
    type: Literal["tool-call"] = "tool-call"
    tool_call: ToolCall = field(default_factory=lambda: ToolCall(id="", name="", input={}))


@dataclass(slots=True)
class ToolResultPart:
    type: Literal["tool-result"] = "tool-result"
    tool_result: ToolExecutionResult = field(
        default_factory=lambda: ToolExecutionResult(tool_call_id="", tool_name="", is_error=False)
    )


ContentPart: TypeAlias = TextPart | ImagePart | FilePart | ToolCallPart | ToolResultPart


@dataclass(slots=True)
class ModelMessage:
    role: MessageRole
    parts: list[ContentPart]


@dataclass(slots=True)
class UIMessage:
    id: str
    role: MessageRole
    parts: list[ContentPart]


@dataclass(slots=True)
class UIMessageTextChunk:
    type: Literal["text-delta"] = "text-delta"
    message_id: str = ""
    role: Literal["assistant"] = "assistant"
    text_delta: str = ""


@dataclass(slots=True)
class UIMessageToolCallChunk:
    type: Literal["tool-call"] = "tool-call"
    message_id: str = ""
    role: Literal["assistant"] = "assistant"
    tool_call: ToolCall = field(default_factory=lambda: ToolCall(id="", name="", input={}))


@dataclass(slots=True)
class UIMessageToolResultChunk:
    type: Literal["tool-result"] = "tool-result"
    message_id: str = ""
    role: Literal["tool"] = "tool"
    tool_result: ToolExecutionResult = field(
        default_factory=lambda: ToolExecutionResult(tool_call_id="", tool_name="", is_error=False)
    )


@dataclass(slots=True)
class UIMessageFinishChunk:
    type: Literal["finish"] = "finish"
    message_id: str = ""
    finish_reason: FinishReason | None = None
    provider_finish_reason: str | None = None
    usage: TokenUsage | None = None


@dataclass(slots=True)
class UIMessageErrorChunk:
    type: Literal["error"] = "error"
    message_id: str = ""
    error: ToolExecutionError = field(default_factory=lambda: ToolExecutionError(message=""))


UIMessageChunk: TypeAlias = (
    UIMessageTextChunk
    | UIMessageToolCallChunk
    | UIMessageToolResultChunk
    | UIMessageFinishChunk
    | UIMessageErrorChunk
)


@dataclass(slots=True)
class ModelCapabilities:
    streaming: bool
    tools: bool
    structured_output: bool
    json_mode: bool
    tool_choice: bool
    parallel_tool_calls: bool
    vision: bool
    files: bool
    audio_input: bool
    audio_output: bool
    embeddings: bool
    reasoning: bool
    web_search: bool


@dataclass(slots=True)
class RetryOptions:
    timeout_ms: int | None = None
    max_retries: int | None = None
    retry_backoff_ms: int | None = None


@dataclass(slots=True)
class StructuredOutputConfig:
    schema: Any
    mode: Literal["native", "prompted"]
    name: str | None = None
    description: str | None = None


@dataclass(slots=True)
class ReasoningConfig:
    effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    budget_tokens: int | None = None


@dataclass(slots=True)
class ToolChoiceName:
    tool_name: str


ToolChoice: TypeAlias = ToolChoiceMode | ToolChoiceName
ToolSource: TypeAlias = Literal["local", "remote", "mcp"]


@dataclass(slots=True)
class RemoteHTTPToolConfig:
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    timeout_ms: int | None = None


@dataclass(slots=True)
class MCPServerConfig:
    transport: Literal["stdio", "streamable-http"]
    name: str = "default"
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout_ms: int | None = None


@dataclass(slots=True)
class MCPToolConfig:
    server: MCPServerConfig
    tool_name: str


@dataclass(slots=True)
class ToolExecutionOptions:
    parallel: bool | None = None
    max_concurrency: int | None = None
    timeout_ms: int | None = None
    stop_on_error: bool = False


@dataclass(slots=True)
class ToolExecutionContext:
    tool_name: str
    tool_call_id: str = ""
    run_id: str | None = None
    session_id: str | None = None
    agent_name: str | None = None
    memory_summary: str | None = None
    permissions: list[str] = field(default_factory=list)
    source: ToolSource = "local"
    metadata: dict[str, Any] = field(default_factory=dict)
    handoff_path: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StreamTextDeltaEvent:
    type: Literal["text-delta"] = "text-delta"
    text_delta: str = ""


@dataclass(slots=True)
class StreamToolCallEvent:
    type: Literal["tool-call"] = "tool-call"
    tool_call: ToolCall = field(default_factory=lambda: ToolCall(id="", name="", input={}))


@dataclass(slots=True)
class StreamToolResultEvent:
    type: Literal["tool-result"] = "tool-result"
    tool_result: ToolExecutionResult = field(
        default_factory=lambda: ToolExecutionResult(tool_call_id="", tool_name="", is_error=False)
    )


@dataclass(slots=True)
class StreamFinishEvent:
    type: Literal["finish"] = "finish"
    finish_reason: FinishReason | None = None
    provider_finish_reason: str | None = None
    usage: TokenUsage | None = None


@dataclass(slots=True)
class StreamErrorEvent:
    type: Literal["error"] = "error"
    error: Exception | None = None


StreamEvent: TypeAlias = (
    StreamTextDeltaEvent | StreamToolCallEvent | StreamToolResultEvent | StreamFinishEvent | StreamErrorEvent
)


@dataclass(slots=True)
class StreamObjectDeltaEvent:
    type: Literal["object-delta"] = "object-delta"
    text_delta: str = ""
    partial_text: str = ""


@dataclass(slots=True)
class StreamObjectPartialEvent:
    type: Literal["object-partial"] = "object-partial"
    partial_object: Any = None


@dataclass(slots=True)
class StreamObjectCompleteEvent:
    type: Literal["object-complete"] = "object-complete"
    object: Any = None


ObjectStreamEvent: TypeAlias = (
    StreamEvent | StreamObjectDeltaEvent | StreamObjectPartialEvent | StreamObjectCompleteEvent
)


@dataclass(slots=True)
class GenerateResult:
    message: ModelMessage | None = None
    messages: list[ModelMessage] | None = None
    text: str | None = None
    finish_reason: FinishReason | None = None
    provider_finish_reason: str | None = None
    usage: TokenUsage | None = None
    raw_response: Any = None


@dataclass(slots=True)
class GroundingSource:
    url: str
    title: str | None = None
    snippet: str | None = None
    provider_metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class GroundedGenerateResult(GenerateResult):
    sources: list[GroundingSource] = field(default_factory=list)


@dataclass(slots=True)
class EmbedResult:
    embeddings: list[list[float]]
    usage: TokenUsage | None = None
    raw_response: Any = None


@dataclass(slots=True)
class TranscriptionOutput:
    text: str
    audio: AudioInput | None = None
    raw_response: Any = None


@dataclass(slots=True)
class SpeechOutput:
    audio: bytes
    media_type: str
    input: str | None = None
    raw_response: Any = None


ProviderOptions = dict[str, Any]


@dataclass(slots=True)
class ModelGenerateInput(RetryOptions):
    messages: list[ModelMessage] = field(default_factory=list)
    tools: dict[str, "ToolDefinition[Any]"] | None = None
    tool_choice: ToolChoice | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning: ReasoningConfig | None = None
    provider_options: ProviderOptions | None = None
    structured_output: StructuredOutputConfig | None = None


@dataclass(slots=True)
class GroundedModelGenerateInput(RetryOptions):
    messages: list[ModelMessage] = field(default_factory=list)
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning: ReasoningConfig | None = None
    provider_options: ProviderOptions | None = None


class LanguageModel(Protocol):
    provider: str
    model_id: str
    capabilities: ModelCapabilities

    async def generate(self, input: ModelGenerateInput) -> GenerateResult: ...

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[StreamEvent]: ...


class GroundedLanguageModel(Protocol):
    provider: str
    model_id: str
    capabilities: ModelCapabilities

    async def generate(self, input: GroundedModelGenerateInput) -> GroundedGenerateResult: ...


class EmbeddingModel(Protocol):
    provider: str
    model_id: str
    capabilities: ModelCapabilities

    async def embed(self, values: list[str], options: RetryOptions | None = None) -> EmbedResult: ...


class TranscriptionModel(Protocol):
    provider: str
    model_id: str
    capabilities: ModelCapabilities

    async def transcribe(
        self,
        *,
        audio: AudioInput,
        prompt: str | None = None,
        language: str | None = None,
        provider_options: ProviderOptions | None = None,
        options: RetryOptions | None = None,
    ) -> TranscriptionOutput: ...


class SpeechModel(Protocol):
    provider: str
    model_id: str
    capabilities: ModelCapabilities

    async def generate_speech(
        self,
        *,
        input: str,
        voice: str | None = None,
        provider_options: ProviderOptions | None = None,
        options: RetryOptions | None = None,
    ) -> SpeechOutput: ...


TSchema = TypeVar("TSchema")


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str | None
    schema: Any
    execute: Callable[..., Awaitable[JsonValue] | JsonValue] | None = None
    tags: list[str] = field(default_factory=list)
    requires_approval: bool | None = None
    permissions: list[str] = field(default_factory=list)
    source: ToolSource = "local"
    metadata: dict[str, Any] = field(default_factory=dict)
    supports_streaming: bool = False
    remote_config: RemoteHTTPToolConfig | None = None
    mcp_config: MCPToolConfig | None = None


ToolSet = dict[str, ToolDefinition]


@dataclass(slots=True)
class GenerateTextStep:
    request: ModelGenerateInput
    response: GenerateResult


@dataclass(slots=True)
class GenerateTextOutput:
    text: str
    finish_reason: FinishReason | None = None
    provider_finish_reason: str | None = None
    usage: TokenUsage | None = None
    steps: list[GenerateTextStep] = field(default_factory=list)
    messages: list[ModelMessage] = field(default_factory=list)
    tool_results: list[ToolExecutionResult] = field(default_factory=list)


@dataclass(slots=True)
class GenerateObjectOutput(GenerateTextOutput):
    object: Any = None
    object_mode: Literal["native", "prompted"] = "prompted"


@dataclass(slots=True)
class GenerateGroundedTextOutput:
    text: str
    sources: list[GroundingSource] = field(default_factory=list)
    finish_reason: FinishReason | None = None
    provider_finish_reason: str | None = None
    usage: TokenUsage | None = None
    messages: list[ModelMessage] = field(default_factory=list)
    raw_response: Any = None


class StreamTextResult(Protocol):
    def event_stream(self) -> AsyncIterable[StreamEvent]: ...

    def text_stream(self) -> AsyncIterable[str]: ...

    async def collect(self) -> GenerateTextOutput: ...


class StreamObjectResult(Protocol):
    def event_stream(self) -> AsyncIterable[ObjectStreamEvent]: ...

    def text_stream(self) -> AsyncIterable[str]: ...

    def partial_object_stream(self) -> AsyncIterable[Any]: ...

    async def collect(self) -> GenerateObjectOutput: ...


@dataclass(slots=True)
class EmbedOutput(EmbedResult):
    values: list[str] = field(default_factory=list)
