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
PortableProviderTier = Literal["portable", "native-only", "compatibility"]


@dataclass(slots=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(slots=True)
class PortableSupport:
    text_generation: bool = False
    streaming: bool = False
    structured_output: bool = False
    tools: bool = False
    embeddings: bool = False
    grounding: bool = False
    retrieval: bool = False
    transcription: bool = False
    speech: bool = False
    portable_badge: bool = False
    tier: PortableProviderTier = "native-only"


@dataclass(slots=True)
class NativeSupport:
    text_generation: bool = False
    streaming: bool = False
    tools: bool = False
    structured_output: bool = False
    embeddings: bool = False
    grounding: bool = False
    transcription: bool = False
    speech: bool = False
    realtime: bool = False
    files: bool = False
    file_search: bool = False
    images: bool = False
    uploads: bool = False
    moderations: bool = False
    batches: bool = False
    responses: bool = False
    conversations: bool = False


@dataclass(slots=True)
class PortableGroundingConfig:
    max_sources: int | None = None


@dataclass(slots=True)
class PortableDocument:
    document_id: str
    text: str
    title: str | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(slots=True)
class PortableRetrievalConfig:
    documents: list[PortableDocument] = field(default_factory=list)
    max_documents: int = 5
    max_document_chars: int = 4_000


@dataclass(slots=True)
class PortableTranscriptionConfig:
    prompt: str | None = None
    language: str | None = None


@dataclass(slots=True)
class PortableSpeechConfig:
    voice: str | None = None
    audio_format: str | None = None


@dataclass(slots=True)
class AudioInput:
    data: bytes | bytearray | memoryview | str
    media_type: str
    filename: str | None = None


@dataclass(slots=True)
class AudioFrame:
    data: bytes | bytearray | memoryview | str
    media_type: str
    sample_rate_hz: int | None = None
    channels: int | None = None
    timestamp_ms: int | None = None
    is_final: bool = False


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
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ImagePart:
    type: Literal["image"] = "image"
    image: str = ""
    media_type: str | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FilePart:
    type: Literal["file"] = "file"
    data: str | None = None
    text: str | None = None
    document_content: list[JsonValue] | None = None
    media_type: str | None = None
    filename: str | None = None
    file_id: str | None = None
    file_uri: str | None = None
    url: str | None = None
    title: str | None = None
    context: str | None = None
    citations_enabled: bool | None = None
    cache_control: dict[str, Any] | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProviderFile:
    provider: str
    id: str
    filename: str | None = None
    media_type: str | None = None
    size_bytes: int | None = None
    status: str | None = None
    url: str | None = None
    file_uri: str | None = None
    created_at: str | int | None = None
    downloadable: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProviderImage:
    provider: str
    b64_json: str | None = None
    url: str | None = None
    revised_prompt: str | None = None
    media_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ImagesResult:
    images: list[ProviderImage] = field(default_factory=list)
    created_at: str | int | None = None
    raw_response: Any = None


@dataclass(slots=True)
class ProviderUploadPart:
    provider: str
    id: str
    upload_id: str | None = None
    created_at: str | int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProviderUpload:
    provider: str
    id: str
    filename: str | None = None
    purpose: str | None = None
    bytes: int | None = None
    status: str | None = None
    mime_type: str | None = None
    created_at: str | int | None = None
    expires_at: str | int | None = None
    completed_at: str | int | None = None
    cancelled_at: str | int | None = None
    file: ProviderFile | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FilesClient(Protocol):
    async def upload(
        self,
        *,
        data: bytes | bytearray | memoryview,
        filename: str,
        media_type: str = "application/pdf",
        purpose: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProviderFile: ...

    async def list(self) -> list[ProviderFile]: ...

    async def get(self, file_id: str) -> ProviderFile: ...

    async def download(self, file_id: str) -> bytes: ...

    async def delete(self, file_id: str) -> bool: ...


class ImagesClient(Protocol):
    async def generate(
        self,
        *,
        prompt: str,
        model: str | None = None,
        size: str | None = None,
        quality: str | None = None,
        background: str | None = None,
        output_format: str | None = None,
        moderation: str | None = None,
        user: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ImagesResult: ...

    async def edit(
        self,
        *,
        prompt: str,
        image: bytes | bytearray | memoryview | list[bytes | bytearray | memoryview],
        image_filenames: str | list[str] | None = None,
        image_media_type: str | list[str] | None = None,
        model: str | None = None,
        mask: bytes | bytearray | memoryview | None = None,
        mask_filename: str | None = None,
        mask_media_type: str | None = None,
        size: str | None = None,
        quality: str | None = None,
        background: str | None = None,
        output_format: str | None = None,
        moderation: str | None = None,
        user: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ImagesResult: ...

    async def variation(
        self,
        *,
        image: bytes | bytearray | memoryview,
        image_filename: str | None = None,
        image_media_type: str | None = None,
        model: str | None = None,
        size: str | None = None,
        quality: str | None = None,
        background: str | None = None,
        output_format: str | None = None,
        user: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ImagesResult: ...


class UploadsClient(Protocol):
    async def create(
        self,
        *,
        filename: str,
        bytes: int,
        mime_type: str,
        purpose: str,
        expires_after: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProviderUpload: ...

    async def add_part(
        self,
        *,
        upload_id: str,
        data: bytes | bytearray | memoryview,
        filename: str | None = None,
        media_type: str | None = None,
    ) -> ProviderUploadPart: ...

    async def complete(
        self,
        upload_id: str,
        *,
        part_ids: list[str],
        md5: str | None = None,
    ) -> ProviderUpload: ...

    async def cancel(self, upload_id: str) -> ProviderUpload: ...

    async def upload_bytes(
        self,
        *,
        data: bytes | bytearray | memoryview,
        filename: str,
        mime_type: str,
        purpose: str,
        part_size_bytes: int = 64 * 1024 * 1024,
        expires_after: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        md5: str | None = None,
    ) -> ProviderFile: ...


class ModerationsClient(Protocol):
    async def create(self, body: dict[str, Any], options: "RetryOptions | None" = None) -> dict[str, Any]: ...


class BatchesClient(Protocol):
    async def create(self, body: dict[str, Any], options: "RetryOptions | None" = None) -> dict[str, Any]: ...

    async def retrieve(self, batch_id: str, options: "RetryOptions | None" = None) -> dict[str, Any]: ...

    async def list(
        self,
        *,
        after: str | None = None,
        limit: int | None = None,
        options: "RetryOptions | None" = None,
    ) -> dict[str, Any]: ...

    async def cancel(self, batch_id: str, options: "RetryOptions | None" = None) -> dict[str, Any]: ...


@dataclass(slots=True)
class TokenCountDetail:
    modality: str | None = None
    token_count: int | None = None
    billable_characters: int | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CountTokensResult:
    total_tokens: int | None = None
    cached_content_token_count: int | None = None
    total_billable_characters: int | None = None
    details: list[TokenCountDetail] = field(default_factory=list)
    raw_response: Any = None


class CountTokensClient(Protocol):
    async def count(
        self,
        *,
        model_id: str,
        prompt: str | None = None,
        messages: list["ModelMessage"] | None = None,
        system: str | None = None,
        tools: dict[str, "ToolDefinition"] | None = None,
        provider_options: dict[str, Any] | None = None,
        options: "RetryOptions | None" = None,
    ) -> CountTokensResult: ...


@dataclass(slots=True)
class FileSearchStore:
    name: str
    display_name: str | None = None
    create_time: str | None = None
    update_time: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FileSearchStoreListResult:
    stores: list[FileSearchStore] = field(default_factory=list)
    next_page_token: str | None = None
    raw_response: Any = None


@dataclass(slots=True)
class FileSearchDocument:
    name: str
    display_name: str | None = None
    custom_metadata: list[dict[str, Any]] = field(default_factory=list)
    state: str | None = None
    size_bytes: int | None = None
    media_type: str | None = None
    create_time: str | None = None
    update_time: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FileSearchDocumentListResult:
    documents: list[FileSearchDocument] = field(default_factory=list)
    next_page_token: str | None = None
    raw_response: Any = None


@dataclass(slots=True)
class FileSearchOperation:
    name: str
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    response: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    raw_response: Any = None


@dataclass(slots=True)
class FileSearchSearchResult:
    results: list[dict[str, Any]] = field(default_factory=list)
    raw_response: Any = None


class FileSearchStoresClient(Protocol):
    async def create(
        self,
        *,
        display_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FileSearchStore: ...

    async def list(
        self,
        *,
        page_size: int | None = None,
        page_token: str | None = None,
    ) -> FileSearchStoreListResult: ...

    async def get(self, name: str) -> FileSearchStore: ...

    async def delete(self, name: str, *, force: bool = False) -> bool: ...

    async def update(
        self,
        name: str,
        *,
        display_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        expires_after: dict[str, Any] | None = None,
    ) -> FileSearchStore: ...

    async def upload(
        self,
        *,
        file_search_store_name: str,
        data: bytes | bytearray | memoryview,
        filename: str,
        media_type: str | None = None,
        display_name: str | None = None,
        custom_metadata: list[dict[str, Any]] | None = None,
        chunking_config: dict[str, Any] | None = None,
    ) -> FileSearchOperation: ...

    async def import_file(
        self,
        *,
        file_search_store_name: str,
        file_name: str,
        custom_metadata: list[dict[str, Any]] | None = None,
        chunking_config: dict[str, Any] | None = None,
    ) -> FileSearchOperation: ...

    async def list_documents(
        self,
        *,
        file_search_store_name: str,
        page_size: int | None = None,
        page_token: str | None = None,
    ) -> FileSearchDocumentListResult: ...

    async def get_document(self, name: str) -> FileSearchDocument: ...

    async def delete_document(self, name: str) -> bool: ...

    async def update_document(
        self,
        name: str,
        *,
        custom_metadata: list[dict[str, Any]] | None = None,
        chunking_config: dict[str, Any] | None = None,
    ) -> FileSearchDocument: ...

    async def search(
        self,
        *,
        file_search_store_name: str,
        query: str | list[str],
        filters: dict[str, Any] | None = None,
        max_num_results: int | None = None,
        ranking_options: dict[str, Any] | None = None,
        rewrite_query: bool | None = None,
    ) -> FileSearchSearchResult: ...

    async def get_operation(self, name: str) -> FileSearchOperation: ...

    async def wait_operation(
        self,
        name: str,
        *,
        poll_interval_ms: int = 500,
        timeout_ms: int | None = None,
    ) -> FileSearchOperation: ...


class ResponsesClient(Protocol):
    async def create(self, body: dict[str, Any], options: "RetryOptions | None" = None) -> dict[str, Any]: ...

    async def create_background(
        self,
        body: dict[str, Any],
        options: "RetryOptions | None" = None,
    ) -> dict[str, Any]: ...

    async def retrieve(
        self,
        response_id: str,
        *,
        include: list[str] | None = None,
        stream: bool | None = None,
        starting_after: int | None = None,
        include_obfuscation: bool | None = None,
        options: "RetryOptions | None" = None,
    ) -> dict[str, Any]: ...

    async def delete(self, response_id: str, options: "RetryOptions | None" = None) -> dict[str, Any]: ...

    async def list_input_items(
        self,
        response_id: str,
        *,
        after: str | None = None,
        before: str | None = None,
        limit: int | None = None,
        order: Literal["asc", "desc"] | None = None,
        include: list[str] | None = None,
        options: "RetryOptions | None" = None,
    ) -> dict[str, Any]: ...

    async def count_input_tokens(
        self,
        body: dict[str, Any],
        options: "RetryOptions | None" = None,
    ) -> dict[str, Any]: ...

    async def cancel(self, response_id: str, options: "RetryOptions | None" = None) -> dict[str, Any]: ...

    async def compact(
        self,
        body: dict[str, Any],
        options: "RetryOptions | None" = None,
    ) -> dict[str, Any]: ...

    async def wait(
        self,
        response_id: str,
        *,
        include: list[str] | None = None,
        stream: bool | None = None,
        starting_after: int | None = None,
        include_obfuscation: bool | None = None,
        poll_interval_ms: int = 500,
        timeout_ms: int | None = None,
        options: "RetryOptions | None" = None,
    ) -> dict[str, Any]: ...


class ConversationsClient(Protocol):
    async def create(self, body: dict[str, Any], options: "RetryOptions | None" = None) -> dict[str, Any]: ...

    async def retrieve(self, conversation_id: str, options: "RetryOptions | None" = None) -> dict[str, Any]: ...

    async def update(
        self,
        conversation_id: str,
        body: dict[str, Any],
        options: "RetryOptions | None" = None,
    ) -> dict[str, Any]: ...

    async def delete(self, conversation_id: str, options: "RetryOptions | None" = None) -> dict[str, Any]: ...

    async def create_item(
        self,
        conversation_id: str,
        body: dict[str, Any],
        *,
        include: list[str] | None = None,
        options: "RetryOptions | None" = None,
    ) -> dict[str, Any]: ...

    async def retrieve_item(
        self,
        conversation_id: str,
        item_id: str,
        *,
        include: list[str] | None = None,
        options: "RetryOptions | None" = None,
    ) -> dict[str, Any]: ...

    async def delete_item(
        self,
        conversation_id: str,
        item_id: str,
        options: "RetryOptions | None" = None,
    ) -> dict[str, Any]: ...

    async def list_items(
        self,
        conversation_id: str,
        *,
        after: str | None = None,
        before: str | None = None,
        limit: int | None = None,
        order: Literal["asc", "desc"] | None = None,
        include: list[str] | None = None,
        options: "RetryOptions | None" = None,
    ) -> dict[str, Any]: ...


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


@dataclass(slots=True)
class GeneratedCodePart:
    type: Literal["generated-code"] = "generated-code"
    code: str = ""
    language: str | None = "python"


@dataclass(slots=True)
class CodeExecutionResultPart:
    type: Literal["code-result"] = "code-result"
    output: str = ""
    outcome: str | None = None


ContentPart: TypeAlias = TextPart | ImagePart | FilePart | ToolCallPart | ToolResultPart | GeneratedCodePart | CodeExecutionResultPart


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
    realtime: bool = False
    realtime_audio_input: bool = False
    realtime_audio_output: bool = False
    realtime_tools: bool = False
    realtime_browser_tokens: bool = False


@dataclass(slots=True)
class RetryOptions:
    timeout_ms: int | None = None
    max_retries: int | None = None
    retry_backoff_ms: int | None = None


@dataclass(slots=True)
class RealtimeConnectOptions(RetryOptions):
    metadata: dict[str, Any] = field(default_factory=dict)
    browser_client: bool = False
    subprotocols: list[str] = field(default_factory=list)


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
    kind: str | None = None
    provider_metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class GroundingSupport:
    start_index: int | None = None
    end_index: int | None = None
    segment_text: str | None = None
    source_indices: list[int] = field(default_factory=list)
    provider_metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class GroundedGenerateResult(GenerateResult):
    sources: list[GroundingSource] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    supports: list[GroundingSupport] = field(default_factory=list)
    search_entry_point: dict[str, Any] | None = None


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
class RealtimeSessionConfig:
    instructions: str | None = None
    voice: str | None = None
    tools: dict[str, "ToolDefinition"] | None = None
    tool_choice: ToolChoice | None = None
    input_audio_media_type: str | None = None
    output_audio_media_type: str | None = None
    input_sample_rate_hz: int | None = None
    output_sample_rate_hz: int | None = None
    channels: int | None = None
    turn_detection: dict[str, Any] | None = None
    provider_options: ProviderOptions | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    auto_response: bool = True


@dataclass(slots=True)
class RealtimeTokenResult:
    value: str
    expires_at_ms: int | None = None
    raw_response: Any = None


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


@dataclass(slots=True)
class RealtimeSessionStartedEvent:
    type: Literal["realtime-start"] = "realtime-start"
    session_id: str | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RealtimeTextDeltaEvent:
    type: Literal["realtime-text-delta"] = "realtime-text-delta"
    text_delta: str = ""
    item_id: str | None = None
    response_id: str | None = None
    role: Literal["assistant"] = "assistant"
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RealtimeAudioOutputEvent:
    type: Literal["realtime-audio-output"] = "realtime-audio-output"
    audio: bytes = b""
    media_type: str = "audio/pcm"
    sample_rate_hz: int | None = None
    channels: int | None = None
    item_id: str | None = None
    response_id: str | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RealtimeTranscriptEvent:
    type: Literal["realtime-transcript"] = "realtime-transcript"
    text: str = ""
    role: Literal["user", "assistant"] = "assistant"
    is_final: bool = False
    item_id: str | None = None
    response_id: str | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RealtimeToolCallEvent:
    type: Literal["realtime-tool-call"] = "realtime-tool-call"
    tool_call: ToolCall = field(default_factory=lambda: ToolCall(id="", name="", input={}))


@dataclass(slots=True)
class RealtimeToolResultEvent:
    type: Literal["realtime-tool-result"] = "realtime-tool-result"
    tool_result: ToolExecutionResult = field(
        default_factory=lambda: ToolExecutionResult(tool_call_id="", tool_name="", is_error=False)
    )


@dataclass(slots=True)
class RealtimeSessionEndedEvent:
    type: Literal["realtime-end"] = "realtime-end"
    reason: str | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RealtimeSessionResumptionEvent:
    type: Literal["realtime-session-resumption"] = "realtime-session-resumption"
    new_handle: str | None = None
    resumable: bool | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RealtimeGoAwayEvent:
    type: Literal["realtime-go-away"] = "realtime-go-away"
    time_left_ms: int | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RealtimeErrorEvent:
    type: Literal["realtime-error"] = "realtime-error"
    error: Exception | None = None
    message: str | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)


RealtimeEvent: TypeAlias = (
    RealtimeSessionStartedEvent
    | RealtimeTextDeltaEvent
    | RealtimeAudioOutputEvent
    | RealtimeTranscriptEvent
    | RealtimeToolCallEvent
    | RealtimeToolResultEvent
    | RealtimeSessionEndedEvent
    | RealtimeSessionResumptionEvent
    | RealtimeGoAwayEvent
    | RealtimeErrorEvent
)


class RealtimeSession(Protocol):
    provider: str
    model_id: str
    capabilities: ModelCapabilities
    config: RealtimeSessionConfig

    async def send_audio(self, frame: AudioFrame) -> None: ...

    async def send_text(self, text: str) -> None: ...

    async def send_tool_result(self, result: ToolExecutionResult) -> None: ...

    async def update(
        self,
        *,
        instructions: str | None = None,
        voice: str | None = None,
        tools: dict[str, "ToolDefinition"] | None = None,
        tool_choice: ToolChoice | None = None,
        turn_detection: dict[str, Any] | None = None,
        provider_options: ProviderOptions | None = None,
    ) -> None: ...

    def event_stream(self) -> AsyncIterable[RealtimeEvent]: ...

    async def aclose(self) -> None: ...


class RealtimeModel(Protocol):
    provider: str
    model_id: str
    capabilities: ModelCapabilities

    async def connect(
        self,
        config: RealtimeSessionConfig | None = None,
        options: RealtimeConnectOptions | None = None,
    ) -> RealtimeSession: ...

    async def create_browser_token(
        self,
        config: RealtimeSessionConfig | None = None,
        options: RealtimeConnectOptions | None = None,
    ) -> RealtimeTokenResult: ...


TSchema = TypeVar("TSchema")


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str | None
    schema: Any
    execute: Callable[..., Awaitable[JsonValue] | JsonValue] | None = None
    input_examples: list[JsonValue] = field(default_factory=list)
    strict: bool | None = None
    defer_loading: bool | None = None
    eager_input_streaming: bool | None = None
    allowed_callers: list[str] = field(default_factory=list)
    cache_control: dict[str, Any] | None = None
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
    queries: list[str] = field(default_factory=list)
    supports: list[GroundingSupport] = field(default_factory=list)
    search_entry_point: dict[str, Any] | None = None
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
