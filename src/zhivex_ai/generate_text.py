from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import AsyncIterable, Awaitable, Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

from .errors import (
    ParseError,
    ToolExecutionOutcomeUnknown,
    ToolExecutionSuspended,
    UnsupportedFeatureError,
    ValidationError,
)
from .messages import (
    create_text_message,
    get_agent_capabilities,
    get_hosted_tool_class,
    get_text_from_result,
    is_hosted_tool_definition,
    is_callable_tool_definition,
    normalize_finish_reason,
    provider_data_part,
    result_messages,
    serialize_json_value,
    tool_call_part,
    tool_result_part,
    validate_message_parts,
)
from .schema import create_schema_adapter
from .types import (
    GenerateResult,
    GenerateTextOutput,
    GenerateTextStep,
    LanguageModel,
    PortableRetrievalConfig,
    ModelGenerateInput,
    ModelMessage,
    ReasoningConfig,
    RetryOptions,
    StreamErrorEvent,
    StreamEvent,
    StreamFinishEvent,
    StreamProviderDataEvent,
    StreamToolCallEvent,
    StreamToolResultEvent,
    StreamTextDeltaEvent,
    HostedToolDefinition,
    ToolCall,
    ToolChoice,
    ToolChoiceName,
    ToolDefinition,
    ToolExecutionError,
    ToolExecutionContext,
    ToolExecutionOptions,
    ToolExecutionResult,
    ToolSet,
    TokenUsage,
)

_GEMINI_BUILTIN_SEARCH_TOOLS = {"search", "google_search", "googleSearch"}
_HOSTED_TOOL_CAPABILITY_FLAGS = {
    "web-search": "hosted_web_search",
    "file-search": "hosted_file_search",
    "remote-mcp": "remote_mcp",
    "computer-use": "computer_use",
    "code-execution": "code_execution",
    "toolset": "toolsets",
}
_NAMED_HOSTED_TOOL_CHOICE_PROVIDERS = {"anthropic"}
_REQUIRED_HOSTED_TOOL_CHOICE_ONLY_UNSUPPORTED_PROVIDERS = {"gemini", "vertex"}


def _is_provider_managed_tool_call(call: ToolCall) -> bool:
    return bool(call.provider_metadata.get("provider_managed"))


def _validate_reasoning(model: LanguageModel, reasoning: ReasoningConfig | None) -> None:
    if reasoning is None:
        return
    if not model.capabilities.reasoning:
        raise UnsupportedFeatureError(f'Model "{model.provider}/{model.model_id}" does not support reasoning.')
    if reasoning.effort is None and reasoning.budget_tokens is None:
        raise ValidationError('The "reasoning" config must include at least one supported field.')
    if reasoning.budget_tokens is not None and (reasoning.budget_tokens <= 0 or int(reasoning.budget_tokens) != reasoning.budget_tokens):
        raise ValidationError('The "reasoning.budgetTokens" field must be a positive integer.')


def _validate_input_source(prompt: str | None, messages: list[ModelMessage] | None) -> None:
    if prompt is not None and messages is not None:
        raise ValidationError('Pass either "prompt" or "messages", but not both.')


def _validate_tool_choice(model: LanguageModel, tools: ToolSet | None, tool_choice: str | ToolChoiceName | None) -> None:
    if tool_choice is None:
        return
    if not model.capabilities.tools:
        raise UnsupportedFeatureError(f'Model "{model.provider}/{model.model_id}" does not support tools.')
    if not model.capabilities.tool_choice:
        raise UnsupportedFeatureError(f'Model "{model.provider}/{model.model_id}" does not support tool choice.')
    if not tools:
        raise ValidationError('The "tool_choice" option requires at least one registered tool.')
    if isinstance(tool_choice, str) and tool_choice not in {"none", "auto", "required"}:
        raise ValidationError('The "tool_choice" option must be "none", "auto", "required", or ToolChoiceName(...).')
    if isinstance(tool_choice, ToolChoiceName) and tool_choice.tool_name not in tools:
        raise ValidationError(f'The selected tool "{tool_choice.tool_name}" is not registered.')
    if isinstance(tool_choice, ToolChoiceName):
        selected = tools.get(tool_choice.tool_name) if tools else None
        if (
            selected is not None
            and is_hosted_tool_definition(selected)
            and model.provider not in _NAMED_HOSTED_TOOL_CHOICE_PROVIDERS
        ):
            raise ValidationError(
                f'The selected tool "{tool_choice.tool_name}" is provider-managed. '
                f'ToolChoiceName(...) only supports hosted tools for {", ".join(sorted(_NAMED_HOSTED_TOOL_CHOICE_PROVIDERS))}.'
            )


def _validate_hosted_tools(model: LanguageModel, tools: ToolSet | None, tool_choice: ToolChoice | None) -> None:
    if not tools:
        return
    hosted_tools: list[HostedToolDefinition] = []
    callable_tools: list[ToolDefinition] = []
    for tool in tools.values():
        if is_hosted_tool_definition(tool):
            hosted_tools.append(cast(HostedToolDefinition, tool))
        else:
            callable_tools.append(cast(ToolDefinition, tool))
    if not hosted_tools:
        return
    if _is_portable_model(model):
        raise ValidationError(
            "Portable foundation APIs do not accept hosted tools. Use `provider.native.*` models for provider-managed tools."
        )

    capabilities = get_agent_capabilities(model)
    accepted_providers = {None, model.provider}
    if model.provider == "azure-openai":
        accepted_providers.add("openai")

    for hosted in hosted_tools:
        if hosted.provider not in accepted_providers:
            raise ValidationError(
                f'Hosted tool "{hosted.name}" targets provider "{hosted.provider}", but this model uses "{model.provider}".'
            )
        tool_class = get_hosted_tool_class(hosted)
        capability_name = _HOSTED_TOOL_CAPABILITY_FLAGS.get(tool_class)
        if capability_name is not None and not getattr(capabilities, capability_name):
            raise UnsupportedFeatureError(
                f'Model "{model.provider}/{model.model_id}" does not support hosted tool class "{tool_class}".'
            )

    if tool_choice == "none" and not capabilities.tool_choice_none:
        raise UnsupportedFeatureError(
            f'Model "{model.provider}/{model.model_id}" does not support `tool_choice=\"none\"` with hosted tools.'
        )
    if (
        tool_choice == "required"
        and not callable_tools
        and model.provider in _REQUIRED_HOSTED_TOOL_CHOICE_ONLY_UNSUPPORTED_PROVIDERS
    ):
        raise UnsupportedFeatureError(
            f'Model "{model.provider}/{model.model_id}" cannot guarantee `tool_choice=\"required\"` when only hosted tools are registered.'
        )


def normalize_messages(
    *,
    prompt: str | None = None,
    messages: list[ModelMessage] | None = None,
    system: str | None = None,
) -> list[ModelMessage]:
    _validate_input_source(prompt, messages)
    built = list(messages or [])
    if system:
        built.insert(0, create_text_message("system", system))
    if prompt:
        built.append(create_text_message("user", prompt))
    return built


def _is_portable_model(model: Any) -> bool:
    return bool(getattr(model, "portable", False))


def _validate_portable_provider_options(model: Any, provider_options: dict[str, Any] | None) -> None:
    if _is_portable_model(model) and provider_options is not None:
        raise ValidationError(
            "Portable foundation APIs do not accept provider_options. "
            "Use `provider.native.*` models when you need provider-specific configuration."
        )


def _apply_retrieval(
    messages: list[ModelMessage],
    retrieval: PortableRetrievalConfig | None,
) -> list[ModelMessage]:
    if retrieval is None:
        return messages
    if not retrieval.documents:
        raise ValidationError('The "retrieval.documents" field must include at least one document.')
    if retrieval.max_documents <= 0:
        raise ValidationError('The "retrieval.max_documents" field must be positive.')
    if retrieval.max_document_chars <= 0:
        raise ValidationError('The "retrieval.max_document_chars" field must be positive.')
    selected = retrieval.documents[: retrieval.max_documents]
    sections: list[str] = []
    for index, document in enumerate(selected, start=1):
        excerpt = document.text.strip()[: retrieval.max_document_chars]
        sections.append(
            f"[Document {index}] id={document.document_id}"
            + (f" title={document.title}" if document.title else "")
            + f"\n{excerpt}"
        )
    retrieval_message = create_text_message(
        "system",
        "Use the following retrieved context when it is relevant.\n\n" + "\n\n".join(sections),
    )
    return [retrieval_message, *messages]


def _to_request(
    *,
    messages: list[ModelMessage],
    tools: ToolSet | None,
    temperature: float | None,
    max_tokens: int | None,
    reasoning: ReasoningConfig | None,
    provider_options: dict[str, Any] | None,
    structured_output: Any,
    tool_choice: ToolChoice | None,
    retry: RetryOptions,
) -> ModelGenerateInput:
    return ModelGenerateInput(
        messages=list(messages),
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning=reasoning,
        provider_options=provider_options,
        structured_output=structured_output,
        timeout_ms=retry.timeout_ms,
        max_retries=retry.max_retries,
        retry_backoff_ms=retry.retry_backoff_ms,
    )


async def _invoke_tool_callable_async(execute: Any, parsed: Any, context: ToolExecutionContext) -> Any:
    is_async = inspect.iscoroutinefunction(execute) or inspect.iscoroutinefunction(getattr(execute, "__call__", None))
    if is_async:
        output = _invoke_tool_callable(execute, parsed, context)
    else:
        output = await asyncio.to_thread(_invoke_tool_callable, execute, parsed, context)
    return await output if inspect.isawaitable(output) else output


async def _execute_tool(call: ToolCall, tools: ToolSet | None, timeout_ms: int | None = None) -> ToolExecutionResult:
    tool = tools.get(call.name) if tools else None
    if tool is None:
        raise ValidationError(f'Tool "{call.name}" was requested by the model but is not registered.')
    if not is_callable_tool_definition(tool):
        raise ValidationError(f'Tool "{call.name}" is provider-managed and cannot run in the local tool runtime.')
    callable_tool = cast(ToolDefinition, tool)
    if callable_tool.execute is None:
        raise ValidationError(f'Tool "{call.name}" is registered but does not have a local executor.')
    adapter = create_schema_adapter(callable_tool.schema)
    try:
        parsed = adapter.validate_python(call.input)
    except Exception as error:
        raise ValidationError(f'Invalid input for tool "{call.name}": {error}') from error
    try:
        idempotency_prefix = str(callable_tool.metadata.get("zhivex_tool_idempotency_prefix") or "").strip()
        idempotency_key = (
            f"{idempotency_prefix}:{call.id or call.name}"
            if idempotency_prefix
            else call.id or f"{call.name}:tool-call"
        )
        context = ToolExecutionContext(
            tool_name=call.name,
            tool_call_id=call.id,
            idempotency_key=idempotency_key,
            deadline_ms=int(time.time() * 1000) + timeout_ms if timeout_ms is not None else None,
            permissions=list(callable_tool.permissions),
            source=callable_tool.source,
            metadata=dict(callable_tool.metadata),
        )
        execution = asyncio.create_task(_invoke_tool_callable_async(callable_tool.execute, parsed, context))
        try:
            output = await asyncio.wait_for(execution, timeout_ms / 1000) if timeout_ms is not None else await execution
        except TimeoutError as error:
            if not execution.cancelled():
                raise
            raise ToolExecutionOutcomeUnknown(
                f'Tool "{call.name}" exceeded its {timeout_ms} ms timeout; its external outcome is unknown. '
                f'Reconcile the side effect with idempotency key "{idempotency_key}" before retrying.',
                tool_name=call.name,
                tool_call_id=call.id,
                timeout_ms=cast(int, timeout_ms),
                idempotency_key=idempotency_key,
            ) from error
        return ToolExecutionResult(
            tool_call_id=call.id,
            tool_name=call.name,
            output=serialize_json_value(output),
            is_error=False,
            provider_metadata=dict(call.provider_metadata),
        )
    except ToolExecutionSuspended:
        raise
    except ToolExecutionOutcomeUnknown:
        raise
    except Exception as error:
        return ToolExecutionResult(
            tool_call_id=call.id,
            tool_name=call.name,
            error=ToolExecutionError(message=str(error) or "Tool execution failed."),
            is_error=True,
            provider_metadata=dict(call.provider_metadata),
        )


def _invoke_tool_callable(execute: Any, parsed: Any, context: ToolExecutionContext) -> Any:
    mode = _tool_callable_mode(execute)
    if mode == "kwargs":
        return execute(parsed, context=context)
    if mode == "positional":
        return execute(parsed, context)
    return execute(parsed)


@lru_cache(maxsize=256)
def _tool_callable_mode(execute: Any) -> str:
    try:
        signature = inspect.signature(execute)
    except (TypeError, ValueError):
        return "single"

    parameters = list(signature.parameters.values())
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return "kwargs"
    if any(parameter.name == "context" for parameter in parameters):
        return "kwargs"
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) >= 2:
        return "positional"
    return "single"


async def _execute_tools(
    tool_calls: list[ToolCall],
    tools: ToolSet | None,
    *,
    options: ToolExecutionOptions | None = None,
    default_parallel: bool = False,
) -> list[ToolExecutionResult]:
    parallel = options.parallel if options and options.parallel is not None else default_parallel
    # A human-approval suspension is a transactional boundary. Execute the
    # whole batch in order whenever one of the tools can suspend so later side
    # effects cannot race ahead of the pending approval.
    if parallel and tools is not None:
        for call in tool_calls:
            definition = tools.get(call.name)
            if (
                definition is not None
                and is_callable_tool_definition(definition)
                and (
                    definition.requires_approval is True
                    or bool(definition.metadata.get("zhivex_agent_approval_gated"))
                )
            ):
                parallel = False
                break
    timeout_ms = options.timeout_ms if options else None
    if timeout_ms is not None and timeout_ms <= 0:
        raise ValidationError('The "tool_execution.timeout_ms" field must be greater than zero.')
    stop_on_error = options.stop_on_error if options else False
    max_concurrency = max(1, options.max_concurrency or len(tool_calls) or 1) if options else max(1, len(tool_calls) or 1)

    async def execute_single(call: ToolCall) -> ToolExecutionResult:
        return await _execute_tool(call, tools, timeout_ms=timeout_ms)

    if not parallel or len(tool_calls) <= 1:
        sequential_results: list[ToolExecutionResult] = []
        for call in tool_calls:
            try:
                result = await execute_single(call)
            except ToolExecutionSuspended as suspended:
                suspended.tool_results = [*sequential_results, *suspended.tool_results]
                raise
            sequential_results.append(result)
            if stop_on_error and result.is_error:
                raise RuntimeError(
                    f'Tool "{result.tool_name}" failed: {(result.error.message if result.error else "Unknown tool error.")}'
                )
        return sequential_results

    results: list[ToolExecutionResult | None] = [None] * len(tool_calls)
    cursor = 0

    async def worker() -> None:
        nonlocal cursor
        while cursor < len(tool_calls):
            index = cursor
            cursor += 1
            results[index] = await execute_single(tool_calls[index])

    await asyncio.gather(*(worker() for _ in range(min(max_concurrency, len(tool_calls)))))
    resolved = [result for result in results if result is not None]
    if stop_on_error:
        first_error = next((result for result in resolved if result.is_error), None)
        if first_error is not None:
            raise RuntimeError(
                f'Tool "{first_error.tool_name}" failed: {(first_error.error.message if first_error.error else "Unknown tool error.")}'
            )
    return resolved


def _record_suspended_tool_state(
    suspended: ToolExecutionSuspended,
    *,
    messages: list[ModelMessage],
    steps: list[GenerateTextStep],
    previous_tool_results: list[ToolExecutionResult],
) -> None:
    partial_results = cast(list[ToolExecutionResult], suspended.tool_results)
    completed_ids = {result.tool_call_id for result in partial_results}
    pending_call_id = str(getattr(suspended.pending_approval, "tool_call_id", "") or "")
    allowed_call_ids = completed_ids | ({pending_call_id} if pending_call_id else set())
    filtered_messages = [
        ModelMessage(
            role=message.role,
            parts=[
                part
                for part in message.parts
                if part.type != "tool-call" or part.tool_call.id in allowed_call_ids
            ],
        )
        for message in messages
    ]
    for result in partial_results:
        filtered_messages.append(ModelMessage(role="tool", parts=[tool_result_part(result)]))
    suspended.messages = filtered_messages
    suspended.steps = list(steps)
    suspended.tool_results = [*previous_tool_results, *partial_results]


def _raise_for_provider_builtin_tool_calls(
    model: LanguageModel,
    tool_calls: list[ToolCall],
    provider_options: dict[str, Any] | None,
) -> None:
    if model.provider != "gemini":
        return
    builtin_search_calls = [call for call in tool_calls if call.name in _GEMINI_BUILTIN_SEARCH_TOOLS]
    if not builtin_search_calls:
        return
    if provider_options and provider_options.get("google_search"):
        raise UnsupportedFeatureError(
            'Gemini returned an unexpected built-in Google Search tool call while `provider_options={"google_search": True}` '
            "was enabled. Use `create_gemini().grounded_language_model(...)` when you need grounded sources."
        )
    raise UnsupportedFeatureError(
        'Gemini requested its built-in Google Search tool, but Google Search is opt-in in this SDK. '
        'Retry with `provider_options={"google_search": True}` or use `create_gemini().grounded_language_model(...)`.'
    )


def _merge_usage(usages: list[TokenUsage | None]) -> TokenUsage | None:
    present = [usage for usage in usages if usage is not None]
    if not present:
        return None
    input_tokens = (
        sum(usage.input_tokens for usage in present if usage.input_tokens is not None)
        if all(usage.input_tokens is not None for usage in present)
        else None
    )
    output_tokens = (
        sum(usage.output_tokens for usage in present if usage.output_tokens is not None)
        if all(usage.output_tokens is not None for usage in present)
        else None
    )
    total_tokens = (
        sum(usage.total_tokens for usage in present if usage.total_tokens is not None)
        if all(usage.total_tokens is not None for usage in present)
        else None
    )
    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens)


def _extract_tool_calls(messages: list[ModelMessage]) -> list[ToolCall]:
    tool_calls: list[ToolCall] = []
    for message in messages:
        for part in message.parts:
            if part.type == "tool-call" and not _is_provider_managed_tool_call(part.tool_call):
                tool_calls.append(part.tool_call)
    return tool_calls


async def generate_text(
    *,
    model: LanguageModel,
    prompt: str | None = None,
    messages: list[ModelMessage] | None = None,
    system: str | None = None,
    tools: ToolSet | None = None,
    tool_choice: ToolChoice | None = None,
    tool_execution: ToolExecutionOptions | None = None,
    max_steps: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning: ReasoningConfig | None = None,
    provider_options: dict[str, Any] | None = None,
    retrieval: PortableRetrievalConfig | None = None,
    structured_output: Any = None,
    timeout_ms: int | None = None,
    max_retries: int | None = None,
    retry_backoff_ms: int | None = None,
) -> GenerateTextOutput:
    steps_limit = max(1, max_steps or 1)
    all_messages = normalize_messages(prompt=prompt, messages=messages, system=system)
    all_messages = _apply_retrieval(all_messages, retrieval)
    validate_message_parts(model, all_messages)
    _validate_portable_provider_options(model, provider_options)
    _validate_reasoning(model, reasoning)
    if tools and not model.capabilities.tools:
        raise UnsupportedFeatureError(f'Model "{model.provider}/{model.model_id}" does not support tools.')
    _validate_tool_choice(model, tools, tool_choice)
    _validate_hosted_tools(model, tools, tool_choice)

    retry = RetryOptions(timeout_ms=timeout_ms, max_retries=max_retries, retry_backoff_ms=retry_backoff_ms)
    steps: list[GenerateTextStep] = []
    tool_results: list[ToolExecutionResult] = []
    final_result: GenerateResult | None = None

    for _ in range(steps_limit):
        request = _to_request(
            messages=all_messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning=reasoning,
            provider_options=provider_options,
            structured_output=structured_output,
            tool_choice=tool_choice,
            retry=retry,
        )
        response = await model.generate(request)
        steps.append(GenerateTextStep(request=request, response=response))
        final_result = response
        response_messages = result_messages(response)
        if response_messages:
            all_messages.extend(response_messages)
        step_tool_calls = _extract_tool_calls(response_messages)
        if not step_tool_calls:
            break
        _raise_for_provider_builtin_tool_calls(model, step_tool_calls, provider_options)
        try:
            current_results = await _execute_tools(
                step_tool_calls,
                tools,
                options=tool_execution,
                default_parallel=model.capabilities.parallel_tool_calls,
            )
        except ToolExecutionSuspended as suspended:
            _record_suspended_tool_state(
                suspended,
                messages=all_messages,
                steps=steps,
                previous_tool_results=tool_results,
            )
            raise
        tool_results.extend(current_results)
        for result in current_results:
            all_messages.append(ModelMessage(role="tool", parts=[tool_result_part(result)]))

    if final_result is None:
        raise ParseError("Model did not return a result.")

    merged_usage = _merge_usage([step.response.usage for step in steps])
    final_text = get_text_from_result(final_result)
    return GenerateTextOutput(
        text=final_text,
        finish_reason=final_result.finish_reason,
        provider_finish_reason=final_result.provider_finish_reason,
        usage=merged_usage,
        steps=steps,
        messages=all_messages,
        tool_results=tool_results,
    )


@dataclass
class _Broadcast:
    history: list[StreamEvent]
    done: bool = False
    error: Exception | None = None
    subscribers: list[asyncio.Queue[StreamEvent | None]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.subscribers = []

    async def publish(self, event: StreamEvent) -> None:
        self.history.append(event)
        for queue in list(self.subscribers):
            await queue.put(event)

    async def close(self) -> None:
        self.done = True
        for queue in list(self.subscribers):
            await queue.put(None)

    def stream(self) -> AsyncIterable[StreamEvent]:
        async def generator() -> AsyncIterable[StreamEvent]:
            queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
            cursor = 0
            self.subscribers.append(queue)
            try:
                while True:
                    while cursor < len(self.history):
                        event = self.history[cursor]
                        cursor += 1
                        yield event
                    if self.done:
                        return
                    item = await queue.get()
                    if item is None:
                        return
            finally:
                if queue in self.subscribers:
                    self.subscribers.remove(queue)

        return generator()


class _StreamTextResult:
    def __init__(self, runner: asyncio.Task[GenerateTextOutput], broadcast: _Broadcast) -> None:
        self._runner = runner
        self._broadcast = broadcast

    def event_stream(self) -> AsyncIterable[StreamEvent]:
        return self._broadcast.stream()

    def text_stream(self) -> AsyncIterable[str]:
        async def generator() -> AsyncIterable[str]:
            async for event in self._broadcast.stream():
                if isinstance(event, StreamTextDeltaEvent):
                    yield event.text_delta

        return generator()

    async def collect(self) -> GenerateTextOutput:
        return await self._runner


def stream_text(
    *,
    model: LanguageModel,
    prompt: str | None = None,
    messages: list[ModelMessage] | None = None,
    system: str | None = None,
    tools: ToolSet | None = None,
    tool_choice: ToolChoice | None = None,
    tool_execution: ToolExecutionOptions | None = None,
    max_steps: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning: ReasoningConfig | None = None,
    provider_options: dict[str, Any] | None = None,
    retrieval: PortableRetrievalConfig | None = None,
    structured_output: Any = None,
    timeout_ms: int | None = None,
    max_retries: int | None = None,
    retry_backoff_ms: int | None = None,
    on_event: Callable[[StreamEvent], Awaitable[None]] | None = None,
) -> _StreamTextResult:
    base_messages = normalize_messages(prompt=prompt, messages=messages, system=system)
    base_messages = _apply_retrieval(base_messages, retrieval)
    validate_message_parts(model, base_messages)
    _validate_portable_provider_options(model, provider_options)
    _validate_reasoning(model, reasoning)
    if not model.capabilities.streaming:
        raise ValidationError(f'Model "{model.provider}/{model.model_id}" does not support streaming.')
    if tools and not model.capabilities.tools:
        raise UnsupportedFeatureError(f'Model "{model.provider}/{model.model_id}" does not support tools.')
    _validate_tool_choice(model, tools, tool_choice)
    _validate_hosted_tools(model, tools, tool_choice)

    steps_limit = max(1, max_steps or 1)
    retry = RetryOptions(timeout_ms=timeout_ms, max_retries=max_retries, retry_backoff_ms=retry_backoff_ms)
    broadcast = _Broadcast(history=[])

    async def runner() -> GenerateTextOutput:
        all_messages = list(base_messages)
        steps: list[GenerateTextStep] = []
        tool_results: list[ToolExecutionResult] = []
        final_result: GenerateResult | None = None

        try:
            for _ in range(steps_limit):
                request = _to_request(
                    messages=all_messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    reasoning=reasoning,
                    provider_options=provider_options,
                    structured_output=structured_output,
                    tool_choice=tool_choice,
                    retry=retry,
                )
                stream = await model.stream(request)
                assistant_parts: list[Any] = []
                text_buffer = ""
                finish_reason = normalize_finish_reason("stop")
                provider_finish_reason: str | None = None
                usage = None

                async for event in stream:
                    if on_event is not None:
                        await on_event(event)
                    await broadcast.publish(event)
                    if isinstance(event, StreamTextDeltaEvent):
                        text_buffer += event.text_delta
                        assistant_parts.append(create_text_message("assistant", event.text_delta).parts[0])
                    elif isinstance(event, StreamToolCallEvent):
                        assistant_parts.append(tool_call_part(event.tool_call))
                    elif isinstance(event, StreamProviderDataEvent):
                        assistant_parts.append(provider_data_part(event.provider, event.data))
                    elif isinstance(event, StreamFinishEvent):
                        finish_reason = event.finish_reason
                        provider_finish_reason = event.provider_finish_reason
                        usage = event.usage
                        if finish_reason == "refusal":
                            text_buffer = ""
                            assistant_parts = [part for part in assistant_parts if part.type != "text"]

                response_messages = [ModelMessage(role="assistant", parts=assistant_parts)] if assistant_parts else []
                response = GenerateResult(
                    messages=response_messages,
                    text=text_buffer,
                    finish_reason=finish_reason,
                    provider_finish_reason=provider_finish_reason,
                    usage=usage,
                )
                steps.append(GenerateTextStep(request=request, response=response))
                final_result = response
                if response_messages:
                    all_messages.extend(response_messages)
                step_tool_calls = _extract_tool_calls(response_messages)
                if not step_tool_calls:
                    break
                _raise_for_provider_builtin_tool_calls(model, step_tool_calls, provider_options)
                try:
                    current_results = await _execute_tools(
                        step_tool_calls,
                        tools,
                        options=tool_execution,
                        default_parallel=model.capabilities.parallel_tool_calls,
                    )
                except ToolExecutionSuspended as suspended:
                    _record_suspended_tool_state(
                        suspended,
                        messages=all_messages,
                        steps=steps,
                        previous_tool_results=tool_results,
                    )
                    raise
                tool_results.extend(current_results)
                for result in current_results:
                    all_messages.append(ModelMessage(role="tool", parts=[tool_result_part(result)]))
                    tool_result_event = StreamToolResultEvent(tool_result=result)
                    if on_event is not None:
                        await on_event(tool_result_event)
                    await broadcast.publish(tool_result_event)
            if final_result is None:
                raise ParseError("Model did not return a result.")
            merged_usage = _merge_usage([step.response.usage for step in steps])
            final_text = get_text_from_result(final_result)
            return GenerateTextOutput(
                text=final_text,
                finish_reason=final_result.finish_reason,
                provider_finish_reason=final_result.provider_finish_reason,
                usage=merged_usage,
                steps=steps,
                messages=all_messages,
                tool_results=tool_results,
            )
        except Exception as error:
            error_event = StreamErrorEvent(error=error)
            if on_event is not None:
                await on_event(error_event)
            await broadcast.publish(error_event)
            raise
        finally:
            await broadcast.close()

    return _StreamTextResult(asyncio.create_task(runner()), broadcast)
