# Claude 5.5 on the direct Anthropic API

The SDK validates the September–October 2026 Claude API contracts offline. The
package remains Beta; these tests do not establish live access or release
certification. Azure remains outside this refresh. Bedrock lists
`anthropic.claude-haiku-5-5` in the catalog.

| Model | Portable reasoning | Native thinking | Tool choice |
| --- | --- | --- | --- |
| `claude-opus-5-5` | `low` through `max` | adaptive, always on | auto / none |
| `claude-sonnet-5-5` | `low` through `max`; `none` maps to `between_tools` | adaptive / between_tools | auto / none |
| `claude-haiku-5-5` | `low` through `max` (no `none`) | adaptive only | auto / none / forced |

Sonnet's `none` skips up-front reasoning; it still permits thinking between tool
calls. Haiku 5.5 rejects `effort=none`, `thinking.disabled`, and `between_tools`;
use a lower adaptive effort such as `low`. Native `disabled` and manual token
budgets are rejected on Opus/Sonnet/Haiku 5.5. `between_tools` remains
Sonnet-only and accepts effort at most `high`. Opus defaults to `medium`;
Sonnet defaults to `high`; Haiku defaults to `medium`. Opus and Sonnet reject
forced tool use; Haiku 5.5 allows it per the Claude API. Haiku 5.5 has no
managed server-side fallback path for refusals.

Run `ANTHROPIC_API_KEY=... .venv/bin/python examples/text/anthropic_55.py`.

## Native client tools

For client computer or browser tools, use `provider.native.language_model(...)`
with `provider_options={"tools": [{"type": "computer_toolset_20260801"}]}`
or the corresponding `browser_toolset_20260801`. The SDK also accepts generic
`hosted_tool(..., tool_class="toolset")` declarations and omits the toolset's
legacy name field. These are client toolsets: your application supplies execution.
Older computer tool types are rejected on 5.5.

Each member call keeps `toolset_name` in `ToolCall.provider_metadata`. Dispatch
using that value together with `ToolCall.name`, and execute batched calls in order.
Copy `toolset_name` to `ToolExecutionResult.provider_metadata` when returning a
result. The mapper echoes it on the wire. For screenshot/image result blocks,
use native Messages with the raw content block structure.

## Conversation ownership

Preserve returned assistant messages, including signed thinking and encrypted
advisor content. Generated and streamed tool calls retain thinking metadata;
raw Messages preserves complete response blocks. Keep the conversation append-only
while reusing signatures. Local serialization cannot verify Anthropic's account,
model or history bindings. Inspect `raw_response.input_transformations` when using
`thinking-binding-controls-2026-08-01`; the service can drop incompatible blocks. Streams surface reported transformations
as provider-data events.

Sonnet 5.5 supports mid-conversation system messages. Rich inline tools and
per-message effort are available through raw native Messages with explicit beta
headers. When using `between_tools`, keep per-message effort unchanged. On-demand
compaction uses the existing native Messages client; retain the original history
until a usable signed summary is returned.

## Sources

- [Opus 5.5 migration](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide)
- [Sonnet 5.5 migration](https://platform.claude.com/docs/en/models/sonnet-5-5/migration-guide)
- [Haiku 5.5 overview](https://platform.claude.com/docs/en/models/haiku-5-5/overview)
- [Haiku 5.5 migration](https://platform.claude.com/docs/en/models/haiku-5-5/migration-guide)
- [Client toolsets](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference)
- [Handling tool calls](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls)
- [Advisor compatibility](https://platform.claude.com/docs/en/agents-and-tools/tool-use/advisor-tool)
