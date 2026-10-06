# Experimental computer execution

This document describes the pending 0.29.0 release, not a claim about the published
0.28.4 wheel. The published declaration helpers alone do not execute a desktop.

`zhivex_ai.experimental.openai_computer_tool` integrates OpenAI Responses GA
`computer` / `actions[]` with `generate_text`, `stream_text`, `run_agent`, and durable
approval continuation through `resume_agent_run`. Use an OpenAI **native** language
model. The host supplies an isolated environment, async authorization and execution;
there is no OS driver, browser, VM, paid-model access, or desktop permission in this
SDK. See the offline [example](../examples/agents/computer_use.py).

The ordered batch contains `call_id`, `actions`, and complete
`pending_safety_checks`. `authorize(batch, context)` must return
`ComputerApproval(approved=True, acknowledged_safety_check_ids=(...))`, explicitly
listing exactly the warning IDs that were reviewed. Generic Agent approval does not
acknowledge provider warnings. A denied Agent policy stops before this authorizer.
A durable generic approval still invokes the provider-warning authorizer when
resumed, so the host must recheck the current observation and session.

Approval and execution receive separate snapshots; changing the approval snapshot
does not change the executed actions. The runtime rejects malformed batches,
missing executors, mismatched call IDs, modified guarded inputs, unknown action
types, and repeated native call IDs in the retained transcript. Supported GA actions
are click, double_click, move, drag, scroll, keypress, type, wait, and screenshot.
Coordinates are nonnegative integer screenshot pixels; the host must validate
viewport bounds and perform DPI/crop/coordinate transformations itself. The host
must validate the whole batch before any effect, execute actions sequentially,
revalidate session/origin/observation authorization immediately before effects, and
capture a fresh image after all actions. Returning `None` is not success.
Explicitly unfinished or unsupported computer-call item statuses are rejected,
even inside a top-level completed response or terminal stream event.

The executor returns `ComputerScreenshot(image_url=...)`, a bounded base64 PNG,
JPEG, or WebP data URL. The SDK checks encoding and byte limits, not decoded pixels,
dimensions, target ownership, or actual task success. The host must decode safely,
mask sensitive pixels, and verify viewport/session identity. The next request uses
`computer_call_output` with the original call ID, image at `detail: "original"`
(to preserve resolution for coordinates), and only the acknowledged
provider checks. Retained transcripts contain screenshots and typed text: keep them
out of ordinary logs and apply application retention/access controls.

Both callbacks must be async. `callback_timeout_ms` bounds the combined approval
and execution time and respects the tool context's earlier deadline. Cancellation
or timeout stops the SDK from waiting even if a callback suppresses cancellation.
It cannot stop an external driver or undo an effect; late callback exceptions are
observed and late results are discarded. The app must terminate or quarantine the
session itself. Before executor entry, failure prevents execution; after executor
entry, failures, invalid/missing screenshots, timeouts and cancellation raise
`ToolExecutionOutcomeUnknown` with call/idempotency correlation. The same
classification applies if Agent output guardrails, end hooks, or result serialization
fail after executor entry, including durable approval continuation. Approval
timeouts before executor entry do not indicate an unknown external effect. Never retry that
batch blindly. Journal before effects, retain receipts, reconcile with the session,
and only then start new work. The same limitation applies to a new `run_agent`,
`resume_agent(session_id=...)`, restored checkpoint, or application retry: the SDK
has no external effect journal or exactly-once guarantee. A failed durable run is
not an approval-suspended run and cannot use `resume_agent_run` to retry effects.

`run_computer_use(agent=..., is_complete=..., **run_options)` is a small experimental
wrapper around the existing Agent runtime. Its required async, read-only app
postcondition runs even when the model ends with text and no tool calls. It returns
`ComputerRunResult(run=..., verified=...)`; only literal `True` verifies success.
A suspended run is unverified and does not call the postcondition. Verification
has its own bounded deadline; errors/timeouts propagate without rerunning effects.
Plain `Agent` status `completed` means the model/runtime finished, not that a UI
goal was independently verified. An unchanged screenshot or no-op action alone is
never proof of task success.

## Provider boundaries

The legacy `openai_computer_use_tool()` remains a declaration helper with its
existing preview default. Preview singular `action`, Azure preview, Anthropic
computer/browser toolsets, Gemini GenerateContent/Interactions and Vertex native
computer protocols have **no automatic computer executor in this change**. Raw
native clients and existing generic tools remain available for application-owned
controllers. Native computer calls are client effects, so unsupported/missing
handlers fail closed rather than yielding a falsely completed Agent. This does not
claim portable provider parity, model availability, or live certification.

Generic tool execution context metadata now includes `provider_metadata` from the
call, including Anthropic toolset namespace where present. That aids explicit host
dispatch but does not implement a namespaced registry or certify native toolsets.
