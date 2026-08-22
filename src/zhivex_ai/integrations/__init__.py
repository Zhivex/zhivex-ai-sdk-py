"""Public protocol and hosting integrations.

The namespace is an additive facade over the existing Beta A2A, AG-UI, and
Responses-compatible hosting APIs. Existing imports remain supported.
"""

from __future__ import annotations

from .protocols import (
    A2A_PROTOCOL_VERSION,
    A2AAgentCard,
    A2AAgentExecutor,
    A2AAgentSkill,
    AGUIEvent,
    HostedAgentRunOptions,
    ProtocolErrorMapper,
    ProtocolEventCallback,
    ProtocolInvocation,
    ProtocolLimits,
    ProtocolRunOptionsResolver,
    create_a2a_agent_card,
    create_a2a_app,
    stream_agent_ag_ui,
    to_ag_ui_sse_response,
)
from .responses import (
    AgentResolver,
    InMemoryResponsesEventStore,
    ResponsesAgentHost,
    ResponsesEventStore,
    StoredResponsesRun,
    create_agent_playground_app,
    create_responses_app,
)

__all__ = [
    "A2A_PROTOCOL_VERSION",
    "A2AAgentCard",
    "A2AAgentExecutor",
    "A2AAgentSkill",
    "AGUIEvent",
    "AgentResolver",
    "HostedAgentRunOptions",
    "InMemoryResponsesEventStore",
    "ProtocolErrorMapper",
    "ProtocolEventCallback",
    "ProtocolInvocation",
    "ProtocolLimits",
    "ProtocolRunOptionsResolver",
    "ResponsesAgentHost",
    "ResponsesEventStore",
    "StoredResponsesRun",
    "create_a2a_agent_card",
    "create_a2a_app",
    "create_agent_playground_app",
    "create_responses_app",
    "stream_agent_ag_ui",
    "to_ag_ui_sse_response",
]
