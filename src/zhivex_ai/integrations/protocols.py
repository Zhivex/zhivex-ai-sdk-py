"""A2A and AG-UI integration contracts.

All objects are compatibility re-exports from :mod:`zhivex_ai.protocols`.
"""

from __future__ import annotations

from ..protocols import (
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

__all__ = [
    "A2A_PROTOCOL_VERSION",
    "A2AAgentCard",
    "A2AAgentExecutor",
    "A2AAgentSkill",
    "AGUIEvent",
    "HostedAgentRunOptions",
    "ProtocolErrorMapper",
    "ProtocolEventCallback",
    "ProtocolInvocation",
    "ProtocolLimits",
    "ProtocolRunOptionsResolver",
    "create_a2a_agent_card",
    "create_a2a_app",
    "stream_agent_ag_ui",
    "to_ag_ui_sse_response",
]
