# Stability

Zhivex AI SDK uses three stability levels so production integrators can understand which surfaces are intended to remain predictable over time.

Supported public imports should come from `zhivex_ai`. Deep imports from internal modules are not part of the stable contract unless this document names an explicit exception.

Related documents:

- [README.md](./README.md)
- [VERSIONING.md](./VERSIONING.md)
- [CHANGELOG.md](./CHANGELOG.md)

## Stable

These APIs are the supported public contract for application code and production integrations:

- Provider factories: `create_openai`, `create_anthropic`, `create_azure_openai`, `create_gemini`, `create_vertex`
- Text generation: `generate_text`, `stream_text`
- Structured output: `generate_object`, `stream_object`
- Grounded text: `generate_grounded_text`
- Embeddings: `embed`, `embed_many`
- Agent runtime: `Agent`, `AgentSession`, `ToolRegistry`, `run_agent`, `stream_agent`, `resume_agent`, `create_agent_session`, `load_agent_session`
- Agent skills: `skill`, `load_skill`, `discover_skills`, `SkillDefinition`, `SkillDependency`, `SkillRegistry`
- Agent skill session controls: `set_agent_session_skills`, `get_agent_session_skills`, `clear_agent_session_skills`
- Agent skill observability: `AgentSkillActivatedEvent`, `AgentSkillSkippedEvent`
- Agent persistence: `create_postgres_agent_memory_store`, `create_postgres_checkpoint_store`
- MCP helpers and registries: `discover_mcp_tools`, `mcp_stdio_server`, `mcp_http_server`, `create_mcp_tool_registry`
- Gateway: `GatewayAttempt`, `GatewayConfig`, `GatewayError`, `GatewayImageAttachment`, `GatewayMessage`, `GatewayModelTarget`, `GatewayObjectResponse`, `GatewayResponse`, `create_gateway`
- Core errors: `ProviderHTTPError`, `ConfigurationError`, `ValidationError`, `UnsupportedFeatureError`
- HTTP and SSE helpers: `HTTPResponse`, `stream_sse`, `to_sse_response`, `to_sse_stream`, `to_text_stream`, `to_text_stream_response`, `to_ui_message_stream_response`

The stable surface is intentionally narrow. It reflects the most defendable cross-provider experience and the main API-building primitives in this SDK today.

## Beta

These APIs are supported and documented, but they may still change between minor releases as the SDK matures:

- Middleware helpers
- Model catalog helpers
- Provider agent capability metadata: `AgentCapabilities`, `AgentSupportTier`, `get_agent_capabilities`, `get_agent_support_tier`
- First-class hosted tool model: `HostedToolDefinition`, `HostedToolClass`, `AnyToolDefinition`, `hosted_tool`, `is_hosted_tool_definition`, `is_callable_tool_definition`, `get_hosted_tool_class`, `is_hosted_tool_class`
- Provider-data content parts and hosted-tool control payloads: `ProviderDataPart`, `provider_data_part`, `get_provider_data_parts`, `get_last_provider_data_part`, `openai_mcp_approval_response`, `azure_openai_mcp_approval_response`
- Typed OpenAI/Azure provider-data payloads and parsers: `OpenAIResponseReference`, `OpenAIMcpApprovalRequest`, `OpenAIMcpApprovalResponse`, `OpenAIMcpCall`, `OpenAIMcpListTools`, `OpenAIProviderData`, `AzureOpenAIResponseReference`, `AzureOpenAIMcpApprovalRequest`, `AzureOpenAIMcpApprovalResponse`, `AzureOpenAIMcpCall`, `AzureOpenAIMcpListTools`, `AzureOpenAIProviderData`, `parse_openai_provider_data_part`, `parse_azure_openai_provider_data_part`
- Response-reference helpers: `openai_response_reference`, `get_openai_response_reference`, `get_openai_response_id`, `azure_openai_response_reference`, `get_azure_openai_response_reference`, `get_azure_openai_response_id`
- Hosted-tool streaming transport: `StreamProviderDataEvent`, `UIMessageProviderDataChunk`
- Packaged skill APIs and installers: `load_skill_package`, `validate_skill`, `install_skill`, `list_installed_skills`, `run_skill`, `publish_skill`
- Packaged skill types and artifacts: `SkillArtifact`, `SkillEntrypoint`, `SkillPermissions`, `SkillPackageManifest`, `InstalledSkill`, `SkillRegistryIndex`, `SkillRunResult`
- Packaged skill runtime events: `AgentSkillResolvedEvent`, `AgentSkillDependencyCheckEvent`, `AgentSkillExecutionStartEvent`, `AgentSkillExecutionFinishEvent`, `AgentSkillArtifactCreatedEvent`

Beta APIs still require changelog coverage when they change, but they do not carry the same compatibility guarantees as the stable surface.

The README support matrix is generated from runtime metadata. Its `Agent Capabilities` section is useful product guidance for hosted tools and provider-managed events, but it should be read with the same beta expectations as the APIs listed above.

## Experimental

These areas are available for evaluation, but they should not be treated as a long-term compatibility contract yet:

- Realtime and live voice flows, including `stream_live_agent()`
- Raw provider payload escape hatches that do not map cleanly to the hosted-tool beta surface
- Provider areas currently marked as `native-only` or `compatibility` in the support matrix

Experimental areas may change faster than the rest of the SDK. Production adopters should isolate usage behind their own service layer before depending on them.

## Provider scope

The current tier-1 provider story for the stable surface is:

- OpenAI
- Anthropic
- Azure OpenAI
- Gemini
- Vertex

In this repository, tier-1 means the provider is part of the stable surface story, production API guidance, and support-matrix contract checks.

Anthropic is included in the tier-1 set for text-generation API paths. Embeddings, transcription, and speech remain outside the current Anthropic provider surface in this SDK.

Other providers remain useful, but they should be evaluated with the support matrix and the stability level of the specific feature area in mind.
