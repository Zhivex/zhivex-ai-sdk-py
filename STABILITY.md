# Stability

Zhivex AI SDK uses three stability levels so production integrators can understand which surfaces are intended to remain predictable over time.

Supported public imports should come from `zhivex_ai`. Deep imports from internal modules are not part of the stable contract unless this document names an explicit exception.

The machine-readable stability contract lives in `src/zhivex_ai/api_stability.py`. That manifest classifies every symbol exported through `zhivex_ai.__all__` as `stable`, `beta`, or `experimental`; tests fail when the public export list changes without a matching manifest and documentation update.

Related documents:

- [README.md](./README.md)
- [VERSIONING.md](./VERSIONING.md)
- [CHANGELOG.md](./CHANGELOG.md)
- [docs/PARITY_MATRIX.md](./docs/PARITY_MATRIX.md)
- [SECURITY.md](./SECURITY.md)
- [docs/OPERATIONS.md](./docs/OPERATIONS.md)
- [docs/EVALUATIONS.md](./docs/EVALUATIONS.md)
- [docs/PROTOCOLS.md](./docs/PROTOCOLS.md)

## Stable

These APIs are the supported public contract for application code and production integrations:

- Provider factories: `create_openai`, `create_anthropic`, `create_azure_openai`, `create_gemini`, `create_vertex`, `create_qwen`, `create_kimi`, `create_deepseek`, `create_vllm`
- Text generation: `generate_text`, `stream_text`
- Structured output: `generate_object`, `stream_object`
- Grounded text: `generate_grounded_text`
- Embeddings: `embed`, `embed_many`, `embed_content`, `embed_content_many`
- Agent runtime: `Agent`, `AgentContext`, `AgentSession`, `AgentRuntime`, `AgentRegistry`, `AgentRunRequest`, `AgentRunResult`, `AgentStreamResult`, `AgentHandoff`, `run_agent`, `stream_agent`, `resume_agent`, `resume_agent_run`, `handoff_to`, `create_agent_session`, `load_agent_session`
- Agent extension contracts: `AgentHooks`, `AgentMiddleware`, `AgentMiddlewareNext`, `AgentObserver`, `DynamicInstructions`
- Agent tools: `ToolRegistry`, `ToolDefinition`, `ToolSet`, `ToolExecutionContext`, `ToolExecutionOptions`, `ToolExecutionResult`, `ToolExecutionError`, `tool`
- Agent run state and replay: `AgentRunStore`, `AgentRunState`, `AgentRunStatus`, `AgentRunStep`, `AgentChildRun`, `PostgresAgentRunStore`, `create_postgres_agent_run_store`, `serialize_agent_run_state`, `deserialize_agent_run_state`, `agent_run_state_to_json`, `agent_run_state_from_json`, `cancel_agent_run`, `cancel_agent_run_tree`, `AgentRunTreeCancellationResult`, `AgentRunSnapshot`, `create_agent_run_snapshot`, `AgentReplayEvent`, `AgentReplayResult`, `replay_agent_run`
- Durable agent approvals: `ApprovalDecision`, `ToolApprovalRequest`, `AgentToolApprovalEvent`, `PendingApproval`, `get_pending_agent_approvals`
- Agent skills: `skill`, `load_skill`, `discover_skills`, `SkillDefinition`, `SkillDependency`, `SkillRegistry`
- Agent skill session controls: `set_agent_session_skills`, `get_agent_session_skills`, `clear_agent_session_skills`
- Agent skill observability: `AgentSkillActivatedEvent`, `AgentSkillSkippedEvent`
- Agent persistence: `create_postgres_agent_memory_store`, `create_postgres_checkpoint_store`
- MCP helpers and registries: `discover_mcp_tools`, `mcp_stdio_server`, `mcp_http_server`, `create_mcp_tool_registry`
- Gateway: `GatewayAttempt`, `GatewayConfig`, `GatewayError`, `GatewayImageAttachment`, `GatewayMessage`, `GatewayModelTarget`, `GatewayObjectResponse`, `GatewayResponse`, `create_gateway`
- Core errors: `AgentEventDeliveryError`, `AgentRunCancelled`, `ProviderHTTPError`, `ToolExecutionOutcomeUnknown`, `ConfigurationError`, `ValidationError`, `UnsupportedFeatureError`
- HTTP and SSE helpers: `HTTPResponse`, `stream_sse`, `to_sse_response`, `to_sse_stream`, `to_text_stream`, `to_text_stream_response`, `to_ui_message_stream_response`

The stable surface is intentionally narrow. It reflects the most defendable cross-provider experience and the main API-building primitives in this SDK today.

## Beta

These APIs are supported and documented, but they may still change between minor releases as the SDK matures:

- Middleware helpers
- Model catalog helpers
- Provider agent capability metadata: `AgentCapabilities`, `AgentSupportTier`, `get_agent_capabilities`, `get_agent_support_tier`
- First-class hosted tool model: `HostedToolDefinition`, `HostedToolClass`, `AnyToolDefinition`, `hosted_tool`, `is_hosted_tool_definition`, `is_callable_tool_definition`, `get_hosted_tool_class`, `is_hosted_tool_class`; provider-native helpers such as `anthropic_web_fetch_tool()` remain beta
- Provider-data content parts and hosted-tool control payloads: `ProviderDataPart`, `provider_data_part`, `get_provider_data_parts`, `get_last_provider_data_part`, `openai_mcp_approval_response`, `azure_openai_mcp_approval_response`
- Typed OpenAI/Azure provider-data payloads and parsers: `OpenAIResponseReference`, `OpenAIMcpApprovalRequest`, `OpenAIMcpApprovalResponse`, `OpenAIMcpCall`, `OpenAIMcpListTools`, `OpenAIProviderData`, `AzureOpenAIResponseReference`, `AzureOpenAIMcpApprovalRequest`, `AzureOpenAIMcpApprovalResponse`, `AzureOpenAIMcpCall`, `AzureOpenAIMcpListTools`, `AzureOpenAIProviderData`, `parse_openai_provider_data_part`, `parse_azure_openai_provider_data_part`
- Response-reference helpers: `openai_response_reference`, `get_openai_response_reference`, `get_openai_response_id`, `azure_openai_response_reference`, `get_azure_openai_response_reference`, `get_azure_openai_response_id`
- Hosted-tool and approval streaming transport: `StreamProviderDataEvent`, `UIMessageProviderDataChunk`, `UIMessageToolApprovalChunk`; OpenAI Programmatic Tool Calling is beta through `openai_programmatic_tool_calling_tool()` and provider-data replay of `program` / `program_output` items
- Packaged skill APIs and installers: `load_skill_package`, `validate_skill`, `install_skill`, `list_installed_skills`, `run_skill`, `publish_skill`
- Packaged skill types and artifacts: `SkillArtifact`, `SkillEntrypoint`, `SkillPermissions`, `SkillPackageManifest`, `InstalledSkill`, `SkillRegistryIndex`, `SkillRunResult`
- Packaged skill runtime events: `AgentSkillResolvedEvent`, `AgentSkillDependencyCheckEvent`, `AgentSkillExecutionStartEvent`, `AgentSkillExecutionFinishEvent`, `AgentSkillArtifactCreatedEvent`
- Google native media and job clients: `ImagesClient`, `VideosClient`, `MediaClient`, `BatchesClient`, `InteractionsClient`, `CachedContentsClient`, `CachedContent`, `CachedContentListResult`, `ProviderImage`, `GeneratedMedia`, `MediaResult`, `VideoOperation`, and `VideoResult`
- Qwen native hosted-tool, Files, Batch, ASR, and TTS helpers exposed through `provider.native`, `provider.responses()`, `provider.files()`, and `provider.batches()`
- Kimi/Moonshot native helpers: `KimiFormulaClient`, `kimi_formula_toolset`, `KIMI_OFFICIAL_TOOL_URIS`, and `provider.formulas()`
- Multimodal embedding content aliases: `EmbeddingContent` and `EmbeddingPart`
- Agent platform helpers beyond the stable runtime/run-state/replay/approval surface: in-memory and SQLite run stores, native subagent tools such as `create_subagent_tool`, evaluation fixtures/reports, trace artifacts, run-tree snapshots, safety policies, redaction policies, and budget guards
- Evaluation experiments and gates: `AgentEvaluationMetric`, `AgentEvaluationGate`, `AgentEvaluationVariant`, `AgentEvaluationVariantResult`, `AgentEvaluationGateResult`, `AgentEvaluationExperimentResult`, `AgentEvaluationScorer`, `AgentEvaluationAgentFactory`, and `run_agent_evaluation_experiment`
- Agent protocols and hosting: `A2A_PROTOCOL_VERSION`, `A2AAgentSkill`, `A2AAgentCard`, `A2AAgentExecutor`, `AGUIEvent`, `AgentResolver`, `ResponsesAgentHost`, `create_a2a_agent_card`, `create_a2a_app`, `stream_agent_ag_ui`, `to_ag_ui_sse_response`, `create_responses_app`, and `create_agent_playground_app`
- General `zhivex` CLI commands for inspect, run, eval, protocol serve, and the local playground
- Declarative workflow agents: `SequentialAgent`, `ParallelAgent`, `LoopAgent`, `WorkflowAgent`, `WorkflowStep`, `WorkflowRunResult`, `WorkflowStepResult`, `WorkflowTraceEvent`, `WorkflowState`, `WorkflowErrorPolicy`, `WorkflowRunStatus`, `WorkflowStepStatus`, `WorkflowStopCondition`, `run_workflow`, `workflow_step`, shared `session.state`, and workflow expectation helpers
- Durable workflow graphs: `WorkflowBuilder`, `WorkflowGraph`, `GraphWorkflow`, `WorkflowEdge`, `WorkflowEdgeCondition`, `WorkflowContext`, `WorkflowInterruptPhase`, `WorkflowFunctionContext`, `WorkflowFunctionResult`, `WorkflowFunctionExecutor`, `resume_workflow`, `fork_workflow`, `WorkflowRetryPolicy`, and `WorkflowRetryPredicate`
- Workflow durable state: `WORKFLOW_CHECKPOINT_SCHEMA_VERSION`, `WorkflowCheckpoint`, `WorkflowCheckpointStatus`, `WorkflowNodeCheckpoint`, `WorkflowNodeStatus`, `WorkflowInterrupt`, `WorkflowTransition`, `WorkflowCheckpointStore`, `serialize_workflow_checkpoint`, `deserialize_workflow_checkpoint`, `workflow_checkpoint_to_json`, `workflow_checkpoint_from_json`, `InMemoryWorkflowCheckpointStore`, `SQLiteWorkflowCheckpointStore`, `PostgresWorkflowCheckpointStore`, and their factories
- External workflow runtime contracts: `WORKFLOW_ADAPTER_SCHEMA_VERSION`, `WorkflowStepRequest`, `WorkflowStepOutcome`, `WorkflowStepExecutor`, `WorkflowStepExecutorRegistry`, `CallbackWorkflowAdapter`, `WorkflowAdapter`, `WorkflowAdapterCapabilities`, and callback-adapter factories for DBOS, Temporal, Prefect, and Restate. These factories are contracts for application-owned integrations, not certified engine integrations.

Workflow semantics and operational boundaries are documented in [docs/WORKFLOWS.md](./docs/WORKFLOWS.md). SQLite and Postgres workflow checkpoint stores are durable storage implementations, but their workflow APIs remain beta and are not promoted by the stable Postgres agent-store guarantee.

Beta APIs still require changelog coverage when they change, but they do not carry the same compatibility guarantees as the stable surface.

The README support matrix is generated from runtime metadata. Its `Agent Capabilities` section is useful product guidance for hosted tools and provider-managed events, but it should be read with the same beta expectations as the APIs listed above.

Agent production guidance lives in [docs/AGENTS.md](./docs/AGENTS.md), [docs/PRODUCTION.md](./docs/PRODUCTION.md), [docs/OPERATIONS.md](./docs/OPERATIONS.md), [docs/OBSERVABILITY.md](./docs/OBSERVABILITY.md), and [SECURITY.md](./SECURITY.md).

## Experimental

These areas are available for evaluation, but they should not be treated as a long-term compatibility contract yet:

- Realtime and live voice flows, including `stream_live_agent()`
- Raw provider payload escape hatches that do not map cleanly to the hosted-tool beta surface
- Provider areas currently marked as `native-only` or `compatibility` in the support matrix

Experimental areas may change faster than the rest of the SDK. Production adopters should isolate usage behind their own service layer before depending on them.

## Deprecation pattern

Stable APIs should not be removed or reclassified silently. When a stable API needs to change, first document the replacement path in this file and [VERSIONING.md](./VERSIONING.md), add a changelog entry with migration guidance, and keep the old top-level export available until a planned breaking release.

## Provider scope

The current tier-1 provider story for the stable surface is:

- OpenAI
- Anthropic
- Azure OpenAI
- Gemini
- Vertex
- Qwen
- Kimi/Moonshot
- DeepSeek
- vLLM

In this repository, tier-1 means the provider is part of the stable surface story, production API guidance, support-matrix contract checks, shared offline provider contract tests, and documented optional live smoke setup.

Anthropic is included in the tier-1 set for text-generation API paths. The stable factory supports direct `claude-opus-5` Messages calls with model-specific adaptive-thinking, effort, tool-loop replay, refusal, and mid-conversation system-section validation. Hosted web search/code execution and raw server-side fallback remain provider-native beta behavior; Opus 5 Web Fetch, Priority Tier, assistant prefill, and Opus 5 through the current Bedrock Converse adapter are not claimed. Embeddings, transcription, and speech remain outside the current Anthropic provider surface.

Azure OpenAI is tier-1 for the portable production surface. `create_azure_openai(...)` supports either API key authentication or Microsoft Entra ID token/provider authentication. Its native Responses, Conversations, and File Search store lifecycle clients are beta provider-specific surfaces exposed through `provider.native` / the bundle helper methods, not additions to the stable portable contract.

Qwen is tier-1 for portable text generation, streaming, structured output, callable tools, and embeddings through the current `/compatible-mode/v1` route. Its hosted tools, raw Responses settings, Files, region-dependent Batch behavior, ASR, and TTS surfaces remain beta provider-specific paths exposed through `provider.native` / bundle helper methods.

Kimi/Moonshot is tier-1 for portable text generation, streaming, structured output, and callable tools through Chat Completions. K3 `reasoning_effort`, K2 thinking controls, Files, Batch, token estimation, and Formulas remain beta provider-specific paths, and this SDK does not claim Kimi embeddings, speech, or transcription.

DeepSeek is tier-1 for portable text generation, streaming, JSON structured output, callable tools, and reasoning through its Chat Completions API. The stable factory targets the current `deepseek-v4-flash` and `deepseek-v4-pro` model contract, preserves reasoning state across tool loops, and rejects retired model IDs or incompatible thinking options before dispatch. Strict-tool and prefix beta routing plus raw `provider_options` remain provider-specific beta/experimental behavior; vision, files, embeddings, audio, moderation, and hosted tools are not claimed.

vLLM is included in the tier-1 set for the SDK primitives backed by its OpenAI-compatible server: text generation, streaming, structured output/tools, embeddings, transcription, and realtime ASR. The guarantee is model/task-dependent: vLLM must be serving compatible generation, embedding, or ASR models for those surfaces to work, and vLLM custom endpoints such as tokenize, rerank, classify, and score are outside the stable SDK surface.

Other providers remain useful, but they should be evaluated with the support matrix and the stability level of the specific feature area in mind.
