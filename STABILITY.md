# Stability

Zhivex AI SDK uses three stability levels so production integrators can understand which surfaces are intended to remain predictable over time.

The `zhivex_ai` package root remains the compatibility entrypoint for the existing public surface. New application code should use the root for the Stable agent-first core and the focused namespaces documented below for extension areas. Deep imports from implementation modules are not part of the public contract unless this document names an explicit exception.

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
- [docs/SCOPE.md](./docs/SCOPE.md)

## Import Boundaries

The recommended public import paths are:

- `zhivex_ai` for the Stable portable foundation, agent runtime, provider factories, gateway, and backend transport contracts
- `zhivex_ai.evals` for Beta evaluation APIs
- `zhivex_ai.workflows` for the Stable declarative and durable workflow core plus clearly identified Beta named-engine factories
- `zhivex_ai.integrations.protocols` for Beta A2A, AG-UI, and Responses-compatible hosting
- `zhivex_ai.experimental` for Experimental realtime/live-agent and non-portable provider surfaces

Existing top-level Beta and Experimental imports remain available for compatibility in the current Beta release line. A focused namespace changes the recommended ownership boundary; it does not promote the symbols inside it. New Beta or Experimental features should land in a focused namespace and should not expand the package root by default.

## Stable

These APIs are the supported public contract for application code and production integrations:

- Provider factories: `create_openai`, `create_anthropic`, `create_azure_openai`, `create_gemini`, `create_vertex`, `create_qwen`, `create_kimi`, `create_deepseek`, `create_meta`, `create_vllm`
- HTTP lifecycle: `HTTPTransport`, `aclose_default_clients`
- Text generation: `generate_text`, `stream_text`
- Structured output: `generate_object`, `stream_object`
- Grounded text: `generate_grounded_text`
- Embeddings: `embed`, `embed_many`, `embed_content`, `embed_content_many`
- Portable foundation contracts: `LanguageModel`, `GroundedLanguageModel`, `EmbeddingModel`, `ModelMessage`, `GenerateTextOutput`, `GenerateObjectOutput`, `GenerateGroundedTextOutput`, `StreamTextResult`, `StreamObjectResult`, `StreamEvent`, `EmbedOutput`, `EmbeddingContent`, `TokenUsage`, `FinishReason`, `JsonValue`
- Agent runtime: `Agent`, `AgentContext`, `AgentSession`, `AgentRuntime`, `AgentRegistry`, `AgentRunRequest`, `AgentRunResult`, `AgentStreamResult`, `AgentHandoff`, `run_agent`, `stream_agent`, `resume_agent`, `resume_agent_run`, `handoff_to`, `create_agent_session`, `load_agent_session`
- Agent result contracts: `AgentTrace`, `AgentCheckpoint`, `ToolCall`
- Agent extension contracts: `AgentHooks`, `AgentMiddleware`, `AgentMiddlewareNext`, `AgentObserver`, `DynamicInstructions`
- Agent tools: `ToolRegistry`, `ToolDefinition`, `ToolSet`, `ToolExecutionContext`, `ToolExecutionOptions`, `ToolExecutionResult`, `ToolExecutionError`, `tool`
- Agent run state and replay: `AgentRunStore`, `AgentRunState`, `AgentRunStatus`, `AgentRunStep`, `AgentChildRun`, `PostgresAgentRunStore`, `create_postgres_agent_run_store`, `serialize_agent_run_state`, `deserialize_agent_run_state`, `agent_run_state_to_json`, `agent_run_state_from_json`, `cancel_agent_run`, `cancel_agent_run_tree`, `AgentRunTreeCancellationResult`, `AgentRunSnapshot`, `create_agent_run_snapshot`, `AgentReplayEvent`, `AgentReplayResult`, `replay_agent_run`
- Durable agent approvals: `ApprovalDecision`, `ToolApprovalRequest`, `AgentToolApprovalEvent`, `PendingApproval`, `get_pending_agent_approvals`
- Agent skills: `skill`, `load_skill`, `discover_skills`, `SkillDefinition`, `SkillDependency`, `SkillRegistry`
- Agent skill session controls: `set_agent_session_skills`, `get_agent_session_skills`, `clear_agent_session_skills`
- Agent skill observability: `AgentSkillActivatedEvent`, `AgentSkillSkippedEvent`
- Agent persistence: `create_postgres_agent_memory_store`, `create_postgres_checkpoint_store`
- MCP helpers and registries: `discover_mcp_tools`, `mcp_stdio_server`, `mcp_http_server`, `create_mcp_tool_registry`
- Gateway: `GatewayAttempt`, `GatewayConfig`, `GatewayError`, `GatewayImageAttachment`, `GatewayMessage`, `GatewayModelTarget`, `GatewayObjectResponse`, `GatewayResponse`, `create_gateway`. Additive cost controls resolve provider/model overrides before catalog prices and the deprecated provider-wide fallback. A configured ceiling fails closed on unknown pricing, while no ceiling preserves route eligibility. When the Stable `ModelCatalog` is explicitly configured, cataloged fallback scoring uses typed recommendations instead of model-name substrings, required capabilities fail closed on absent catalog metadata, and `route_decision.target_evidence` records the metadata used. Uncataloged targets retain the legacy scoring path for compatibility. `GatewayAttempt.reason` is the Stable machine-readable policy-skip/refusal field and `GatewayAttempt.error_type` is its additive typed-error companion. `GatewayConfig.on_attempt` emits one terminal payload per executed retry or skipped target; provider latency excludes observer time, and observer failures are non-authoritative.
- Declarative workflows: `SequentialAgent`, `ParallelAgent`, `LoopAgent`, `WorkflowAgent`, `WorkflowStep`, `WorkflowRunResult`, `WorkflowStepResult`, `WorkflowTraceEvent`, `WorkflowState`, `WorkflowErrorPolicy`, `WorkflowRunStatus`, `WorkflowStepStatus`, `WorkflowStopCondition`, `WorkflowRetryPolicy`, `WorkflowRetryPredicate`, `WorkflowFunctionContext`, `WorkflowFunctionResult`, `WorkflowFunctionExecutor`, `run_workflow`, `workflow_step`, and `validate_workflow_expectations`
- Durable workflow graphs: `WorkflowBuilder`, `WorkflowGraph`, `GraphWorkflow`, `WorkflowEdge`, `WorkflowEdgeCondition`, `WorkflowContext`, `WorkflowInterruptPhase`, `resume_workflow`, `fork_workflow`, and `cancel_workflow`
- Workflow durable state: `WORKFLOW_CHECKPOINT_SCHEMA_VERSION`, `WorkflowCheckpoint`, `WorkflowCheckpointMigration`, `WorkflowCheckpointStatus`, `WorkflowNodeCheckpoint`, `WorkflowNodeStatus`, `WorkflowInterrupt`, `WorkflowTransition`, `WorkflowCheckpointStore`, `serialize_workflow_checkpoint`, `deserialize_workflow_checkpoint`, `workflow_checkpoint_to_json`, `workflow_checkpoint_from_json`, `migrate_workflow_checkpoint`, `migrate_workflow_checkpoint_payload`, and `migrate_workflow_run_checkpoint`
- Workflow stores and ownership: `InMemoryWorkflowCheckpointStore`, `SQLiteWorkflowCheckpointStore`, `PostgresWorkflowCheckpointStore`, `create_in_memory_workflow_checkpoint_store`, `create_sqlite_workflow_checkpoint_store`, `create_postgres_workflow_checkpoint_store`, `WorkflowExecutionLease`, `WorkflowLeaseManager`, `InMemoryWorkflowLeaseManager`, `SQLiteWorkflowLeaseManager`, `PostgresWorkflowLeaseManager`, `create_in_memory_workflow_lease_manager`, `create_sqlite_workflow_lease_manager`, and `create_postgres_workflow_lease_manager`
- Workflow adapter envelope: `WORKFLOW_ADAPTER_SCHEMA_VERSION`, `WorkflowStepRequest`, `WorkflowStepOutcome`, `WorkflowStepExecutor`, `WorkflowStepExecutorRegistry`, `CallbackWorkflowAdapter`, `WorkflowAdapter`, and `WorkflowAdapterCapabilities`
- Workflow failures: `WorkflowConflictError`, `WorkflowLeaseLostError`, `WorkflowDefinitionMismatchError`, `WorkflowRunNotFoundError`, and `WorkflowInterruptError`
- Core errors: `AgentEventDeliveryError`, `AgentRunCancelled`, `ProviderHTTPError`, `ToolExecutionOutcomeUnknown`, `ConfigurationError`, `ValidationError`, `UnsupportedFeatureError`
- HTTP and SSE helpers: `HTTPResponse`, `stream_sse`, `to_sse_response`, `to_sse_stream`, `to_text_stream`, `to_text_stream_response`, `to_ui_message_stream_response`

The Stable surface is the agent-first core contract plus the application-owned workflow orchestration primitives required to coordinate it durably. Stable workflows do not turn external effects into distributed transactions: destinations still need idempotency, fencing, an outbox, or reconciliation.

- Local agent persistence: `AgentMemoryState`, `SummaryConfig`, `InMemoryAgentRunStore`, `SQLiteAgentRunStore`, `create_in_memory_agent_run_store`, `create_sqlite_agent_run_store`, `create_in_memory_agent_memory_store`, `create_sqlite_agent_memory_store`, `create_in_memory_checkpoint_store`, `create_sqlite_checkpoint_store`. InMemory is process-local storage for tests and demos; SQLite is durable local storage for a single host. See [local persistence guarantees](./docs/agents/durable-state.md#local-storage-guarantees).
- Application-owned model catalogs: `ModelCatalog`, `ModelCatalogEntry`, `create_model_catalog`, `ModelCapabilities`, `AgentCapabilities`, `AgentSupportTier`, `CatalogProviderId`, `ModelApiSurface`, `ModelAvailability`, `ModelSupportEvidence`, `RecommendedUse`, `ModelPricing`. Stable covers construction, lookup, defensive copies, metadata types, and price units/validity windows, not the truth or freshness of provider metadata.

## Beta

These APIs are supported and documented, but they may still change between minor releases as the SDK matures:

- Meta Model API native extensions: `meta_hosted_tool`, `meta_web_search_tool`, `meta_tool_search_tool`, Files, raw Responses, hosted tools, Contributor models, and multimodal/native extras. These remain Beta even though `create_meta` and the Standard `muse-spark-1.2` portable text/tool/application-supplied retrieval contract are Stable.
- Middleware helpers. `create_file_generate_cache(...)` preserves its JSON format but now publishes complete entries with same-directory atomic replacement. Corrupt, truncated, symlinked, non-regular, or incompatible entries raise `ValidationError`; they are not cache misses. This Beta guarantee is limited to application-owned local filesystems with atomic `os.replace(...)` semantics.
- Maintained catalog snapshot: `default_model_catalog` remains Beta; its entries, recommendations, lifecycle, sources, capabilities, and prices may change. Construction rejects identifier collisions, catalog reads are defensive, recommendations do not imply capabilities, and the default snapshot does not constitute live certification or automatic provider discovery.
- Provider agent capability metadata: `get_agent_capabilities`, `get_agent_support_tier`; the returned capability types are Stable, while provider discovery metadata remains Beta
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
- Multimodal embedding part alias: `EmbeddingPart`
- Example and deterministic-test fixture contracts: `GenerateResult`, `ModelGenerateInput`, `JsonValue`
- Agent platform helpers beyond the stable runtime/run-state/replay/approval surface: native subagent tools such as `create_subagent_tool`, evaluation fixtures/reports, trace artifacts, run-tree snapshots, safety policies, redaction policies, and budget guards
- Cooperative agent cancellation and tool policy extensions: `AgentCancellationToken`, `ToolGuardrailResult`, `ToolGuardrailStage`, `ToolGuardrailTripwireTriggered`, `ToolInputGuardrail`, `ToolInputGuardrailRequest`, `ToolOutputGuardrail`, and `ToolOutputGuardrailRequest`
- Evaluation trials, experiments, and gates: `AGENT_EVALUATION_ARTIFACT_SCHEMA_VERSION`, `AgentEvaluationTrialResult`, `AgentEvaluationTrajectory`, `AgentEvaluationTrajectoryEvent`, `AgentEvaluationCostEstimator`, trace extractor aliases, `create_agent_evaluation_trajectory`, `create_agent_evaluation_dataset_from_traces`, `AgentEvaluationMetric`, `AgentEvaluationGate`, `AgentEvaluationVariant`, `AgentEvaluationVariantResult`, `AgentEvaluationGateResult`, `AgentEvaluationExperimentResult`, `AgentEvaluationScorer`, `AgentEvaluationAgentFactory`, and `run_agent_evaluation_experiment`
- Agent protocols and hosting: `A2A_PROTOCOL_VERSION`, `A2AAgentSkill`, `A2AAgentCard`, `A2AAgentExecutor`, `AGUIEvent`, `HostedAgentRunOptions`, `ProtocolInvocation`, `ProtocolLimits`, `ProtocolRunOptionsResolver`, `ProtocolErrorMapper`, `ProtocolEventCallback`, `AgentResolver`, `ResponsesAgentHost`, `StoredResponsesRun`, `ResponsesEventStore`, `InMemoryResponsesEventStore`, `create_a2a_agent_card`, `create_a2a_app`, `stream_agent_ag_ui`, `to_ag_ui_sse_response`, `create_responses_app`, and `create_agent_playground_app`
- General `zhivex` CLI commands for inspect, run, eval, protocol serve, and the local playground
- Named external workflow-engine factories: `create_dbos_workflow_adapter`, `create_temporal_workflow_adapter`, `create_prefect_workflow_adapter`, and `create_restate_workflow_adapter`. These dependency-free factories only label the Stable callback envelope with conservative engine capability metadata; they are not certified DBOS, Temporal, Prefect, or Restate clients, workers, schedulers, or integrations.

Workflow semantics, checkpoint migration, and operational boundaries are documented in [docs/WORKFLOWS.md](./docs/WORKFLOWS.md). Stable classification covers the SDK contracts and built-in storage behavior; deployment-specific database, proxy, retention, authorization, and external-engine validation remain application responsibilities.

Beta APIs still require changelog coverage when they change, but they do not carry the same compatibility guarantees as the Stable surface. Prefer their focused namespaces in new code so the compatibility risk is visible at the import site.

The README support matrix combines runtime metadata with the versioned provider-certification policy. Its `Agent Capabilities` section is useful product guidance for hosted tools and provider-managed events, but it should be read with the same beta expectations as the APIs listed above. A `release-certified` badge is operational evidence for one exact provider target, model, operation set, wheel, commit, and workflow; it does not promote an API or a Beta provider surface to Stable.

Agent production guidance lives in [docs/AGENTS.md](./docs/AGENTS.md), [docs/PRODUCTION.md](./docs/PRODUCTION.md), [docs/OPERATIONS.md](./docs/OPERATIONS.md), [docs/OBSERVABILITY.md](./docs/OBSERVABILITY.md), and [SECURITY.md](./SECURITY.md).

## Experimental

These areas are available for evaluation, but they should not be treated as a long-term compatibility contract yet:

- Realtime and live voice flows, including `stream_live_agent()`
- Raw provider payload escape hatches that do not map cleanly to the hosted-tool beta surface
- Non-portable provider factories currently marked as `native-only` or `compatibility` in the support matrix: `create_bedrock`, `create_openrouter`, and `create_ollama`

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
- Meta Model API
- vLLM

In this repository, tier-1 means the provider is part of the stable surface story, production API guidance, support-matrix contract checks, shared offline provider contract tests, and documented optional live smoke setup. It is a contract-level support classification, not evidence that every tier-1 provider was live-smoked for the current release SHA. Live certification is provider-, model-, operation-, and SHA-specific and should be claimed only when matching smoke evidence was recorded.

Anthropic is included in the tier-1 set for text-generation API paths. The stable factory supports direct `claude-opus-5` Messages calls with model-specific adaptive-thinking, effort, tool-loop replay, refusal, and mid-conversation system-section validation. Hosted web search/code execution and raw server-side fallback remain provider-native beta behavior; Opus 5 Web Fetch, Priority Tier, assistant prefill, and Opus 5 through the current Bedrock Converse adapter are not claimed. Embeddings, transcription, and speech remain outside the current Anthropic provider surface.

Azure OpenAI is tier-1 for the portable production surface. `create_azure_openai(...)` supports either API key authentication or Microsoft Entra ID token/provider authentication. Its native Responses, Conversations, and File Search store lifecycle clients are beta provider-specific surfaces exposed through `provider.native` / the bundle helper methods, not additions to the stable portable contract.

Qwen is tier-1 for portable text generation, streaming, structured output, callable tools, and embeddings through the current `/compatible-mode/v1` route. GA `qwen3.8-max` uses Responses for text, image, reasoning, callable/hosted tools, and streaming, with operation-aware Chat Completions fallback for native JSON Schema output, image/video `FilePart` input, and reasoning budgets. That routing is an implementation detail of the stable `create_qwen(...)` factory; its raw Responses settings, hosted-tool helpers, Files, region-dependent Batch behavior, ASR, and TTS surfaces remain beta provider-specific paths exposed through `provider.native` / bundle helper methods.

Kimi/Moonshot is tier-1 for portable text generation, streaming, structured output, and callable tools through Chat Completions. K3 `reasoning_effort`, K2 thinking controls, Files, Batch, token estimation, and Formulas remain beta provider-specific paths, and this SDK does not claim Kimi embeddings, speech, or transcription.

DeepSeek is tier-1 for portable text generation, streaming, JSON structured output, callable tools, and reasoning through its Chat Completions API. The stable factory targets the current `deepseek-v4-flash` and `deepseek-v4-pro` model contract, preserves reasoning state across tool loops, and rejects retired model IDs or incompatible thinking options before dispatch. Strict-tool and prefix beta routing plus raw `provider_options` remain provider-specific beta/experimental behavior; vision, files, embeddings, audio, moderation, and hosted tools are not claimed.

Meta Model API is tier-1 for the Stable `create_meta(...)` factory and Standard `muse-spark-1.2` portable text generation, streaming, structured output, callable tools, the resulting agent tool loop, and application-supplied retrieval through `PortableRetrievalConfig`. Portable retrieval injects bounded `PortableDocument` text into the request and does not invoke Meta Files, hosted search, or raw Responses. Meta accepts only `tool_choice="auto"` in this contract. Contributor models, hosted-tool helpers, Files, raw Responses/continuation, hosted tools, and multimodal/native extras remain Beta. Embeddings, speech output, transcription, grounding, Realtime, image generation, and video generation are not claimed. Tier-1 is contract support; release certification still requires matching evidence for the exact model, operations, artifact, and source revision.

vLLM is included in the tier-1 set for the SDK primitives backed by its OpenAI-compatible server: text generation, streaming, structured output/tools, embeddings, transcription, and realtime ASR. The guarantee is model/task-dependent: vLLM must be serving compatible generation, embedding, or ASR models for those surfaces to work, and vLLM custom endpoints such as tokenize, rerank, classify, and score are outside the stable SDK surface.

Other providers remain useful, but they should be evaluated with the support matrix and the stability level of the specific feature area in mind.

## Streaming resource ownership

Built-in `StreamTextResult`, `StreamObjectResult`, and `AgentStreamResult` results expose `aclose()` and async context management. Closing a result cancels and joins its producer; closing a single event iterator only detaches that consumer. The Experimental live-agent result uses the same ownership helper and retains its Experimental classification.

`stream_buffer_size` accepts a positive event count or `None`. Stable defaults preserve full history (`None`). Set a finite limit, such as 4096, for request-owned production streams. Consumers whose cursor has been evicted raise `ValidationError`, including late subscribers requesting unavailable history. There is no silent event loss. `collect()` retains its final-result contract independently of subscriber retention. A limit bounds retained event count, not individual payload bytes or final output size.

DeepSeek's `deepseek-v4-flash-vision-exp` is an upstream Experimental model tracked as `preview` in the catalog. Its tested image path accepts user `ImagePart` URL/base64 inputs only. This does not promote vision to the Stable Tier-1 DeepSeek cohort or add Files, Responses, audio, or other native endpoints.

## Dependency compatibility update

Development and CI use a reviewed uv lock with independent minimum/latest range tests. Realtime remains Experimental and its default websocket transport now requires `zhivex-ai-sdk[realtime]`; core/provider imports remain available without websockets. See [dependency compatibility](./docs/DEPENDENCY_COMPATIBILITY.md) for migration and update commands.
