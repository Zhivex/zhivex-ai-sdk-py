from zhivex_ai import (
    AgentCheckpoint,
    GatewayModelTarget,
    ProviderBundle,
    WorkflowGraph,
    create_meta,
    create_openai,
    create_qwen,
)


openai_provider: ProviderBundle = create_openai(api_key="test")
qwen_provider: ProviderBundle = create_qwen(api_key="test")
meta_provider: ProviderBundle = create_meta(api_key="test")
checkpoint_type: type[AgentCheckpoint] = AgentCheckpoint
workflow_graph_type: type[WorkflowGraph] = WorkflowGraph
vllm_gateway_target = GatewayModelTarget(provider="vllm", model_id="meta-models/Muse-Glimmer-30B")
meta_gateway_target = GatewayModelTarget(provider="meta", model_id="muse-spark-1.2")
