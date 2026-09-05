# Providers and evidence

The Tier-1 portable contract covers OpenAI, Anthropic, Azure OpenAI, Gemini,
Vertex, Qwen, Kimi/Moonshot, DeepSeek, Meta Standard and a configured vLLM deployment.

```python
from zhivex_ai import create_openai, create_anthropic, create_qwen, create_meta
```

`provider(model_id)` uses the SDK-owned portable interface. Provider-native
resources and hosted tools have narrower compatibility guarantees. Meta Standard
and Contributor are separate targets; a hosted model on another API is not a
direct integration with the model's author.

Contract tests, installed-wheel checks and authenticated live certification are
separate evidence levels. Certification is specific to the artifact, model,
operation set and time window. This site does not certify a provider.

Consult the [provider guide]({{SOURCE}}/docs/PROVIDERS.md) and the release's
[support policy]({{SOURCE}}/SUPPORT.md). Their links are pinned to the documented
artifact's source revision.
