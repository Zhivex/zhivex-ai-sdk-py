# Foundation APIs

Use supported root imports for the Stable portable contract:

```python
from zhivex_ai import generate_text, stream_text, generate_object, stream_object, embed
```

- `generate_text` returns normalized text, usage and finish information.
- `stream_text` supports incremental consumption; close or cancel streams when the consumer exits.
- `generate_object` and `stream_object` bind structured generation to an application-owned schema.
- `embed` produces embeddings for models that declare that capability.

Provider construction selects a model; the shared operation stays the same.
Model capabilities differ, so verify the provider contract before enabling a
feature. A successful local mock establishes runtime behavior, not live support.

See the exact signatures and stability labels in the [root reference](reference/root.md).
