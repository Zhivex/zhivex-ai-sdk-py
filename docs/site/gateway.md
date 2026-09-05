# Gateway routing

The gateway centralizes model targets, fallback and operation budgets:

```python
from zhivex_ai import GatewayConfig, GatewayModelTarget, create_gateway
```

Configure an explicit target list and cost policy. A strict cost ceiling requires
known pricing; unknown prices must not silently bypass the budget. The catalog is
Beta and affects routing only when explicitly configured. A provider's API host
is separate from the author of its hosted model.

Terminal attempt observations describe actual completion or failure. Keep
observers bounded and redact request data before export. Review the
[gateway guide]({{SOURCE}}/docs/GATEWAY.md) and [reference](reference/root.md)
before configuring retries or fallback in a production service.
