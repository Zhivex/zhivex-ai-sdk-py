# Production responsibilities

Use Stable public APIs with application-owned operational controls. The overall
package is Beta; evaluate each extension's classification before adoption.

- Authenticate every request and authorize actions against a tenant-owned data partition.
- Keep provider credentials server-side, and apply request, time, concurrency and cost limits.
- Own Postgres backups, retention, migrations, checkpoint recovery and side-effect idempotency.
- Redact prompts, tool arguments, responses and secrets before telemetry export.
- Enable observability explicitly and bound label cardinality and sampling.
- Close streams and HTTP clients on cancellation and service shutdown.

See [production APIs]({{SOURCE}}/PRODUCTION_APIS.md),
[observability]({{SOURCE}}/docs/OBSERVABILITY.md) and
[stability policy]({{SOURCE}}/STABILITY.md) for the documented version.
[Evaluations](reference/evals.md), [protocol hosting](reference/protocols.md) and
[Experimental APIs](reference/experimental.md) remain separately labeled extensions.
