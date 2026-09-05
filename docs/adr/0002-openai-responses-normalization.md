# OpenAI-compatible Responses normalization

Status: accepted for implementation (PY-HU-11).

`providers/_openai_responses_normalization.py` owns conversion of Responses output
items, usage, finish reasons and provider-data envelopes into SDK message parts.
It depends only on SDK types, message normalization and the null-field helper.
The OpenAI-compatible provider maintainers own this private module.

```text
openai / azure_openai / qwen adapters
                  |
            openai_compat (HTTP, requests, streaming state)
                  |
       _openai_responses_normalization
                  |
           types / messages / _payload
```

Request serialization reuses the provider-data decoder so MCP and response
references retain exactly the same interpretation in both directions. Transport,
retry behavior, header filtering and error redaction remain in their existing
owners. The extracted module must never import the adapter or perform I/O.

The extraction moves 16 functions without changing their ASTs. Characterization
fixtures were captured from commit `67f56fc4a0a28a453129b51936cb4e91f81591c8`
before moving code: 110 synthetic cases cover OpenAI, Azure and Qwen output shapes,
MCP, hosted tools, image/code/file results, malformed tool JSON and finish states.
Fixtures contain no live prompts, responses or credentials. Tests also verify
input payloads remain unchanged. Public exports match the original checkout.

Full suite: 1008 passed, 16 skipped, 161 subtests. Combined statement coverage of
the original module and its extracted boundary increased from 82.85% to 84.52%.
Mypy explicitly includes the new module; scoped Ruff rules match the adapter.
Future semantic corrections should be reviewed separately from this extraction.
