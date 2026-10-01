# Qwen native text reranking (Beta)

Use the Stable `create_qwen` factory and Beta `provider.native.rerank()` client
for `qwen3.7-text-rerank` or the earlier `qwen3-rerank`. These are ranking APIs,
not language generation or embedding models. No portable ranking contract or
live release certification is implied.

```python
from zhivex_ai import create_qwen

provider = create_qwen(region="cn", workspace_id="your-workspace")
documents = ["The weather is sunny.", "Reset passwords in account settings."]
response = await provider.native.rerank().create(
    query="How can I reset my password?", documents=documents,
    model="qwen3.7-text-rerank", top_n=1,
    instruct="Given a web search query, retrieve relevant passages that answer the query.",
)
for item in response["output"]["results"]:
    print(item["relevance_score"], documents[item["index"]])
```

`QWEN_API_KEY` or `DASHSCOPE_API_KEY` supplies the account credential.
`DASHSCOPE_WORKSPACE_ID` supplies the optional workspace when no explicit
`workspace_id` is given. Workspace routing uses `cn-beijing` for `cn`,
`ap-southeast-1` for `intl`, and `us-east-1` for `us`; the key must belong to that
workspace and region. The reviewed pricing lists the new 3.7 model only in Beijing, and
`qwen3-rerank` in Beijing and Singapore. The client rejects the new model on
official Singapore and US endpoints before sending a request; explicit custom
gateways remain application-owned routes. The example selects Beijing explicitly.

Without a workspace, the factory's regional DashScope host is used. With an
official DashScope host and a workspace, the request goes to
`https://<workspace>.<region>.maas.aliyuncs.com`. An explicit custom `base_url`
keeps its host and any gateway prefix; credentials are sent to that gateway.
Recognized `/compatible-mode/v1`, `/compatible-api/v1`, and `/api/v1` suffixes
are replaced by the model's native path.

The client sends `qwen3.7-text-rerank` to
`/api/v1/services/rerank/text-rerank/text-rerank` with nested `input` and
`parameters`; `qwen3-rerank` uses `/compatible-api/v1/reranks` with flat fields.
Responses remain native: the latest model has `output.results`,
`usage.prompt_tokens`, `usage.total_tokens`, and `request_id`; the earlier model
has top-level `results`, `usage.total_tokens`, and `id`. Additional provider
fields are preserved. Each result's `index` refers to the input document list.
Scores are only relative within a request.

The client validates non-empty query/text documents, 1–500 documents, positive
`top_n`, and an optional non-empty English `instruct`. `top_n` may exceed the
number of candidates; the provider returns all candidates. The 3.7 reference
lists 30,000 tokens per item and recommends a 120,000-token request budget;
the provider performs token-limit validation, and the SDK never truncates text.
The SDK checks result indices and finite relevance scores in [0, 1]. HTTP errors
and provider error envelopes raise `ProviderHTTPError`; malformed results raise
`ParseError`. Requests are sent once; automatic retries are not applied.

Multimodal `qwen3-vl-rerank`, GTE models, and `return_documents` are outside this
text-only client's contract. Select document text by the returned index.

Run `python examples/retrieval/qwen_rerank.py` with exported credentials and
workspace ID. Offline verification: `python -m pytest tests/test_qwen_rerank.py`.

Sources reviewed September 30, 2026:

- [Text rerank API](https://help.aliyun.com/en/model-studio/text-rerank-api)
- [Regional prices](https://help.aliyun.com/zh/model-studio/model-pricing)
