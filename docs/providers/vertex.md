# Google Gemini Enterprise Agent Platform (formerly Vertex AI)

The SDK retains `create_vertex` and provider ID `vertex` for compatibility.
The Google platform's new product name does not change those imports or the
`aiplatform.googleapis.com` service. Google Gemini Developer API remains a
separate provider (`create_gemini`).

## Authentication and endpoints

```python
from zhivex_ai import create_vertex, generate_text

# Express Mode: VERTEX_API_KEY, falling back to GOOGLE_API_KEY.
vertex = create_vertex(express_mode=True)
result = await generate_text(model=vertex("gemini-2.5-flash"), prompt="Hello")

# Standard Google Cloud: install zhivex-ai-sdk[vertex] for ADC.
# Configure gcloud application-default credentials, a service account, or
# workload identity using Google Auth's supported mechanisms.
import google.auth
credentials, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
    quota_project_id="my-project",
)
vertex = create_vertex(credentials=credentials, project_id="my-project", location="global")

# An explicit Google Auth credentials object can also be supplied.
# vertex = create_vertex(credentials=credentials, project_id="my-project")
```

Express Mode uses `https://aiplatform.googleapis.com/v1` without project/location.
Standard global uses the same hostname with `/projects/.../locations/global`.
Regional requests retain `<location>-aiplatform.googleapis.com`.
Jurisdictional locations `us` and `eu` use
`aiplatform.<location>.rep.googleapis.com`; availability must still be checked per model.
Set `express_mode=False` for project-scoped API key requests. An API key without
a project selects Express Mode automatically. An explicit auth argument wins over
environment values; specifying multiple auth arguments is an error. Environment
access tokens take precedence over environment API keys.

ADC is selected when no explicit or environment key/token is available. Credential
discovery happens when constructing the provider; per-request refresh happens in
a worker thread, serialized across concurrent requests. Google quota-project
headers are preserved. Explicit access tokens remain caller-managed. API keys are
sent in headers, never inserted into request URLs. Downloads to other origins do not receive provider credentials, except explicit
Cloud Storage JSON API object reads using bearer credentials. `gs://` video
references are translated to those object reads; the caller needs storage.objects.get.

## Native services

### Endpoint lifecycle and deployment (Beta)

`vertex.native.model_garden()` exposes `create_endpoint`, `update_endpoint`,
`delete_endpoint`, `deploy_model`, `undeploy_model`, `get_operation` and
`wait_operation`, alongside endpoint discovery and inference. Mutation methods
take native Google JSON bodies; endpoint arguments use IDs in the configured
project/region. Creation optionally accepts `endpoint_id`. Updating requires
`update_mask`. Deleting an endpoint never implicitly undeploys its models.

```python
garden = vertex.native.model_garden()
operation = await garden.create_endpoint({"displayName": "my-endpoint"})
# Persist operation["name"] before polling; resume that operation on timeout.
completed = await garden.wait_operation(operation["name"], timeout_ms=120_000)
if completed.get("error"):
    raise RuntimeError("Endpoint creation failed")
endpoint_id = completed["response"]["name"].rsplit("/", 1)[-1]
# deploy_model(endpoint_id, body) requires an existing registered model and
# explicit machine/replica configuration in body["deployedModel"].
# body["trafficSplit"] is sent unchanged; the SDK does not assign traffic.
```

Create/delete/deploy/undeploy return long-running operations; update returns an
Endpoint. Polling returns terminal errors intact and does not resubmit on timeout.
Keep the operation name to resume observation. Deploying can provision billable
compute, so hardware, replicas and traffic routing belong in the caller's payload.
Model registration uses the separate registry methods below; one-click Model
Garden deployment is not implemented by these methods. Tests cover routing, caller-data isolation, error propagation
and timeout without mutation. A subsequent [empty endpoint lifecycle check](../releases/0.27.0-vertex-integration-history.json)
passed creation, reading, updating and deletion with a final 404 in `us-central1`.
Creation resumed the same saved operation after a polling timeout. No model or
GPU was deployed; deployment and undeployment remain contract-tested only.

Sources: Google [create](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/rest/v1/projects.locations.endpoints/create),
[deploy](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/rest/v1/projects.locations.endpoints/deployModel),
[undeploy](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/rest/v1/projects.locations.endpoints/undeployModel)
and [update](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/rest/v1/projects.locations.endpoints/patch).

### Project model registry (Beta)

The same native client exposes `upload_model(body)`, `get_registered_model(id)`,
`list_registered_models(...)`, `update_registered_model(id, body, update_mask=...)`
and `delete_registered_model(id)`. These operate on the configured project's
regional model registry. Existing `get_model(publisher=..., model=...)` and
`list_models(publisher=...)` continue to read the public publisher catalog.

```python
operation = await garden.upload_model({
    "modelId": "my-model",
    "model": {
        "displayName": "My model",
        "artifactUri": "gs://my-artifact-bucket/model/",
        "containerSpec": {"imageUri": "us-docker.pkg.dev/my-project/models/server:release"},
    },
})
# Persist the operation name and poll it; inspect completed["error"] if present.
completed = await garden.wait_operation(operation["name"], timeout_ms=120_000)
```

Upload registers existing cloud artifacts/container configuration; it does not
transfer local files. Native `parentModel`, `serviceAccount` and model version
metadata are forwarded unchanged. Reads accept `id@version` or `id@alias`.
Listings return one page and preserve `nextPageToken`; callers control pagination.
Upload/delete return operations; update returns the Model. Delete does not remove
deployments automatically. Google rejects deleting a model still used by an endpoint.
This is offline contract coverage only: no live upload, update or delete was run.

The [installed-wheel registry discovery](../releases/0.27.0-vertex-integration-history.json)
passed one complete empty page in both `us-central1` and `europe-west4`.
[Endpoint discovery](../releases/0.27.0-vertex-integration-history.json)
also found no endpoints in either region. These checks validate list access only;
get-by-ID was not run without a resource. They do not imply that publisher-hosted
models are unavailable. The discovery runner accepts `--registered-models` and
rejects repeated pagination tokens instead of declaring a complete inventory.

Sources: Google [upload](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/rest/v1/projects.locations.models/upload),
[list](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/rest/v1/projects.locations.models/list),
[update](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/rest/v1/projects.locations.models/patch)
and [delete](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/rest/v1/projects.locations.models/delete).

### Mistral partner models

Google documents Mistral's regional publisher `mistralai` with `rawPredict` and
`streamRawPredict`, separately from Gemini `generateContent` and Gemma's Chat
Completions endpoint. Use the existing native client with a supported region:

```python
mistral_vertex = create_vertex(
    credentials=credentials, project_id="my-project", location="us-central1",
)
response = await mistral_vertex.native.model_garden().raw_predict(
    publisher="mistralai", model="mistral-medium-3",
    body={"model": "mistral-medium-3", "stream": False,
          "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 128},
)
```

For streaming, both `stream=True` on the client call and `"stream": True` in the
native body are required. The result is a raw SSE response; consume and close
its `iter_lines()` iterator. For normalized text and streaming, the Beta adapter
now accepts `mistral_vertex("mistral-medium-3")` or
`mistral_vertex("mistralai/mistral-medium-3")`; the native Model Garden
`language_model` factory selects the same transport for this recognized ID when
using its default endpoint. Explicit deployed endpoints retain Chat Completions.
Use a supported region; the SDK does not silently redirect `global` requests.
Express Mode rejects this model. Image input and client function tools are also
enabled, with offline coverage for image serialization, tool declarations,
normalized calls and tool-result roundtrips on both factories. JSON-schema output
and portable reasoning remain unsupported. Google's [Mistral guide](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/mistral)
lists Medium 3, Small 2503, OCR 2505 and Codestral 2; OCR has a different payload.
Do not infer that newer models on Mistral's own API are available through Google.

Native Codestral FIM (Beta) uses `prompt` and optional `suffix` rather than chat
messages. Options such as `stop`, `max_tokens`, `temperature`, `top_p` and
`random_seed` are forwarded unchanged; the explicit model argument controls the
route and overrides any model in the body.

```python
result = await mistral_vertex.native.model_garden().codestral_fim({
    "prompt": "def answer():\n    ",
    "suffix": "\n# end",
    "max_tokens": 32,
    "stream": False,
})
```

With `stream=True`, the result is a caller-owned HTTP response: consume and close
its `iter_lines()` iterator, including on cancellation. The method preserves native
JSON/SSE and does not parse or execute generated code. Contract tests cover routing,
caller-data isolation, stream ownership and HTTP failures; live FIM is unverified.
The protocol follows the official Mistral [FIM client](https://github.com/mistralai/client-python/blob/main/packages/gcp/src/mistralai/gcp/client/fim.py)
and [Vertex route hook](https://github.com/mistralai/client-python/blob/main/packages/gcp/src/mistralai/gcp/client/_hooks/registration.py).

### Mistral OCR (native Beta)

`model_garden().mistral_ocr(body, model="mistral-ocr-2505")` sends a unary
`rawPredict` request to publisher `mistralai` in the configured region. It accepts
`document.type` of `document_url` or `image_url`, with the corresponding string
field containing a public URL or a data URI. The SDK forwards that string; it does
not download the input, upload files or resolve Mistral-hosted file IDs.

```python
result = await mistral_vertex.native.model_garden().mistral_ocr({
    "document": {
        "type": "document_url",
        "document_url": "https://example.com/document.pdf",
    },
    "pages": [0],
    "include_image_base64": True,
})
# result retains native pages[].markdown, images, dimensions and usage_info.
```

The explicit model argument controls routing and overrides `body["model"]`.
Optional provider fields are forwarded; their acceptance depends on the selected
model. In particular, do not infer OCR 4 features from the older Vertex 2505 model.
Streaming is rejected by this method. Offline tests cover serialization, page
metadata preservation, input isolation, validation and HTTP errors. Live OCR and
cross-provider normalization remain pending.

Sources: Google's [Vertex OCR model card](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/mistral/mistral-ocr),
[rawPredict routing table](https://docs.cloud.google.com/pubsub/docs/smts/ai-inference-smt)
and Mistral's [OCR payload documentation](https://docs.mistral.ai/studio/document-processing/basic_ocr).
The native payload mapping combines those contracts; it has not been confirmed
against a successful Vertex OCR response in this project.

### Exa and Parallel grounding (native contract)


The native grounded model accepts these Google tool payloads through the existing
`provider_options["tools"]` escape hatch. An explicit tool list prevents the SDK
from adding Google Search. Returned web sources and citation associations use the
same normalized grounding result. No new Stable helper or export is introduced.

```python
import os
from zhivex_ai import generate_grounded_text

result = await generate_grounded_text(
    model=vertex.native.grounded_language_model("gemini-3.8-flash"),
    prompt="Find public documentation about HTTP semantics and cite your sources.",
    provider_options={"tools": [{"exaAiSearch": {
        "api_key": os.environ["EXA_API_KEY"],
        "customConfigs": {"includeDomains": ["ietf.org"], "numResults": 3},
    }}]},
)

# Alternative tool payload for an existing Parallel API key:
parallel_options = {"tools": [{"parallelAiSearch": {
    "api_key": os.environ["PARALLEL_API_KEY"],
    "customConfigs": {"max_results": 3,
                      "source_policy": {"include_domains": ["ietf.org"]}},
}}]}
# Pass parallel_options as provider_options in the call above.
```

Google documents an Exa API key as required. Parallel supports either a key or
an existing Google Cloud Marketplace subscription; omit `api_key` only for that
subscription route. `enable_zero_data_retention=True` requires the separate ZDR
subscription. The SDK neither provisions these subscriptions nor reads partner
keys automatically. Keep credential-bearing provider options out of logs.
See Google's [Exa guide](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/grounding-with-exa)
and [Parallel guide](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/grounding-with-parallel).

Offline tests cover payload preservation, keyless Parallel configuration, absence
of an injected Google Search tool, caller-data isolation and source/citation
normalization. These tests do not prove partner access, billing, ZDR enforcement
or live grounding quality. No Exa/Parallel live certification is recorded.

### OpenAI GPT OSS 120B on Google

`vertex("openai/gpt-oss-120b-maas")` and the short ID
`gpt-oss-120b-maas` select Google's global Chat Completions route with
standard Cloud authentication. The provider remains `vertex`. The catalog and
adapter cover text, streaming, JSON-schema output and client function tools;
vision and portable reasoning are not advertised. Google's
[model card](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/maas/openai/gpt-oss-120b)
lists the upstream model as GA; this does not certify an SDK release.

The [installed-wheel integration run](../releases/0.27.0-vertex-integration-history.json)
passed generation, structured output and a real agent tool roundtrip. Streaming
finished normally but missed the synthetic marker on the first attempt;
an unchanged [streaming-only repeat](../releases/0.27.0-vertex-integration-history.json)
passed. Both results are retained. A subsequent
[single-target matrix run](../releases/0.27.0-vertex-integration-history.json)
passed all four operations together on the same wheel with unchanged verification
inputs. The matrix and manual workflow expose `gpt-oss-120b` for reproducing these
checks. Batch and other GPT OSS sizes have no coverage implied by these checks.

On the subsequent wheel containing Mistral routing, an initial shared-transport
matrix failed GPT OSS streaming/JSON while generation/tools passed. The chat
verifier was then corrected to apply its configured `--max-tokens` budget to
JSON too (previously fixed at 128). The [repeat with 1024 tokens](../releases/0.27.0-vertex-integration-history.json)
passed all four checks on wheel
`7fff45008247b35bfde751298a9a1e2cb24b5b2f055700e7dabc9f3623a2b371`.
This does not establish the cause of the earlier streaming mismatch.

### Gemma and OpenAI-compatible models

Gemma 4 MaaS uses Chat Completions rather than Gemini `generateContent`:

```python
gemma = vertex("google/gemma-4-26b-a4b-it-maas")
result = await generate_text(model=gemma, prompt="Hello")
# generate_object, stream_text and client function tools use the same model.
```

The short ID `gemma-4-26b-a4b-it-maas` selects the same MaaS model. Standard Cloud
configuration is required; Express Mode rejects this model before dispatch.
The route supports text/image input, JSON schema and client function tools. Google
lists the model as Experimental; its catalog availability is `preview`, and this
addition does not extend the Stable Gemini guarantee to all Model Garden models.
Native reasoning options remain provider-specific (`chat_template_kwargs`); they
are not translated from the portable reasoning contract. Google's model card and
thinking guide disagree on reasoning availability, so no portable reasoning claim
is made until that contract is verified.

For another MaaS model or a model served at an existing compatible endpoint:

```python
model = vertex.native.model_garden().language_model(
    "your-deployed-model", endpoint="your-endpoint-id",
)
result = await generate_text(model=model, prompt="Hello")
```

Unknown deployments default to text and streaming. Pass `capabilities=` only for
features verified on that deployment. Constructing a language model does not
deploy infrastructure or infer tool/vision support from its publisher. Deployment
requires the explicit native lifecycle methods described above. Native raw
`chat_completions` remains available for fields outside the normalized contract.

Sources: [Gemma model card](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/maas/google/gemma-4-26b-a4b-it),
[native thinking options](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/maas/capabilities/thinking).

### Gemma native thinking toggle

The native Chat Completions model accepts Google's boolean thinking control:

```python
from zhivex_ai import generate_text

model = vertex.native.model_garden().language_model("google/gemma-4-26b-a4b-it-maas")
result = await generate_text(
    model=model,
    prompt="Calculate 17 times 19. Reply with the integer answer.",
    max_tokens=1024,
    provider_options={"chat_template_kwargs": {"enable_thinking": True}},
)
```

This is a native boolean control, not a mapping for portable reasoning effort
levels or token budgets. Reasoning content remains provider data and is separate
from final text. The [generation checks](../releases/0.27.0-vertex-integration-history.json)
passed both on/off modes; streaming in that first report failed locally because
the runner omitted an await, before dispatching streaming requests. The corrected
[stream-only check](../releases/0.27.0-vertex-integration-history.json)
passed both modes on the same wheel
`8dbc28ca698eee3233be886956bd6f66a4bb8e0d69f3d7b9ed40227a2053af99`.
Reports retain reasoning length only, not reasoning text. This tests toggle
behavior and one arithmetic result, not reasoning quality or GLM thinking.
Use `scripts/verify_vertex_thinking_integration.py` to reproduce.

### Native RAG Engine (Beta)

`vertex.native.rag()` exposes corpus `create`, `get`, `list`, `update`, and
`delete`, plus `import_files`, `list_files`, `get_file`, `delete_file`,
`retrieve_contexts`, `get_operation`, `wait_operation`, `get_engine_config`, and
`update_engine_config`. Engine configuration uses the configured project/region;
updates return a raw operation and may change managed infrastructure. Requests and responses
use Google JSON; list calls return one page including `nextPageToken`. Corpus
updates do not accept an update mask. Corpus deletion accepts `force=True` to
remove contained files. Operations return raw terminal results, including errors;
callers must inspect `error` before treating a completed operation as successful.

```python
from zhivex_ai import create_vertex

vertex = create_vertex(project_id="YOUR_PROJECT", location="us-central1", credentials=credentials)
rag = vertex.native.rag()
page = await rag.list(page_size=20)
contexts = await rag.retrieve_contexts({
    "query": {"text": "What is the retention policy?", "ragRetrievalConfig": {"topK": 3}},
    "vertexRagStore": {"ragResources": [{
        "ragCorpus": "projects/YOUR_PROJECT/locations/us-central1/ragCorpora/CORPUS_ID"
    }]},
})
```

Use a region enabled for your RAG Engine setup. Express Mode does not expose this
client. This native API is separate from portable retrieval and does not create
infrastructure automatically. Contract tests cover routes, payloads, pagination,
authentication and resource validation. Live corpus lifecycle, import/retrieval,
engine configuration updates, uploads and metadata/schema APIs remain unverified or
unimplemented; this is not complete RAG Engine certification. See the
[REST guide](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/models/rag-api-v1).

The [RAG discovery integration](../releases/0.27.0-vertex-integration-history.json)
passed listing in `us-central1` with ADC, returning zero corpora and no next page,
on wheel `758a7778182f61d048ef3e12c2e95fabfb471d988f29038f8dfe7effd64e4f0e`.
No infrastructure was created and no file contents were read. Reproduce with
`scripts/verify_vertex_rag_discovery.py --wheel WHEEL --project PROJECT
--location us-central1 --output NEW_REPORT`. This verifies listing only, not
corpus lifecycle, import, indexing or retrieval quality. Add `--engine-config`
to read regional engine configuration as well; this flag never updates it.

The [regional configuration read](../releases/0.27.0-vertex-integration-history.json)
also passed with ADC in `us-central1`, reporting the managed DB `basic` field.
The wheel is `8dbc28ca698eee3233be886956bd6f66a4bb8e0d69f3d7b9ed40227a2053af99`.
No configuration update was executed; PATCH remains contract-tested only.

The resumable `scripts/verify_vertex_rag_lifecycle.py` targets a temporary empty
corpus: create, get, update description, then delete. It reserves a checkpoint
before creation and refuses blind resubmission after an uncertain result. It
retains operation names for polling and attempts cleanup of its captured corpus
even if an update fails. The [live attempt](../releases/0.27.0-vertex-integration-history.json)
was rejected at creation in `us-central1` with HTTP 400 / INVALID_ARGUMENT;
sanitized error categories mention an allowlist and region. These operations
remain unverified live. The [subsequent list](../releases/0.27.0-vertex-integration-history.json)
checks for remaining corpora without creating resources.

### Gemini Pro custom-tools endpoint

`vertex("gemini-3.1-pro-preview-customtools")` selects Google's separate Pro
endpoint optimized for custom tools and bash workflows. The catalog marks it
Preview, in `global`, with no default alias reassignment. Use client tools with
the same portable agent contract as other Gemini models. Selecting this model
does not execute bash automatically or grant additional tool permissions.
Google documents this endpoint separately and excludes Provisioned Throughput;
see the [model card](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-1-pro).
The matrix exposes `gemini-pro-customtools` for its own operation set.
The [exact-wheel integration](../releases/0.27.0-vertex-integration-history.json)
passed generation, streaming, structured output, client tools, token counting
and portable retrieval with ADC/global on wheel
`839ac04d2f9a4a3469b094be15fa3fe18e50164cb90c8635dd902f63eba7d389`.
Inputs remained unchanged during the run. Hosted tools, multimodal inputs and
comparative tool-selection quality are not covered by this result.

A separate [hosted-tools matrix run](../releases/0.27.0-vertex-integration-history.json)
passed all three checks for both `gemini-3.1-pro-preview` and
`gemini-3.1-pro-preview-customtools`: Google Search returned sources and associated
citations, code execution returned the expected arithmetic result, and URL
context reported successful retrieval with the expected document title. This
used ADC/global and wheel
`55c014c9fa2eeb6bc9f3f7912467b7f72ecb8961f503f98573babd608df436ca`;
verification inputs remained unchanged. These are bounded integration checks,
not comprehensive grounding accuracy or comparative tool-selection evaluations.

The [Pro multimodal matrix](../releases/0.27.0-vertex-integration-history.json)
also passed PDF marker extraction and synthetic TTS-to-audio transcription for
both variants. Pro additionally identified the exact temporal color sequence in
an MP4. Custom Tools received HTTP 429 on that video check, so this run passed
three of four targets. These short fixtures do not validate long documents,
long audio/video, or general multimodal quality.
A [video-only repeat](../releases/0.27.0-vertex-integration-history.json)
after more than one minute passed for Custom Tools on the same wheel and fixture,
with unchanged limits and verification inputs. The original 3/4 matrix is retained.

### Publisher catalog discovery limitations

`scripts/verify_vertex_model_discovery.py` reads `list_models(...)` pages from an
exact installed wheel, accepts repeated `--publisher`, and writes a fresh report.
It limits each publisher to 20 pages and 120 seconds, rejects repeated cursors
and mismatched model identities, and marks a page-limit result incomplete.
Only selected public model metadata is retained; page tokens are omitted.

The [Google/Meta discovery report](../releases/0.27.0-vertex-integration-history.json)
returned 27 Google entries and no Meta entries, with no next page. This is not
an exhaustive inference inventory: tested Gemma and Veo models were absent, while
historical IDs were present. Neither a catalog entry nor its launch-stage field
proves current inference availability, project entitlement, or SDK support.
Absence does not prove unavailability. Use official model documentation and
model-specific integration evidence alongside discovery.

### Veo video extension

The resumable runner accepts `--video INPUT.mp4`, sends the native
`extra_body.instance.video` inline payload, and binds its SHA256 to the
checkpoint. It rejects combinations with first/last frames or asset references
before authentication. Extension omits `durationSeconds`, using the service's
extension duration. The runner bounds MP4 inputs to 20 MB; this is a runner
limit, not a claim about the service limit.

The [Veo 3.1 Standard extension integration](../releases/0.27.0-vertex-integration-history.json)
passed creation, terminal polling and MP4 validation on wheel
`efe59f1e00cafa3ce9e4a144795509d6d282521bcd8ed658914821558ab7bab5`.
It reused the existing eight-second synthetic asset-reference video, recovered
from its completed operation, without generating another source clip. Input and
output hashes are recorded. This proves the request and payload path, not visual
continuity, measured output duration, or extension on Fast/Lite. See Google's
[extension guide](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/video/extend-videos).

### Veo asset reference images

The resumable runner accepts `--reference ASSET.png` (repeat up to three times)
and sends `referenceImages` with `referenceType: "asset"` through the existing
native instance payload. This mode uses eight seconds and cannot be combined
with `--image` or `--last-frame`. Ordered SHA256 digests bind all references to
the checkpoint; a changed reference, excessive count or incompatible mode is
rejected before authentication or submission.

The [Veo 3.1 Standard asset-reference integration](../releases/0.27.0-vertex-integration-history.json)
passed creation, terminal polling and MP4 validation for one synthetic red-circle
PNG, 720p and no generated audio, on wheel
`efe59f1e00cafa3ce9e4a144795509d6d282521bcd8ed658914821558ab7bab5`.
This is evidence of request acceptance and output format; multi-reference
consistency and visual fidelity remain untested. Reference support is
model-specific, and the runner does not imply it is available on Fast or Lite.

### Veo first and last frames
The existing video client accepts Google's `image` and `lastFrame` fields through
`extra_body={"instance": ...}`. The resumable video runner now accepts
`--image FIRST.png --last-frame LAST.png`, validates both PNG inputs, and binds
both digests to its checkpoint. Resuming with a changed last frame is rejected
before authentication or submission, preventing a different request from being
silently associated with an existing paid job.

The [Veo 3.1 Fast integration](../releases/0.27.0-vertex-integration-history.json)
passed creation, terminal polling and MP4 validation with the same synthetic red
circle as first and last frame, four seconds at 720p without audio. The wheel is
`efe59f1e00cafa3ce9e4a144795509d6d282521bcd8ed658914821558ab7bab5`.
This verifies request acceptance and output format, not visual interpolation or
frame fidelity. Distinct last frames, asset references, video extension and GCS
output still need their own live evidence. See Google's
[first/last frame guide](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/video/generate-videos-from-first-and-last-frames).

### Self-deployed endpoint prediction
Discover existing endpoints with `await garden.list_endpoints(page_size=100)`
and retrieve one with `await garden.get_endpoint("123456789")`. Listing returns
one raw page and preserves `nextPageToken`; pass it as `page_token` to continue.
Optional `filter`, `order_by` and `read_mask` values are URL-encoded. These methods
use the configured project and location and never deploy resources.
The read-only `scripts/verify_vertex_endpoint_discovery.py` validates an installed
wheel and checks up to five pages per requested `--location`; an empty list is
reported separately from a skipped endpoint lookup and never proves inference.

The Beta Model Garden client exposes the native
[endpoint prediction APIs](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/rest/v1/projects.locations.endpoints):

```python
garden = provider.native.model_garden()
result = await garden.endpoint_predict(
    endpoint="123456789",
    body={"instances": [{"prompt": "Hello"}], "parameters": {}},
)
response = await garden.endpoint_raw_predict(
    endpoint="123456789", body=b"arbitrary container input",
    content_type="application/octet-stream",
)
data = await response.read()
```

`endpoint_predict` returns Google's JSON without interpreting the model schema.
`endpoint_raw_predict` preserves the HTTP response, including deployed-model
headers and binary output; `stream=True` selects `streamRawPredict` and transfers
response-consumption ownership to the caller. SSE consumers should close their
line iterator if stopping early. Automatic retries default to zero and explicit
retry options apply only before a response is returned. Endpoint IDs resolve
inside the provider's configured project/location and origin; URLs and full
resource paths are rejected. Dedicated endpoint hosts require an explicitly
configured provider base URL. These methods do not deploy models or infer their
capabilities. Contract tests cover routing, bytes, headers, errors and invalid
destinations; no live self-deployed endpoint has been validated yet.

### Conversational Live audio input check
The [continuous-utterance run](../releases/0.27.0-vertex-integration-history.json)
passed all five operations on wheel
`63483b046b7fb9ca24460ef47d7a03102f953474211e98eb4695f7b445573998`
in `us-central1`: synthetic TTS, independent fixture transcription, setup,
audio-to-audio marker roundtrip and cleanup. It uses a short continuous spoken
instruction and default VAD. This provides basic conversational audio evidence;
it does not certify long speech, interruption, reconnection or arbitrary pauses.
The matrix and manual workflow now include `live-audio` (15 targets total).
Future runner reports also bind the generated PCM fixture by SHA256 and size.
The failures below remain historical evidence; the short utterance result supports
investigating VAD segmentation without proving the root cause of each failure.

The runner now independently transcribes the synthetic PCM fixture with
Transcribe 3.5 before opening Live. The
[validated-fixture run](../releases/0.27.0-vertex-integration-history.json)
passed that prerequisite, setup and cleanup, but failed both Live transcript
marker checks after a completed audio turn. Thus a missing marker in the fixture
does not explain this particular failure. A separate
[2000 ms VAD experiment](../releases/0.27.0-vertex-integration-history.json)
timed out without output; it is not an accepted configuration recommendation.
The optional `--vad-silence-ms 0..2000` is restricted to `--audio-input` and recorded
in reports. Further diagnosis must inspect turn boundaries and received audio;
these reports do not establish a provider or SDK root cause.

### MaaS lifecycle notice (2026-09-19)

The existing Vertex routes for DeepSeek V3.2, GLM 5, Kimi K2 Thinking, MiniMax M2
and Qwen3-Next Instruct are now catalogued as `deprecated`. Google lists their
MaaS retirement date as October 21, 2026 in its
[open model deprecations](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/deprecations/open-models).
Their aliases and runtime routes remain usable for existing workloads while the
upstream endpoints permit them. Successful integration evidence does not override
this lifecycle status. Self-deployment requires a separate endpoint and validation;
the SDK does not silently migrate these IDs. The newer GLM 5.2 and Gemma 4 entries
retain their existing status.

### Embedding protocols

`vertex.embedding_model("gemini-embedding-2")` uses `embedContent`, including
multimodal parts. Each element passed to `embed` produces one vector, while a list
of parts inside that element forms one multimodal input. Calls are sequential and
preserve input order. `output_dimensionality`, task types and titles are mapped
to the native request; `auto_truncate` is rejected for this protocol. Traditional
text embedding models retain `predict`. Missing, non-finite or malformed vectors
raise `ValidationError` instead of returning an apparently successful empty result.

The [expanded Embedding 2 integration](../releases/0.27.0-vertex-integration-history.json)
passed text, multiple inputs, image, PDF, video, synthetic WAV audio and combined
text/image on wheel `927e8b48283b789a5565e8a24ce463430e144a74bac90bd64a01c9cb14a348a5`
with ADC in `us`. The runner's `--multimodal` flag checks finite, nonzero vectors
of 128 dimensions and input/output cardinality; it records media fixture hashes.
This validates modality acceptance, not semantic retrieval quality. The matrix
target `embedding-2` now requires all seven checks. Google's
[multimodal embedding guide](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/embeddings/get-multimodal-embeddings)
describes the model's media inputs.

For provider-specific controls, the Beta native client accepts the complete
`embedContent` request and returns its unmodified JSON response:

```python
result = await provider.native.model_garden().embed_content(
    model="gemini-embedding-2",
    body={
        "content": {"parts": [{"fileData": {
            "fileUri": "gs://YOUR_BUCKET/document.pdf", "mimeType": "application/pdf"
        }}]},
        "embedContentConfig": {"outputDimensionality": 128, "documentOcr": True},
    },
)
```

The native response retains `embedding`, `usageMetadata` and `truncated` when
present. `audioTrackExtraction` controls extraction of a video's audio track.
Options are passed through for Google to validate against the selected model;
this method does not imply that every model accepts every configuration. Google
now documents `embedContentConfig` as the replacement for the legacy top-level
configuration fields. The existing normalized embedding adapter remains compatible
with the legacy fields; native configuration is explicit and separate.

The [native configuration integration](../releases/0.27.0-vertex-integration-history.json)
passed normalized text plus native PDF with `documentOcr: true` and native video
with `audioTrackExtraction: false`, using ADC in `us` on wheel
`3efeca27e2c0a07aa50f25be804f657347a2641730319c5174fc0abab2d781e3`.
The synthetic fixtures returned valid 128-dimensional vectors. This demonstrates
acceptance of those configurations, not OCR quality or extraction of an actual
audio track (the video fixture is silent). Reproduce with the embedding runner's
`--native-config` flag. Other live reports retain their original wheel hashes.

`multimodalembedding@001` uses one `predict` request per input with `text` or
`image` fields, maps `output_dimensionality` to `dimension`, and reads the
corresponding `textEmbedding` or `imageEmbedding`. Portable inputs can be text,
inline images, or GCS images. Combined modalities and segmented video have multiple
native output vectors; use `native.model_garden().raw_predict` to preserve that
response rather than forcing it into one portable vector. Text task/title/truncation
options are rejected for this model. See Google's
[multimodal protocol](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/embeddings/get-multimodal-embeddings).

Installed-wheel checks on 2026-09-19 verified finite, nonzero 128-dimensional
vectors with ADC for these exact targets (hashes are recorded per report):

- [Gemini Embedding 2](../releases/0.27.0-vertex-integration-history.json),
  in `us`: text, multiple inputs and a synthetic image.
- [Gemini Embedding 001](../releases/0.27.0-vertex-integration-history.json),
  in `us-central1`: single text input.
- [Text Embedding 005](../releases/0.27.0-vertex-integration-history.json),
  [004](../releases/0.27.0-vertex-integration-history.json), and
  [Multilingual 002](../releases/0.27.0-vertex-integration-history.json),
  in `us-central1`: single and multiple text inputs.
- [Multimodal Embedding 001](../releases/0.27.0-vertex-integration-history.json),
  in `us-central1`, on the subsequent protocol-fix wheel: text, multiple inputs and image.

Reproduce with `scripts/verify_vertex_embedding_integration.py`, selecting a model,
location, installed wheel and fresh output; `--batch` adds multiple inputs and
`--image` adds a synthetic PNG. These checks do not evaluate semantic retrieval
quality, all dimensions/task types, GCS access, video, or protected release certification.

### Resource clients

These additions are Beta; Google Interactions remains Experimental upstream.
Express Mode exposes only Google's Express API subset, not administrative
clients. Standard Cloud resources require a project, suitable IAM permissions,
and a supported location/model. An API key alone does not establish access to
Agent Runtime, Memory Bank, batch jobs, or partner models.

| Entry point | Contract |
| --- | --- |
| `vertex(...)` | Stable portable Google model generation, streaming, JSON, tools |
| `vertex.tokens()` | Token counting |
| `vertex.caches()` | Create/get/list/update/delete context caches; accepts returned full resource names |
| `vertex.batches()` | Native BatchPredictionJob create/retrieve/list/cancel; Google inputConfig/outputConfig bodies |
| `vertex.interactions()` | Native v1beta1 creation, retrieval and streaming; raw Google bodies |
| `vertex.videos()` | Veo generation and fetchPredictOperation polling |
| `vertex.media()` | Lyria 2 prediction and Lyria 3 Clip/Pro Interactions, normalized audio results |
| `vertex.native.agent_platform()` | Google-managed agents, sessions, Memory Bank, operation polling |
| `vertex.native.model_garden()` | Publisher discovery, rawPredict/streamRawPredict, Claude Messages and OpenAI-compatible Chat Completions |

Platform, Model Garden, batch and Interactions clients return Google payloads;
cache and media clients retain the SDK's normalized result types. Long-running operations remain
operations until polled; a returned resource name is not completion evidence.
Terminal failed operations preserve their error details. Raw streaming responses
must be consumed and closed by the caller. Model Garden native routes do not
promote partner models to the portable Google contract or certify their availability.

```python
vertex = create_vertex(project_id="my-project", location="us-central1", express_mode=False)
cache = await vertex.caches().create({
    "model": "gemini-2.5-flash",
    "contents": [{"role": "user", "parts": [{"text": large_document}]}],
    "ttl": "300s",
})
try:
    result = await generate_text(
        model=vertex.native.language_model("gemini-2.5-flash"),
        prompt="Summarize the document",
        provider_options={"cachedContent": cache.name},
    )
finally:
    await vertex.caches().delete(cache.name)

platform = vertex.native.agent_platform()
# Resource creation can incur charges. Supply your existing agent ID.
session_operation = await platform.sessions(agent_id).create({"userId": "user-123"})
session = await platform.wait_operation(session_operation["name"])
memories = await platform.memory_bank(agent_id).retrieve({
    "scope": {"user_id": "user-123"}, "similaritySearchParams": {"searchQuery": "preferences"},
})

# Claude is authored by Anthropic and hosted by Google on this route.
response = await vertex.native.model_garden().anthropic_messages(
    model="claude-sonnet-4-6",
    body={"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 128},
)
```

Vertex Live uses project-scoped authentication and fully qualified publisher model
names. Setup waits for `setupComplete`; browser ephemeral tokens remain unsupported.
Use a Google-documented Live model and supported region. Express Mode cannot be
used as evidence for Live support.

Lyria 3 Clip/Pro use the Vertex Interactions endpoint with `store=False` by default.
They do not use Gemini `generateContent`. Vertex Interactions has a separate
model/schema lifecycle: availability of a Gemini text model on `generateContent`
does not imply that model works on Interactions.

## Verification


Run `tests/test_vertex_provider.py`, Google provider tests, provider support tests,
and Tier-1 contracts for offline verification. Live smoke recognizes Vertex API
keys as well as token/project or explicitly configured ADC:

```bash
ZHIVEX_SMOKE_PROVIDERS=vertex ZHIVEX_SMOKE_VERTEX_MODEL=gemini-2.5-flash \
ZHIVEX_SMOKE_AGENTS=1 ZHIVEX_SMOKE_PORTABLE_CERTIFICATION=1 make smoke
```

Source smoke, installed-wheel integration evidence, and protected release
certification are different states. Never reuse a Gemini Developer API result to
certify Vertex. Model/region/IAM availability must be verified for each target.

Sources reviewed 2026-09-19:

- [Google name changes](https://docs.cloud.google.com/gemini-enterprise-agent-platform/vertex-ai-name-changes)
- [Express API](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/express-mode/api-reference)
- [Google Auth ADC](https://google-auth.readthedocs.io/en/latest/reference/google.auth.html)
- [Context caching](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/rest/v1/projects.locations.cachedContents)
- [Interactions](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/models/interactions-api)
- [Agent Platform REST](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/rest)
- [Claude on Google](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/claude/use-claude)

A runnable [Vertex example](../../examples/integrations/vertex.py) supports both
Express and explicit ADC. For exact installed-wheel evidence, run
`scripts/verify_vertex_integration.py --wheel WHEEL --output NEW_REPORT --model MODEL`
with either `--express` or `--adc --project PROJECT --location global`. The runner
checks source/wheel/installation equality before sending any model requests.

For temporary Cloud resource lifecycles, use
`scripts/verify_vertex_native_integration.py --wheel WHEEL --output NEW_REPORT
--project PROJECT --location global --music-model lyria-3-clip-preview`.
It explicitly uses ADC, creates synthetic resources, and attempts cleanup in
`finally` blocks. Inspect cleanup results if a run fails.

### Gemini 3.5 Transcribe Preview

`vertex.transcription_model("gemini-3.5-transcribe-preview")` accepts audio and
an optional `language` hint. This model receives audio only: do not provide a
text prompt. For Google-specific options, use the native factory:

```python
result = await vertex.native.transcription_model(
    "gemini-3.5-transcribe-preview"
).transcribe(
    audio=audio,
    language="en-US",
    provider_options={"generationConfig": {"audioTranscriptionConfig": {
        "wordTimestamp": True,
        "diarization": True,
        "customVocabulary": ["Zhivex"],
    }}},
)
```

The adapter normalizes text from text parts or `audioTranscription.text` and
preserves speaker labels and word timing in `raw_response`. This is a Beta SDK
integration with an upstream Preview model. The synchronous factory rejects
`gemini-3.5-transcribe-live-preview`; that model needs the separate realtime flow.

[Official Transcribe protocol](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-5-transcribe).

### Gemini 3.5 Transcribe Live Preview

Use `vertex.realtime_model("gemini-3.5-transcribe-live-preview")` in `global`.
The adapter defaults to TEXT output and enables input transcription. It rejects
text input, system instructions, tools and output audio at connection setup.

```python
from zhivex_ai.live import RealtimeSessionConfig

session = await vertex.realtime_model(
    "gemini-3.5-transcribe-live-preview"
).connect(RealtimeSessionConfig(
    provider_options={"inputAudioTranscription": {"languageCodes": ["en-US"]}},
))
# Send PCM AudioFrame chunks; set is_final=True on the final chunk.
# Read user RealtimeTranscriptEvent events concurrently with sending audio.
# Always close the session in a finally block.
```

Interim transcripts have `is_final=False`; finalized utterances have
`is_final=True` independently of conversational turn completion. Original event
metadata is retained. The adapter waits for `setupComplete` before returning the
session and sends `audioStreamEnd` for a final audio frame.

This is Beta SDK support for an upstream Preview model, with integration-only
baseline evidence. Long sessions, reconnect/resumption, vocabulary accuracy,
multiple languages and production load remain unverified.

### Gemini Omni 1.1 Flash Preview on Vertex

Vertex uses the distinct Preview ID `gemini-omni-1.1-flash-preview` and
`native.interactions()`. Do not substitute the Developer API GA ID or call Veo's
`predictLongRunning` protocol for this model.

```python
from zhivex_ai.types import RetryOptions

result = await vertex.native.interactions().create({
    "model": "gemini-omni-1.1-flash-preview",
    "store": False,
    "background": False,
    "input": [{"type": "text", "text": "A blue ball rolls across a white floor."}],
    "response_format": [{"type": "video", "duration": "3s", "resolution": "360p"}],
    "generation_config": {"video_config": {"task": "text_to_video"}},
}, RetryOptions(timeout_ms=230000, max_retries=0))
```

[Google's protocol reference](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/video/generate-videos-from-text).

### Robotics ER 2 access

The exact ID `gemini-robotics-er-2-preview` returned HTTP 404 for both text and
streaming on the project/global route in the
[recorded check](../releases/0.27.0-vertex-integration-history.json).
Google describes Vertex Robotics ER 2 as [early access](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/gemini-robotics-er).
The 404 alone does not establish its cause. No successful integration or full
capability coverage is claimed for this target.

### Gemini 3.5 Live Translate Preview

Use `vertex.realtime_model("gemini-3.5-live-translate-preview")` with `global`.
Translation settings map to `setup.generationConfig.translationConfig`:

```python
from zhivex_ai.live import RealtimeSessionConfig

session = await vertex.realtime_model(
    "gemini-3.5-live-translate-preview"
).connect(RealtimeSessionConfig(
    translation_target_language_code="es",
    translation_echo_target_language=False,
    input_audio_media_type="audio/pcm;rate=16000",
    input_sample_rate_hz=16000,
    output_audio_media_type="audio/pcm",
))
```

Send mono PCM audio frames and consume audio and transcript events concurrently.
Text input, instructions and tools are unsupported. Capabilities no longer
advertise realtime tools for this model. The model is upstream Preview; this
SDK integration remains Beta.

The [initial check](../releases/0.27.0-vertex-integration-history.json)
passed TTS, setup and cleanup, but translation timed out. A successful handshake
alone does not establish working speech translation. See subsequent evidence
below; no release certification is implied.

[Official model/configuration reference](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-5-live-translate).

Reproduce with `scripts/verify_vertex_translate_live_integration.py`, passing
wheel, project and a fresh output path. No generated audio is retained.

### GLM 5.2 contract and partner diagnostics

`vertex("glm-5.2-maas")` and
`vertex.native.model_garden().language_model("zai-org/glm-5.2-maas")` now share
capabilities for client functions and JSON-schema output. Contract tests verify
function arguments/IDs, schema requests and normalized results on both routes.
Vision and portable reasoning controls are not advertised. This is Beta support
with offline evidence, not a successful live certification.

A bounded diagnostic on 2026-09-19 confirmed the following provider responses:

- `grok-4.6`: HTTP 400 explicitly requires the `publisher/model` format on OpenAPI.
- `xai/grok-4.6`: HTTP 404 says the model was not found or the project has no access;
  it does not distinguish those causes.
- `zai-org/glm-5.2-maas`: HTTP 429 explicitly reports too many concurrent requests.

These diagnostics used the prior installed wheel and synthetic text requests.
No model enablement or IAM permissions were changed. GLM tools/JSON still require
live checks when the target accepts requests; Grok remains unverified.

### Gemini 3.8 temporal video input

The [video-input run](../releases/0.27.0-vertex-integration-history.json)
on the same current wheel recovered the exact red → blue → green order from a
six-second inline MP4. The prompt did not reveal the expected colors or sequence.
The fixture was decoded locally to validate all twelve frames before inference;
its SHA is recorded alongside the wheel SHA. This checks temporal understanding
and schema output, rather than only successful upload.

The [initial attempt](../releases/0.27.0-vertex-integration-history.json)
returned HTTP 429. One retry after the other checks finished passed; both reports
are retained. Reproduce with `scripts/verify_vertex_video_input_integration.py`.
Long videos, audio/video synchronization, clipping, custom frame rates and GCS
video input remain separate checks. This does not certify every video capability.

### Reproducible baseline matrix

`scripts/run_vertex_matrix.py` runs reviewed targets sequentially against one
external installed wheel. Pass `--wheel`, `--project`, a new `--output-dir`, and
optionally repeated `--target` selections. Without a selection it runs all 33
baseline targets, including paid audio/video generation. Each target retains its
own report. The text-family targets are `gemini-core` (3.8 Flash),
`gemini-pro` (3.1 Pro Preview), `gemini-pro-customtools`,
`gemini-lite` (3.5 Flash-Lite), `gemma` (Gemma 4 26B A4B IT MaaS),
and `gpt-oss-120b` (text, streaming, JSON and client function tools).
Image targets include generation and editing
separately for Pro Image, Flash Image, and Flash-Lite Image. Each target has
its own required operation set.
`mistral-medium` checks text, streaming, tools and vision in `us-central1` through the
regional raw-prediction adapter. Availability failures remain failures in the
matrix; a target's inclusion does not mean that it passed live.
Hosted-tool targets `gemini-hosted`, `gemini-pro-hosted` and
`gemini-pro-customtools-hosted` independently check Google Search citation
associations, successful code execution and URL retrieval on each exact model.
The Pro and Custom Tools variants also have separate `-multimodal` targets for
PDF/audio input and `-video-input` targets for temporal color-sequence recognition.
`gemini-cache` creates a short-lived synthetic explicit cache, reads and updates
it, verifies generation with reported cached-token use, deletes it and requires
a subsequent 404. The direct [Gemini 3.8 Flash check](../releases/0.27.0-vertex-integration-history.json)
passed all six operations in `global`, reporting 12,289 cached content tokens.
TTL is initially three minutes and updated to four; cleanup runs even on failure.
If creation returns no identifier, expiry bounds retention; the runner cannot
confirm deletion of an unidentified resource. It does not automatically retry
creation or infer billing savings from the token counter.
`matrix.json` records selected scope, wheel hash, source revision,
dirty-worktree status and hashes of the controlling scripts. New reports also
record `input_sha256` for all local Python scripts, Vertex fixtures,
`pyproject.toml`, and `uv.lock`. The runner checks this snapshot before and after
each target; changes invalidate the run (`inputs_unchanged=false`) and prevent
further dispatch. Credentials and `.env` are excluded. This detects persistent
changes across checkpoints; it is not an immutable filesystem snapshot or
a record of the installed third-party environment. Older reports lack these
fields and are not retroactively upgraded.

The aggregator requires the exact model, region, wheel hash, fresh timestamp and
all required operations, plus a successful runner exit code. It rejects extra
failed operations, incomplete results and stale or mismatched evidence. A pending
entry after interruption is not passed; choose a new directory only after checking
the original process rather than blindly launching duplicate generations.

The manual `.github/workflows/vertex-integration.yml` workflow builds on main,
installs outside the checkout, uses the protected `provider-certification`
environment and retains wheel plus evidence. It defaults to the Gemini core target.
Set environment variable `ZHIVEX_CERT_GOOGLE_CLOUD_PROJECT` and secret
`ZHIVEX_CERT_GOOGLE_CREDENTIALS_JSON` to service-account ADC or an operational
external-account credential configuration. The workflow writes credentials to a
restricted temporary file and removes it with an always-run step. It does not use
or upload a developer's local ADC. Concurrent workflow runs are serialized.

This workflow is a candidate **integration** workflow, not a replacement for the
versioned certification policy or proof of a protected run. It has not been
executed on GitHub from these local changes. The older 0.23.0 certification remains
separate. Baseline targets do not cover every model, every capability, or the
remaining managed platform services; their success must not be labeled complete
Vertex certification.

The [local matrix exercise](../releases/0.27.0-vertex-integration-history.json)
selected only `gemma` and passed all five required operations through the new
orchestrator. It records `worktree_dirty=true` and
`all_baseline_targets_selected=false`; it does not stand in for a full matrix or
GitHub workflow run.

### Resumable Gemini batch integration

`scripts/verify_vertex_batch_integration.py` submits one synthetic JSONL row to
Gemini 3.8 in `global`, following the [Cloud Storage batch contract](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/batch-inference/new-job-from-cloud-storage).
It requires an exact external installed `--wheel`, `--project`, a private local
`--state` path and a fresh `--output` report path. It creates its own private
bucket in `us-central1`; it does not modify an existing bucket or IAM policy.

The checkpoint is reserved before resource creation and records the job name
before polling. A polling timeout leaves the same job and storage intact: rerun
with the same state and a fresh report to continue observation. Interrupted
creation without a job identifier requires reconciliation, not another submission.
Terminal jobs are checked for one successful, complete output row with the expected
marker; row errors, truncation and extra/missing rows fail the check. Cleanup
removes the owned temporary objects with generation preconditions and then the
bucket, only after the job is terminal. The batch job record itself is retained;
this runner does not prove cancellation or job deletion. Reports remain local
integration evidence and pending observations never count as success.

Lifecycle regressions cover resuming without a second create call, preserving
storage on timeout, refusing uncertain submissions before authentication, and
deleting exact object generations only after a confirmed terminal result.
The repository checks after these additions passed: 1,161 tests, 16 skipped,
280 subtests, 85.06% coverage. These offline checks do not convert a running
cloud batch job into a successful integration result.

The [terminal batch observation](../releases/0.27.0-vertex-integration-history.json)
confirmed `JOB_STATE_SUCCEEDED`, the expected row content, and deletion of the
temporary objects and bucket on the current wheel. Earlier pending observations
remain unchanged. A [separate cleanup check](../releases/0.27.0-vertex-integration-history.json)
deleted the completed batch job record, waited for its deletion operation, and
confirmed that retrieving the job returned 404. Cancellation, per-row mixed
errors and resubmission of failed rows remain separate integration checks.

### Veo 3.1 first-frame input

The [Veo image-to-video check](../releases/0.27.0-vertex-integration-history.json)
passed creation, terminal polling and MP4 payload normalization for
`veo-3.1-fast-generate-001` in `us-central1`, using wheel
`246c98a3fd315133c70efb12eda59c74bd2b18eaed17944aba2fdd6c91715d11`.
The request used a synthetic 1280×720 PNG and requested four seconds without audio.
The runner accepts `--image` and binds the image SHA to the checkpoint so a resume
cannot silently change the first frame. It sends native `instance.image` with
`bytesBase64Encoded` and `mimeType`, following the
[official image-to-video contract](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/video/generate-videos-from-an-image).
This verifies input acceptance and binary output, not visual fidelity. First/last
frame pairs, extension, reference images and sound remain separate checks.

### Restricted-model catalog and Cyber capability correction

The catalog includes `gemini-3.8-flash-cyber` with `availability="limited"`
(allowlisted upstream GA) and `gemini-robotics-er-2-preview` as Preview with
catalog-only evidence. Robotics' conservative catalog flags cover documented
visual/file inputs and reasoning; they do not certify its full early-access API.
Cyber's model-specific adapter keeps text/JSON/multimodal capabilities but no
longer advertises function tools, tool choice, embeddings or hosted agent tools.
Native generation and streaming reject mapped tool payloads before HTTP, matching
Google's model reference. The grounding factory and batch creation also reject
Cyber before HTTP; batch checks recognize both publisher and full project resource
names. Ordinary Gemini 3.8 retains its tool, grounding and batch capabilities.
These changes require a new wheel; earlier integration reports retain their
original hashes and are not evidence for the changed artifact.

After completing the Cyber grounding/batch guards, the new installed wheel SHA
`0dc7ffa6f6403025bca2294616785589270f800a128421650308da2125a463e1`
passed the [Gemini 3.8 core matrix target](../releases/0.27.0-vertex-integration-history.json):
generation, streaming, JSON, client tools, token counting and portable retrieval.
All 1,163 local tests passed (16 skips, 280 subtests, 85.08% coverage). Other
families' earlier reports retain their own wheel hashes. This run selected only
Gemini core and does not certify all fourteen targets or a release.

### Native Grok Responses API

`provider.native.model_garden().responses(body, options=...)` posts the native
Responses schema to `endpoints/openapi/responses` using the configured Vertex
ADC/project/location. This Beta method preserves the raw JSON result. With
`stream=True`, it returns a raw HTTP response: consume its `iter_lines()` iterator
and close that iterator when stopping early. It does not normalize Responses into
portable text events or provide stored-response retrieval/deletion.

```python
result = await provider.native.model_garden().responses({
    "model": "xai/grok-4.6",
    "input": "Explain asynchronous programming in one sentence.",
    "max_output_tokens": 512,
    "store": False,
})
```

The runnable example supports `--adc --project PROJECT --responses --model
xai/grok-4.6`. The method follows Google's [Responses contract](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/grok/responses).
Contract tests cover the US endpoint, native payload preservation, streaming,
unsafe endpoint rejection and HTTP errors; the full local suite passed 1,164 tests
with 16 skips, 280 subtests and 85.11% coverage.

Grok 4.6 [US Chat Completions](../releases/0.27.0-vertex-integration-history.json)
returned 404 for text and streaming. On the subsequently built wheel SHA
`24f1036fad9996dea4a1e1334fe590cfb61cbc0729ead845b4a148c7c53723d8`,
[global Responses](../releases/0.27.0-vertex-integration-history.json)
also returned 404 for both operations. The runner is
`scripts/verify_vertex_responses_integration.py`. These results do not establish
whether access, enablement or availability caused the 404; Grok remains unverified.
The new wheel does not inherit live certification from reports on older wheels.

### Normalized Responses models

For the SDK text/streaming contract, explicitly select the Beta Responses route:

```python
from zhivex_ai import generate_text, stream_text

model = provider.native.model_garden().responses_model("xai/grok-4.6")
result = await generate_text(model=model, prompt="Hello", max_tokens=128)
streamed = await stream_text(model=model, prompt="Hello", max_tokens=128).collect()
```

This adapter preserves Vertex identity and ADC and reuses the Responses message,
usage and event normalizer. Defaults advertise only text and streaming. An explicit
`capabilities=` override requires verification for the actual deployment; it is not
proof that Grok tools, JSON or vision were tested. The existing `language_model`
method continues using Chat Completions. No stored-response lifecycle is added.

The runnable example exposes `--normalized-responses` (requires `--adc`, mutually
exclusive with raw `--responses`). The integration runner accepts `--responses
--basic` for this route. Contract checks validate normalized text and token usage
in both modes, the Vertex OAuth header and endpoint, native options and rejected
endpoint overrides. Grok's prior 404 results remain unresolved; this addition is
not a successful live Grok certification.

Normalized Vertex Responses streams reject standalone error events, missing
terminal events (including an empty stream or `[DONE]` alone), and terminal events
whose response status is inconsistent. The adapter closes its line iterator on
completion, failure, or interruption after streaming starts. An incomplete
response caused by `max_output_tokens` maps to `length`; a failed response maps
to `error`. These behaviors have offline contract coverage, not live Grok evidence.
