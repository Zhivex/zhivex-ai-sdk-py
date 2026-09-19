# Vertex support — 0.27.0 release scope

Prepared 2026-09-19. Package maturity: **Beta**. This is a bounded release of the
implemented integration, not a claim that every Google/partner model or platform
service has been certified. Claude is excluded from this work at the user's request.

## Google models

| Area | Evidence and remaining limits |
| --- | --- |
| Gemini 3.8 Flash, Pro 3.1, Custom Tools, Flash Lite 3.5 | Text, streaming, structured output and client tools have live integration evidence. Some later runs received 429; no single all-target release run passed. |
| Gemma 4 26B A4B IT | Text, streaming, JSON, tools and vision passed together on an earlier exact wheel. Other Gemma sizes are not certified. |
| Image and speech | Generation/editing and synthetic speech/transcription passed for selected catalog models. Thought-image filtering also has contract tests. |
| Embedding 2 | Text, multiple inputs, image, PDF, video, audio and combined inputs passed synthetic checks; no semantic-quality certification. |
| Live/Transcribe/Translate | Short synthetic audio flows passed; long sessions, reconnect and broader language coverage remain unverified. |
| Veo/Omni | Selected generation, first/last-frame, reference and extension operations passed; not every combination or output-quality property is certified. |
| Cyber / Robotics ER 2 | Restricted/early-access catalog entries, no successful live inference in this project. |

## Native services and partners

- Cache, batch and selected agent/session/memory resource workflows have live
  evidence, including temporary-resource cleanup. No deployed Agent Runtime is certified.
- Empty endpoint lifecycle passed. Project model deployment, registry mutations,
  complete RAG lifecycle and all tuning/evaluation workflows remain unverified or incomplete.
- GPT OSS has live evidence on an earlier artifact; GLM has mixed results.
  Mistral regional routes, FIM and OCR are implemented with offline contracts but
  all attempted operations returned 404. Regional errors identify publisher model/access;
  model availability versus project enablement is unresolved. Grok also returned 404.
- Partner catalog presence never implies certification of the model author's direct API.

## Evidence and release validation

[Historical integration archive](../releases/0.27.0-vertex-integration-history.json)
contains the original reports keyed by original relative path, including failures,
wheel digests, operation results and cleanup observations. Consolidation does not
upgrade those reports to release evidence for 0.27.0.

The manual Vertex matrix offers 33 targets. Run it against an installed wheel with
project credentials; reports must match the model, region, operations and exact
artifact. Protected publication gates remain separate and must pass in CI.

See [0.27.0 release validation](../releases/0.27.0-evidence.md) for current local
checks and artifact hashes, and the [Vertex guide](vertex.md) for API usage.
