# New-user timing protocol

HU14 requires an external participant unfamiliar with this SDK. Machine smoke
latency and maintainer completion time cannot substitute for this observation.

Prepare a supported Python installation, provider access and a dedicated Postgres
database. Record preparation separately; include SDK installation and tutorial
steps in the measured interval. Do not pre-generate the app or cache answers.

Record an anonymous participant ID, date, wheel SHA256/version, operating system,
Python version and documentation commit. Start the clock when the participant
opens the guide. Record time of first successful provider response and first
successful approval/resume after process restart. Targets: <300 and <900 seconds.
Record every intervention and blocker without prompts, keys, DSNs or client data.
An assisted run does not establish the unassisted target. Repeat after fixing blockers.

Suggested record (no completed participant is implied):

```json
{"schema_version":1,"participant_id":null,"wheel_sha256":null,"docs_commit":null,
 "first_response_seconds":null,"durable_resume_seconds":null,
 "interventions":[],"status":"not_measured","publication_approved":false}
```

The participant approves publication separately. Keep private observations in the
team's restricted workspace. Only sanitized aggregate results belong in the repo.
