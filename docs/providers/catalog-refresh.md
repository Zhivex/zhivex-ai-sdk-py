# Catalog freshness checks

The catalog contains source-backed metadata and verification dates. These dates
record when the metadata was checked; they do not certify a live provider call or
prove that no newer model exists.

The default command keeps the pricing expiry contract: JSON contains `as_of` and
`alerts`, and exits with status 1 when pricing is expired or approaches expiry
within the configured window. No provider requests are made.

```bash
.venv/bin/python scripts/check_catalog_freshness.py --within-days 30
```

Opt in to metadata age checks for the currently refreshed direct providers:

```bash
.venv/bin/python scripts/check_catalog_freshness.py \
  --as-of 2026-09-30 \
  --metadata-max-age-days 30 \
  --provider openai --provider anthropic --provider gemini --provider qwen
```

This adds `metadata_alerts` to the JSON. An entry is `missing` when `verified_at`
is absent, `future` when it is later than `as_of`, and `stale` when its age exceeds
the maximum. The threshold is inclusive: exactly 30 days old passes a 30-day
limit. Retired models are excluded. Repeating `--provider` filters both pricing
and metadata alerts; leaving the filter out includes all catalog providers.
Status 1 means at least one pricing or metadata alert; status 0 means none. Invalid
options return status 2. `--metadata-max-age-days 0` flags any earlier date.

The check detects age and inconsistent dates, **not new upstream releases, API
schema changes, retired aliases or missing model IDs**. It also cannot establish
that source links still describe the same behavior. Review the provider's primary
changelog, model reference, pricing and request schema; add a contract regression
for observable changes; then update the affected entry's metadata and date.
Never refresh dates in bulk just to pass this check.

Azure and Bedrock catalog data are unchanged by this work. Filtering the command
does not edit any catalog entries or change support classifications. Offline
contract coverage and live certification remain separate evidence.
