"""Report pricing expiry and optionally metadata age without provider requests.

Metadata age checks do not discover upstream model releases or API changes.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
from typing import Any

from zhivex_ai.catalog import ModelCatalog, default_model_catalog


def pricing_alerts(
    catalog: ModelCatalog, *, as_of: date, within_days: int = 30
) -> list[dict[str, str]]:
    alerts = []
    for entry in catalog.list():
        if (
            entry.availability == "retired"
            or not entry.pricing
            or not entry.pricing.effective_until
        ):
            continue
        expiry = date.fromisoformat(entry.pricing.effective_until)
        if expiry <= as_of + timedelta(days=within_days):
            alerts.append(
                {
                    "provider": entry.provider,
                    "model_id": entry.model_id,
                    "status": "expired" if expiry < as_of else "expiring",
                    "effective_until": expiry.isoformat(),
                    "source_url": entry.pricing.source_url,
                }
            )
    return alerts


def metadata_alerts(
    catalog: ModelCatalog, *, as_of: date, max_age_days: int,
    providers: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Flag missing, future, or old verification dates; never assert freshness."""
    if max_age_days < 0:
        raise ValueError("max_age_days must be non-negative")
    alerts: list[dict[str, Any]] = []
    for entry in catalog.list():
        if entry.availability == "retired" or (providers is not None and entry.provider not in providers):
            continue
        alert: dict[str, Any] = {
            "provider": entry.provider,
            "model_id": entry.model_id,
            "verified_at": entry.verified_at,
        }
        if entry.verified_at is None:
            alert["status"] = "missing"
        else:
            verified = date.fromisoformat(entry.verified_at)
            age_days = (as_of - verified).days
            if age_days < 0:
                alert["status"] = "future"
            elif age_days > max_age_days:
                alert["status"] = "stale"
            else:
                continue
            alert["age_days"] = age_days
        alerts.append(alert)
    return alerts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=date.fromisoformat, default=None)
    parser.add_argument("--within-days", type=int, default=30)
    parser.add_argument(
        "--metadata-max-age-days", type=int, default=None,
        help="Opt in to verification-date age checks; does not discover releases.",
    )
    parser.add_argument(
        "--provider", action="append", default=None,
        choices=sorted({entry.provider for entry in default_model_catalog.list()}),
        help="Filter alerts by provider; repeat to include multiple providers.",
    )
    args = parser.parse_args()
    if args.within_days < 0:
        parser.error("--within-days must be non-negative")
    if args.metadata_max_age_days is not None and args.metadata_max_age_days < 0:
        parser.error("--metadata-max-age-days must be non-negative")
    providers = set(args.provider) if args.provider is not None else None
    as_of = args.as_of or date.today()
    alerts = pricing_alerts(
        default_model_catalog, as_of=as_of, within_days=args.within_days
    )
    if providers is not None:
        alerts = [alert for alert in alerts if alert["provider"] in providers]
    report: dict[str, Any] = {"as_of": as_of.isoformat(), "alerts": alerts}
    metadata = []
    if args.metadata_max_age_days is not None:
        metadata = metadata_alerts(
            default_model_catalog, as_of=as_of,
            max_age_days=args.metadata_max_age_days, providers=providers,
        )
        report["metadata_alerts"] = metadata
    print(json.dumps(report, indent=2))
    return int(bool(alerts or metadata))


if __name__ == "__main__":
    raise SystemExit(main())
