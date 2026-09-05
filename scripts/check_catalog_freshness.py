"""Report upcoming/expired catalog pricing without making provider requests."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=date.fromisoformat, default=None)
    parser.add_argument("--within-days", type=int, default=30)
    args = parser.parse_args()
    if args.within_days < 0:
        parser.error("--within-days must be non-negative")
    as_of = args.as_of or date.today()
    alerts = pricing_alerts(
        default_model_catalog, as_of=as_of, within_days=args.within_days
    )
    print(json.dumps({"as_of": as_of.isoformat(), "alerts": alerts}, indent=2))
    return int(bool(alerts))


if __name__ == "__main__":
    raise SystemExit(main())
