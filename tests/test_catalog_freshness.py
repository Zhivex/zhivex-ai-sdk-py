from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
from datetime import date
from io import StringIO
import json
from unittest import TestCase
from unittest.mock import patch

from scripts.check_catalog_freshness import main, metadata_alerts, pricing_alerts
from zhivex_ai.catalog import ModelCatalog, ModelCatalogEntry, ModelPricing


class CatalogFreshnessTests(TestCase):
    def setUp(self):
        self.as_of = date(2026, 9, 30)
        self.catalog = ModelCatalog([
            ModelCatalogEntry("openai", "missing"),
            ModelCatalogEntry("openai", "future", verified_at="2026-10-01"),
            ModelCatalogEntry("openai", "boundary", verified_at="2026-08-31"),
            ModelCatalogEntry("gemini", "stale", verified_at="2026-08-30"),
            ModelCatalogEntry("gemini", "current", verified_at="2026-09-30"),
            ModelCatalogEntry("openai", "retired", availability="retired"),
            ModelCatalogEntry("anthropic", "priced", verified_at="2026-09-30", pricing=ModelPricing(
                currency="USD", input_per_1m_tokens=1.0, output_per_1m_tokens=2.0,
                source_url="https://example.com/prices", effective_until="2026-10-10",
            )),
        ])

    def test_missing_future_and_age_boundary_excluding_retired(self):
        alerts = metadata_alerts(self.catalog, as_of=self.as_of, max_age_days=30)
        by_id = {alert["model_id"]: alert for alert in alerts}
        self.assertEqual(set(by_id), {"missing", "future", "stale"})
        self.assertEqual(by_id["missing"]["status"], "missing")
        self.assertIsNone(by_id["missing"]["verified_at"])
        self.assertEqual(by_id["future"]["age_days"], -1)
        self.assertEqual(by_id["future"]["status"], "future")
        self.assertEqual(by_id["stale"]["age_days"], 31)
        self.assertEqual(by_id["stale"]["status"], "stale")
        filtered = metadata_alerts(self.catalog, as_of=self.as_of, max_age_days=30, providers={"gemini"})
        self.assertEqual([alert["model_id"] for alert in filtered], ["stale"])
        with self.assertRaises(ValueError):
            metadata_alerts(self.catalog, as_of=self.as_of, max_age_days=-1)

    def run_cli(self, *args):
        output = StringIO()
        with patch("scripts.check_catalog_freshness.default_model_catalog", self.catalog), patch("sys.argv", ["freshness", "--as-of", "2026-09-30", *args]), redirect_stdout(output):
            status = main()
        return status, json.loads(output.getvalue())

    def test_default_json_and_pricing_behavior_unchanged(self):
        status, report = self.run_cli()
        self.assertEqual(status, 1)
        self.assertEqual(report, {"as_of": "2026-09-30", "alerts": pricing_alerts(self.catalog, as_of=self.as_of)})
        status, report = self.run_cli("--within-days", "0")
        self.assertEqual(status, 0)
        self.assertEqual(report, {"as_of": "2026-09-30", "alerts": []})

    def test_opt_in_and_repeatable_provider_filters(self):
        status, report = self.run_cli("--metadata-max-age-days", "30", "--provider", "openai", "--provider", "gemini")
        self.assertEqual(status, 1)
        self.assertEqual(report["alerts"], [])
        self.assertEqual(len(report["metadata_alerts"]), 3)
        status, report = self.run_cli("--metadata-max-age-days", "0", "--within-days", "0", "--provider", "anthropic")
        self.assertEqual(status, 0)
        self.assertEqual(report["metadata_alerts"], [])
        status, report = self.run_cli("--provider", "openai")
        self.assertEqual(status, 0)
        self.assertNotIn("metadata_alerts", report)

    def test_cli_rejects_negative_threshold_and_unknown_provider(self):
        for args in (("--metadata-max-age-days", "-1"), ("--provider", "not-a-provider")):
            with self.subTest(args=args), redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
                self.run_cli(*args)
            self.assertEqual(raised.exception.code, 2)
