"""Stable application-owned metadata; the bundled default snapshot remains Beta."""
from zhivex_ai import ModelCatalogEntry, ModelPricing, create_model_catalog


def main() -> None:
    # Synthetic application-reviewed rates, not current provider pricing.
    catalog = create_model_catalog([
        ModelCatalogEntry(
            provider="example", model_id="reviewed-model-v1", aliases=("reviewed",),
            pricing=ModelPricing(
                currency="USD", source_url="https://example.com/pricing",
                input_per_1m_tokens=2, output_per_1m_tokens=8,
            ),
        ),
    ])
    entry = catalog.find("example", "reviewed")
    assert entry is not None and entry.pricing is not None
    print(entry.model_id, entry.pricing.conservative_cost_per_1k_tokens())
    print([item.model_id for item in catalog.list()])


if __name__ == "__main__":
    main()
