from zhivex_ai import default_model_catalog


def main() -> None:
    entry = default_model_catalog.find("openai", "gpt-5.6")
    print(entry)

    for item in default_model_catalog.list():
        print(item.provider, item.model_id, item.recommended_for)


if __name__ == "__main__":
    main()
