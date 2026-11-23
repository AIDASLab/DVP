# Value and context catalogs

| File | Contents |
| --- | --- |
| `unique_value_items.json` | 1,946 canonical value names |
| `unique_value_item_definitions.json` | Definitions for 1,515 values |
| `surface_value_regimes_gpt.jsonl` | Ordered expression categories for each value |
| `context_factors_revised.json` | Default value-specific context factors and states |

The launchers use `context_factors_revised.json` by default. Keep the same factor
catalog across annotation, aggregation, and assembly.

Not every value has a definition. Embedding matching uses definition keys by
default, and continuous scoring needs both a definition and a value vector.
Values without projection scores can still have categorical profiles if their
catalog entries exist. The coverage checker reports these differences.

Use `REPRESENTATIVES_JSON` to choose a different value-name list or definition
dictionary for embedding matching.

Names must match exactly across stages. Set `VALUE_ITEMS_JSON`, `VALUE_DEFS_JSON`,
`REGIMES_JSONL`, and `FACTORS_JSON` to use replacement catalogs. Individual Python
modules also accept catalog paths. These files contain catalogs, not source texts.

## License

The catalogs use [CC BY 4.0](../../LICENSE-DATA).
Credit Woojin Kim and Jaeyoung Do and link to the project. Source-corpus and
third-party rights remain separate; see [DATA.md](../../DATA.md#project-licensing-status).
