# Value and context catalogs

| File | Contents |
| --- | --- |
| `unique_value_items.json` | 1,946 canonical value names |
| `unique_value_item_definitions.json` | Definitions for 1,515 values |
| `surface_value_regimes_gpt.jsonl` | Ordered expression categories for each value |
| `context_factors_revised.json` | Default value-specific context factors and states |

The launchers select `context_factors_revised.json` by default. Use the same
chosen factor catalog for annotation, aggregation, and assembly.

Definition coverage is a subset of the value list. By default, embedding matching
uses the definition keys as candidates, and continuous scoring uses values
present in both the definitions and vector index. Values without projection
scores can still contribute to categorical profiles when their catalogs exist.
The coverage checker reports this coverage separately.

Set `REPRESENTATIVES_JSON` to a compatible value-name list or definition
dictionary to change the embedding candidate vocabulary independently.

Value, category, and factor names are matched exactly. Supply compatible
replacement files through `VALUE_ITEMS_JSON`, `VALUE_DEFS_JSON`,
`REGIMES_JSONL`, and `FACTORS_JSON`; individual Python modules also accept
catalog paths. These files are not source-text datasets.
