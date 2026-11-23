# Using precomputed profiles

Start with the [bundled examples](../examples/released_profiles/README.md), or use a full profile bundle. Bundles contain gzip-compressed JSONL, CSV, and JSON files plus a manifest. They do not include source texts or sentence-level annotations.

The organizer selects stored profiles without rerunning annotation or changing their statistics. It does not fill missing profiles. Local use needs only Python's standard library. Profiles and catalogs use [CC BY 4.0](../LICENSE-DATA) for the project's own contributions; see [DATA.md](../DATA.md) for attribution and source terms.

## Local examples and bundles

Run from the repository root. List the available profiles:

```bash
python -m profile_data.organize \
  --source-dir examples/released_profiles --list
```

Select families, groups, and values into a new directory:

```bash
python -m profile_data.organize \
  --source-dir examples/released_profiles \
  --family base --family continuous_base \
  --group culturax_ar_cleaned --group culturax_ko_cleaned \
  --value fairness --value "personal autonomy" \
  --output-dir outputs/profile_examples
```

For a full bundle, give `--source-dir` the directory containing `manifest.json`. Repeat `--family`, `--group`, or `--value` to select several names, or omit an option to include all available entries. Names must match exactly, including group suffixes such as `_cleaned`; value names are not translated. Each run needs a new output directory.

`--list` reads only the manifest and accepts family/group filters. Value and support filters apply when extracting profiles. The organizer checks file sizes and hashes, decompresses selected files, and copies the catalogs. A new manifest records the selection and output hashes.

## Hugging Face release

The full bundle's public release is coming soon. The examples above are available now.

Install `huggingface_hub` to use a remote bundle with `--repo-id` instead of
`--source-dir`. Use `--revision` to choose a version. The organizer records the
resolved commit as `source_bundle.resolved_commit` in the output manifest.

Use the organizer or ordinary JSONL/CSV/JSON readers for these files. `datasets.load_dataset` does not read the bundle's custom manifest. See the [Hugging Face download guide](https://huggingface.co/docs/huggingface_hub/guides/download) for direct downloads.

## Families and native schemas

The `profile-bundle-v1` manifest lists file paths, families, groups, formats, row counts, sizes, SHA-256 hashes, value fields, and support fields. Full-bundle paths follow `data/<family>/<group>.<format>.gz`. Check the manifest for the files available in your version.

| Family | Format | Record meaning and principal fields |
| --- | --- | --- |
| `base` | JSONL | One value's categorical expression profile: `value_item`, `categories` containing names/counts/proportions, and `total_count` |
| `continuous_base` | JSONL | One value's Gaussian score summary: `value_item`, `mean`, `std`, `n_samples`, `n_unique_sentences`, and normalization/fit metadata |
| `context` | CSV | Categorical distribution comparison for `value_item`, `factor`, `state_a`, `state_b`; support `N_a`, `N_b`, OT and shift summaries |
| `continuous_context` | CSV | Normalized continuous-score comparison for the same value/factor/state-pair keys, support, and OT/shift summaries |
| `relations` | JSON | One group object: `attribute` and `value_relations`, mapping anchor values to nested `counts` and `distribution` objects |
| `context_relation_ot` | CSV | Joint relation-distribution comparison for `anchor_value`, factor, and state pair; `N_a`, `N_b` and joint OT summaries |
| `context_relation_counts` | CSV | Partner/relation counts for an anchor's factor/state; `support_N_state` is the state's support |
| `context_relation_deltas` | CSV | Partner/relation probability changes between an anchor's two factor states; `N_a`, `N_b` and delta summaries |

Each family keeps its own columns and nested fields. CSV readers return strings, so convert numeric fields as needed. Use the manifest's group label when rows have no group column. Catalogs describe the values, expression categories, and context factors; they are under `resources/` in full bundles and `resources/original/` in the code repository.

Base profiles have `total_count > 200`. Group and value coverage varies by family.

## Support filters

Support counts mean different things across families. Set a threshold for each family you select:

```bash
python -m profile_data.organize \
  --source-dir examples/released_profiles \
  --family base --family context --group culturax_ko_cleaned \
  --min-support base=500 --min-support context=50 \
  --output-dir outputs/supported_profiles
```

| Family | Field tested by `--min-support FAMILY=N` |
| --- | --- |
| `base` | `total_count` |
| `continuous_base` | `n_unique_sentences` |
| `context`, `continuous_context`, `context_relation_ot`, `context_relation_deltas` | Both `N_a` and `N_b` |
| `context_relation_counts` | `support_N_state` |
| `relations` | Not supported: no single support count |

Filtering keeps the stored distributions and uncertainty estimates unchanged. It does not rebalance groups or rebuild relations. In relation families, `--value` selects anchors and keeps all their partners.

## Interpretation and rights

Counts refer to the records used in each aggregate, not independent people. In continuous profiles, `n_samples` can repeat a sentence score across factor/state records; `n_unique_sentences` counts distinct sentences. Neither guarantees independent or representative observations. Continuous scores depend on the model and normalization, and Gaussian summaries are not categorical probabilities. Context comparisons show associations, not causal effects.

The examples illustrate formats and language/value coverage. A full bundle contains the available aggregates for that version, not every possible group–value combination.

Before sharing profiles, read [DATA.md](../DATA.md) and the bundle's source notices. Removing source text does not remove the source's rights or terms.
