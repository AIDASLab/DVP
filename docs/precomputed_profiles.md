# Using precomputed profiles

The code repository includes small [aggregate examples](../examples/released_profiles/README.md). Full aggregates are packaged separately as a manifest plus gzip-compressed JSONL, CSV, and JSON files. Neither package contains source texts or row-level annotations.

The organizer selects existing records; it does not run annotation, recompute statistics, or fill missing profiles. Local use requires only Python's standard library. Dataset licenses remain unassigned and publication is pending source-specific rights review.

## Local examples and bundles

Run from the code repository root. List available files before selecting:

```bash
python -m profile_data.organize \
  --source-dir examples/released_profiles --list
```

Extract selected families, groups, and values into a new output directory:

```bash
python -m profile_data.organize \
  --source-dir examples/released_profiles \
  --family base --family continuous_base \
  --group culturax_ar_cleaned --group culturax_ko_cleaned \
  --value fairness --value "personal autonomy" \
  --output-dir outputs/profile_examples
```

For a full local bundle, replace the source path with its directory containing `manifest.json`, for example `/path/to/profile-bundle`. Omit `--family`, `--group`, or `--value` to leave that dimension unrestricted. Repeated options select multiple exact names; value names are not fuzzy-matched or translated. Native group IDs retain suffixes such as `_cleaned`. Use a new output directory for each selection.

`--list` reads the manifest only and accepts family/group filters, not value/support filters. Extraction checks selected files against their recorded sizes and hashes. Outputs keep the family/group layout without `.gz`, copy the supplied catalogs, and include a new `manifest.json` with output hashes and selection provenance.

## Download from Hugging Face

Remote access additionally requires:

```bash
python -m pip install huggingface_hub
```

After a bundle has been published, replace `YOUR_ORG/YOUR_DATASET` with its actual dataset repository ID:

```bash
python -m profile_data.organize \
  --repo-id YOUR_ORG/YOUR_DATASET \
  --revision main \
  --cache-dir outputs/profile_cache \
  --family base --group culturax_ko_cleaned --value fairness \
  --output-dir outputs/korean_fairness
```

The repository ID above is a placeholder, not an announced or uploaded dataset. `--source-dir` and `--repo-id` are alternatives. Pin `--revision` to a commit for a fixed remote version; the organizer resolves remote revisions to a commit and records it as `source_bundle.resolved_commit` in the output manifest.

The bundle is an archive of native aggregate formats, not a single tabular Hugging Face dataset configuration. `datasets.load_dataset` does not interpret this custom manifest; use the organizer or ordinary JSONL/CSV/JSON readers. Direct file retrieval is also covered by the [official Hugging Face download guide](https://huggingface.co/docs/huggingface_hub/guides/download).

## Families and native schemas

Manifest schema `profile-bundle-v1` has a `files` list. Each entry identifies its path, `family`, `group`, `format`, row count, byte size, SHA-256 digest, `value_field`, and `support_fields`. Full bundles store data as `data/<family>/<group>.<format>.gz`. Consult the manifest for the exact files and coverage in your version.

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

Native columns and nested fields are retained rather than converted to one shared schema. CSV cells are serialized values; use the column names and family schema when converting numeric fields. The group in the manifest identifies the source file even when individual rows have no group column. Bundled catalogs under `resources/` describe values, expression categories, and context factors; the code repository also provides catalogs under `resources/original/`.

The full bundle uses the available filtered base profiles with `total_count > 200`. Family coverage differs: a group's presence in one family does not imply that every family or value exists for that group.

## Support filters

There is no universal support denominator. Request an explicit threshold for each selected family, for example:

```bash
python -m profile_data.organize \
  --source-dir /path/to/profile-bundle \
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
| `relations` | Unsupported; no single support field is substituted |

Filtering does not rebalance groups, renormalize distributions, recompute uncertainty, or rebuild relations. For `relations`, `--value` selects **anchor values only** and preserves all of each retained anchor's partners. Joint relation families also select by anchor, not partner.

## Interpretation and rights

Counts describe records used by the corresponding aggregate, not independent people. `n_samples` can include repeated sentence scores across factor/state records; it is not interchangeable with `n_unique_sentences`. Even sentence counts do not establish independent observations or representative population sampling. Continuous scores depend on their model and normalization; Gaussian summaries are not categorical probabilities. Context comparisons describe observed associations, not identified causal effects.

Examples provide illustrative language/value coverage, not a statistically representative sample. Full means all available aggregates in the selected bundle version, not complete coverage of every source, value, or family.

Read each bundle's `NOTICE.md` and source-specific metadata before redistribution. Original corpus access and aggregate redistribution are separate questions; removing source texts does not establish an automatic exemption from applicable rights or terms.
