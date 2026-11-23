# Aggregate profile examples

Small, native-format examples for inspecting schemas and exercising the profile organizer without downloading the full bundle. These are actual aggregate records, not synthetic data. The separate `examples/profiles/run_smoke.py` remains the synthetic pipeline check.

The base and continuous-base examples cover **fairness**, **personal autonomy**, and **public safety** across seven CulturaX language groups: Arabic, German, French, Japanese, Korean, Russian, and Chinese. Each family contains 21 profiles.

Each context and joint-relation family includes one fairness record per group (seven records per family). Relation JSON files retain one fairness anchor per group with all of that anchor's partners. These selections illustrate formats and language/value coverage; they are **not a statistically representative sample**, a benchmark split, or the full profiles for those values. The manifest records the exact selection and file counts.

The 56 aggregate files occupy 119,027 bytes (about 116 KiB), excluding the manifest and this README. Counts of records across these different families are not counts of independent observations.

From the code repository root:

```bash
python -m profile_data.organize \
  --source-dir examples/released_profiles --list

python -m profile_data.organize \
  --source-dir examples/released_profiles \
  --family base --group culturax_ko_cleaned --value fairness \
  --output-dir outputs/fairness_example
```

Use a new output directory. See [the precomputed-profile guide](../../docs/precomputed_profiles.md) for native schemas, filters, support denominators, and full-bundle access.

## Source and license

These examples derive from [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX)
and contain aggregate statistics, not source texts or row-level annotations.
See [CulturaX, mC4, and OSCAR source terms](../../DATA.md#culturax-mc4-and-oscar).
The aggregate profiles and their metadata are licensed under
[CC BY 4.0](../../LICENSE-DATA) for the project's own contributions. Credit
Woojin Kim and Jaeyoung Do and link to the project. See the
[project licensing scope](../../DATA.md#project-licensing-status).
This license does not grant rights to the underlying corpus.
