# Aggregate profile examples

Use these small examples to explore profile formats and try the organizer before downloading the full bundle. They contain real aggregate records. For a synthetic pipeline check, use `examples/profiles/run_smoke.py`.

The base and continuous-base examples cover **fairness**, **personal autonomy**, and **public safety** across seven CulturaX language groups: Arabic, German, French, Japanese, Korean, Russian, and Chinese. Each family contains 21 profiles.

Each context and joint-relation family has one fairness record per group (seven per family). Relation JSON files keep one fairness anchor per group and all its partners. These selections show the formats and language/value coverage; they are not a representative sample, benchmark split, or complete profiles for those values. See the manifest for exact selections and counts.

The 56 data files take 119,027 bytes (about 116 KiB), excluding the manifest and README. Their record counts use different units and should not be added as independent observations.

From the code repository root:

```bash
python -m profile_data.organize \
  --source-dir examples/released_profiles --list

python -m profile_data.organize \
  --source-dir examples/released_profiles \
  --family base --group culturax_ko_cleaned --value fairness \
  --output-dir outputs/fairness_example
```

Use a new output directory each time. See the [profile guide](../../docs/precomputed_profiles.md) for schemas, support filters, and full-bundle access.

## Source and license

These examples derive from [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX)
and contain statistics, not source texts or sentence-level annotations.
See [CulturaX, mC4, and OSCAR source terms](../../DATA.md#culturax-mc4-and-oscar).
Profiles and metadata use [CC BY 4.0](../../LICENSE-DATA) for the project's own
contributions. Credit Woojin Kim and Jaeyoung Do and link to the project.
The license does not cover the underlying corpus; see the
[license scope](../../DATA.md#project-licensing-status).
