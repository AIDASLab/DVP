# Input data and generated outputs

Users supply their own prepared CSV/JSONL corpora. No Reddit or CulturaX source
records, row-level annotations, or model weights are included. Source-corpus
acquisition is outside this package.

Small actual aggregate examples are provided in
[`examples/released_profiles/`](examples/released_profiles/README.md); they illustrate
formats and language/value coverage, not a statistically representative sample.
Full aggregate bundles are separate from the code repository. The
[`profile_data.organize` guide](docs/precomputed_profiles.md) covers local selection
and optional Hugging Face retrieval. The aggregate examples and full bundle have no assigned
license and require source-specific rights review before publication. Aggregation
and omission of source texts do not automatically establish redistribution rights.

Place local inputs under `data/` and generated artifacts under `outputs/`.
Both are ignored by Git. See [the workflow](docs/workflow.md) for file formats.

The mapping stage accepts an existing writable mapping file or creates one from
embedding matches to the chosen vocabulary. It updates this file in place; use a
working copy if you need to preserve an existing dictionary. Do not run concurrent
writers against the same mapping file.

The catalogs in `resources/original/` provide value names, definitions,
expression categories, and context factors. Catalogs are read-only pipeline
inputs. If substituting catalogs, keep names and schemas consistent across stages.

Generated files may retain sentence text, identifiers, and local paths. They are
not automatically anonymized or suitable for publication. Check input-data
terms, privacy requirements, and redistribution rights before sharing outputs.
Bundled catalogs do not grant rights to source corpora or third-party models.

Third-party copyright and license notices in the source must be retained.
