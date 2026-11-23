# Data, licenses, and source terms

## Data scope

Users supply their own prepared CSV/JSONL corpora. No Reddit or CulturaX source
records, row-level annotations, or model weights are included.

The current workflow starts with locally prepared data obtained through
authorized access to the relevant source. Source-specific acquisition and
preparation instructions are coming soon.

Small actual aggregate examples are provided in
[`examples/released_profiles/`](examples/released_profiles/README.md); they illustrate
formats and language/value coverage, not a statistically representative sample.
Full aggregate bundles are separate from the code repository. The
[`profile_data.organize` guide](docs/precomputed_profiles.md) covers local selection
and optional Hugging Face retrieval.

## Source-data licenses and terms

These source notices do not grant a license to this project's code, catalogs, or
aggregate profiles. Applicable obligations depend on the source version and
access method. Consult the upstream notices for complete terms; source references
were checked on 2026-10-13.

### Reddit

Reddit contributors retain ownership rights in their content. Public visibility
is not a general open-dataset license. Consult the [Reddit User Agreement](https://redditinc.com/policies/user-agreement)
and [Public Content Policy](https://support.reddithelp.com/hc/en-us/articles/26410290525844-Public-Content-Policy).

For API-based access, the [Data API Terms](https://redditinc.com/policies/data-api-terms)
provide limited permissions, impose use and retention restrictions, and do not
grant general-purpose reuse or model-training rights. Follow the terms and any
separate agreement applicable to your access. This repository does not distribute
Reddit posts or comments or grant rights to them.

### CulturaX, mC4, and OSCAR

The [CulturaX dataset card](https://huggingface.co/datasets/uonlp/CulturaX#license-information)
states that its license terms follow its mC4 and OSCAR sources. CulturaX is not
assigned a blanket permissive license by this repository.

- **mC4:** The [C4/mC4 licensing notice](https://huggingface.co/datasets/allenai/c4#licensing-information)
  identifies [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/) and also
  refers to the [Common Crawl terms](https://commoncrawl.org/terms-of-use) for
  the underlying content.
- **OSCAR:** The [OSCAR 23.01 licensing notice](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301#licensing-information)
  applies [CC0](https://creativecommons.org/publicdomain/zero/1.0/) to packaging,
  metadata, and annotations, not to the crawled source texts. Consult the notices
  for the OSCAR versions in your data and their current access conditions.

We acknowledge the CulturaX authors, the OSCAR project and contributors,
Common Crawl, Google, and the Allen Institute for AI. Obtain source data from
authorized upstream providers and retain required attribution and notices.
No CulturaX, mC4, or OSCAR source records are redistributed here.

## Project licensing status

Copyright (c) 2026 Woojin Kim and Jaeyoung Do.

- **Software:** the project's code and accompanying usage documentation are
  licensed under the [MIT License](LICENSE), except for the data, catalogs,
  and visual assets identified below. Existing third-party notices apply to
  their respective components.
- **Profiles and catalogs:** aggregate data and metadata under
  `examples/released_profiles/`, constructed catalog payloads under
  `resources/original/`, and the corresponding full HF profile/catalog bundle
  are licensed under [CC BY 4.0](LICENSE-DATA) for the project's own copyright
  and database contributions. Suggested credit: Woojin Kim and Jaeyoung Do,
  *Distributional Value Profiling*, with a link to the project or dataset.
- **Paper visuals:** the paper figures and illustrative animation under
  `docs/assets/figures/` and `docs/assets/animations/` retain their existing
  rights; neither license above grants additional rights to those assets.

The CC BY grant covers only rights the project owns or is authorized to grant.
It does not license underlying Reddit/CulturaX records or extend rights held by
third parties. Source-specific redistribution questions remain separate,
including unresolved attribution and terms for additional Korean sources in
the full bundle. Third-party model and dependency licenses also remain separate.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the ValueBench-derived
model adapter's upstream MIT notice.

## Local inputs and generated outputs

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

Retain third-party copyright and license notices when distributing material you
are authorized to share.
