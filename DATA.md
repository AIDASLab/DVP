# Data, licenses, and source terms

## Data scope

Build new profiles from your own prepared CSV/JSONL files. Source-specific
download and preparation instructions are coming soon. Obtain data through
authorized access to the source.

The repository includes small [aggregate examples](examples/released_profiles/README.md)
for exploring the formats. They are not a representative sample. Full profiles
are distributed separately; see the [profile guide](docs/precomputed_profiles.md).
Neither package includes Reddit/CulturaX source records, sentence-level
annotations, or model weights.

## Source-data licenses and terms

Use the terms for your source version and access method. The upstream notices
below are separate from the project licenses listed later on this page.

### Reddit

Reddit contributors own their content. Public posts are not automatically
licensed for reuse. Read the [Reddit User Agreement](https://redditinc.com/policies/user-agreement)
and [Public Content Policy](https://support.reddithelp.com/hc/en-us/articles/26410290525844-Public-Content-Policy).

For API-based access, the [Data API Terms](https://redditinc.com/policies/data-api-terms)
restrict use and retention and do not grant general-purpose reuse or
model-training rights. Follow those terms and any separate agreement that
applies to your access.

### CulturaX, mC4, and OSCAR

The [CulturaX dataset card](https://huggingface.co/datasets/uonlp/CulturaX#license-information)
states that its terms follow the mC4 and OSCAR sources. This repository does not
grant a separate license to the underlying text.

- **mC4:** The [C4/mC4 licensing notice](https://huggingface.co/datasets/allenai/c4#licensing-information)
  identifies [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/) and also
  refers to the [Common Crawl terms](https://commoncrawl.org/terms-of-use) for
  the underlying content.
- **OSCAR:** The [OSCAR 23.01 licensing notice](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301#licensing-information)
  applies [CC0](https://creativecommons.org/publicdomain/zero/1.0/) to packaging,
  metadata, and annotations, not the crawled texts. Check the terms and access
  conditions for the OSCAR version you use.

We acknowledge the CulturaX authors, the OSCAR project and contributors,
Common Crawl, Google, and the Allen Institute for AI. Retain the attribution and
notices required by your data sources.

## Project licensing status

Copyright (c) 2026 Woojin Kim and Jaeyoung Do.

- **Software:** code and usage documentation use the [MIT License](LICENSE),
  except for the data, catalogs, and paper visuals listed below. Keep the
  existing third-party notices.
- **Profiles and catalogs:** aggregate data and metadata under
  `examples/released_profiles/`, constructed catalog files under
  `resources/original/`, and the corresponding full HF profile/catalog bundle
  use [CC BY 4.0](LICENSE-DATA) for the project's own copyright and database
  contributions. Suggested credit: Woojin Kim and Jaeyoung Do,
  *Distributional Value Profiling*, with a link to the project or dataset.
- **Paper visuals:** the paper figures and illustrative animation under
  `docs/assets/figures/` and `docs/assets/animations/` retain their existing
  rights and are not covered by these two licenses.

CC BY covers only rights the project owns or may grant. Source-corpus rights
remain separate, including unresolved attribution and terms for the additional
Korean sources in the full bundle. Models and dependencies keep their own licenses.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the ValueBench-derived
model adapter's upstream MIT notice.

## Local inputs and generated outputs

Place local inputs under `data/` and results under `outputs/`.
Both are ignored by Git. See [the workflow](docs/workflow.md) for file formats.

The mapping stage updates its mapping file in place, creating one if needed.
Use a copy to preserve an existing dictionary, and avoid concurrent writes.

The catalogs in `resources/original/` provide value names, definitions,
expression categories, and context factors. The pipeline reads these files
without changing them. Replacement catalogs must use consistent names and schemas.

Generated files can contain sentence text, identifiers, and local paths. They
are not anonymized automatically. Check data terms, privacy requirements, and
redistribution rights before sharing them, and retain required copyright and
license notices.
