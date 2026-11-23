# Distributional value profiling

Tools for preparing text, annotating values, constructing value profiles, and
working with precomputed aggregates. Profiles describe categorical value
expressions, continuous value scores, relations between values, and their
variation across contexts.

You can use existing profiles without rerunning annotation, or build profiles
from your own prepared text. Annotation models, intensity models, catalogs, and
processing settings are configurable.

![Overview of Distributional Value Profiling: extract value signals from text, construct distributions, predict profiles, and steer model outputs.](docs/assets/figures/framework.png)

Overview of the paper's framework. The current code release covers data
processing, annotation, and profile construction; profiling-model training and
steering code are coming soon.

## Project page

The [project-page source](docs/index.html) includes the paper overview, figures,
findings, and citation. The hosted page is coming soon; see the
[site guide](docs/README.md) for local preview.

## What is included

| Component | Location / availability |
| --- | --- |
| Data preparation, annotation, and profile construction | Code in this repository; see the [workflow guide](docs/workflow.md) |
| Small actual aggregate examples | [Bundled examples](examples/released_profiles/README.md), about **0.15 MiB** including their manifest and documentation |
| Full aggregate profiles | Coming soon on Hugging Face; about **779 MiB compressed** |
| Profile download and selection | [`profile_data.organize`](profile_data/organize.py); see the [precomputed-profile guide](docs/precomputed_profiles.md) |
| Constructed value and context catalogs | [`resources/original/`](resources/original/README.md) |

The full bundle contains eight aggregate families across the union of 24 source
groups: categorical and continuous base profiles, categorical and continuous
context comparisons, value relations, and context-conditioned relation
distances, counts, and probability changes. Coverage differs by family; consult
the bundle manifest rather than assuming every group or value is available.

The bundled examples illustrate all eight families across seven CulturaX language
groups. Base examples cover fairness, personal autonomy, and public safety;
the other families use small selections described in their manifest. These are
illustrative examples, not a statistically representative sample or a benchmark
split.

Here, **full profiles means aggregate statistics, not original corpus records**.
Source texts, row-level annotations, activation arrays, and model weights are
not included. See [Data and licenses](#data-and-licenses) for source terms and
the project licensing status.

## Use precomputed profiles

Run commands from this repository's root. Local profile selection requires only
Python 3.10+ and its standard library; no model, GPU, or inference dependencies
are needed.

### Try the bundled examples

List available families and groups:

```bash
python -m profile_data.organize \
  --source-dir examples/released_profiles --list
```

Select a categorical profile using an explicit support threshold:

```bash
python -m profile_data.organize \
  --source-dir examples/released_profiles \
  --family base --group culturax_ko_cleaned --value fairness \
  --min-support base=500 \
  --output-dir outputs/profile_examples
```

This writes the selected records to
`outputs/profile_examples/data/base/culturax_ko_cleaned.jsonl` and records the
selection and output checksums in `outputs/profile_examples/manifest.json`.
Use a new output directory for each run; existing directories are not overwritten.

### Use the full bundle

The full aggregate bundle is coming soon on Hugging Face. For an existing local
bundle, pass its directory containing `manifest.json` to `--source-dir`. The same
selection commands work on the full bundle and the examples.

The organizer verifies selected files against the manifest's sizes and SHA-256
checksums. Value and support filtering happen locally. Repeat `--family`,
`--group`, or `--value` to select multiple names; omit an option to leave that
dimension unrestricted.

Support thresholds are family-specific: `base=500` tests `total_count`, while
`context=30` requires both context-state counts to be at least 30. Filtering
preserves stored statistics; it does not renormalize distributions, rebalance
groups, or recompute uncertainty. Relation value filters select anchors and
retain their partners.

The full bundle uses native gzip-compressed JSONL, CSV, and JSON files, not one
unified `datasets.load_dataset` configuration. See the
[precomputed-profile guide](docs/precomputed_profiles.md) for all family schemas,
support fields, and access options.

## Build profiles from your own data

<details>
<summary>Watch the profile-construction walkthrough (15 seconds)</summary>

![Animated walkthrough from group text to value identification, expression/intensity/relation signals, and an aggregated profile.](docs/assets/animations/profile-construction.gif)

Illustrative example. Sentences, annotations, scores, and distributions are
schematic rather than actual annotation outputs or empirical results.

</details>

Use Python 3.10+ and a POSIX environment with Bash for the shell launchers.
Install the CPU processing and aggregation dependencies:

```bash
python -m pip install -r requirements.txt
```

Optional continuous-profile plots additionally require `matplotlib`
(`python -m pip install matplotlib`).

For annotation and activation scoring, also install the model dependencies in a
CUDA environment compatible with your chosen models:

```bash
python -m pip install -r requirements-inference.txt
```

The pipeline connects expressive-text selection, value extraction and mapping,
expression/relation/context annotation, activation-based intensity scoring, and
profile aggregation. Use the supplied catalogs or compatible replacements. You
can run stages independently with existing annotations or vectors that match
the documented formats.

| Stage | Location |
| --- | --- |
| Text preparation and expressive selection | `processing/` |
| Value extraction, mapping, and contrastive examples | `value_item_parser/` |
| Expression, relation, and context annotation | `intensity_parser/`, `relation_parser/`, `context_parser/` |
| Activations, value vectors, and projection scores | `activation_parser/` |
| Base distributions | `profile_constructor/` |
| Context effects | `context_parser/context_effect_*.py` |
| Relation distributions and profile assembly | `value_profiler/` |

Follow the [workflow guide](docs/workflow.md) for prepared-data formats, commands,
configuration, and output paths. Python command-line scripts accept `--help`; the shell
launchers also support `--dry-run` without loading models or writing outputs.

Annotation uses compatible local chat models through vLLM. Intensity scoring
uses Hugging Face causal language models with accessible hidden states. Choose
models appropriate for your language and task, and inspect a small sample before
processing a full corpus. Launchers protect existing outputs and check score
alignment and catalog coverage; they do not automatically resume interrupted runs.

## Checks

Run the tests and an entirely synthetic CPU profile example:

```bash
python -m unittest discover -s tests -v
python examples/profiles/run_smoke.py --out_dir outputs/profile_smoke
```

These checks do not download models. Model-interface and remote-download tests
use test doubles. The synthetic pipeline example is separate from the actual
aggregate examples bundled for profile inspection.

## Coming soon

Steering and model-training code, along with source-specific data acquisition
instructions, are planned for a future release. The current pipeline starts
from prepared data.

## Data and licenses

Users provide prepared corpora when constructing new profiles. Place local
inputs under `data/` and generated artifacts under `outputs/`; both are ignored
by Git. The supplied [value and context catalogs](resources/original/README.md)
are reusable pipeline inputs, not source-text datasets.

Profiles describe annotated expressions and model-derived scores in selected
corpora, not representative estimates of individual beliefs or national
cultures. Support counts are not counts of independent people, and support
thresholds do not establish annotation accuracy or privacy guarantees.

Reddit content is subject to its applicable platform terms and content-owner
rights. CulturaX follows the terms of its mC4 and OSCAR sources; OSCAR's CC0
notice does not cover the underlying crawled text. See the linked upstream
notices in [source-data licenses and terms](DATA.md#source-data-licenses-and-terms).

Software and accompanying usage documentation are licensed under [MIT](LICENSE).
Constructed value/context catalogs and aggregate profiles, including the bundled
examples, are licensed under [CC BY 4.0](LICENSE-DATA) for the project's own
contributions. Paper figures and the illustrative animation retain their
existing rights and are outside these license grants.

Source-corpus and other third-party rights remain separate. See
[DATA.md](DATA.md#project-licensing-status), the [example notice](examples/released_profiles/README.md),
and [third-party notices](THIRD_PARTY_NOTICES.md) for scope and attribution.
