# Distributional value profiling

Tools for preparing text, annotating values, constructing value profiles, and
working with precomputed aggregates. Profiles describe categorical value
expressions, continuous value scores, relations between values, and their
variation across contexts.

You can use existing profiles without rerunning annotation, or build profiles
from your own prepared text. Annotation models, intensity models, catalogs, and
processing settings are configurable.

## What is included

| Component | Location / availability |
| --- | --- |
| Data preparation, annotation, and profile construction | Code in this repository; see the [workflow guide](docs/workflow.md) |
| Small actual aggregate examples | [Bundled examples](examples/released_profiles/README.md), about **0.15 MiB** including their manifest and documentation |
| Full aggregate profiles | Separate bundle prepared for Hugging Face, about **779 MiB compressed**; publication is pending |
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
not included. The aggregate examples and full bundle have no
assigned dataset license; see [Data and rights](#data-and-rights).

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

For a downloaded or locally prepared bundle, pass its directory containing
`manifest.json` to `--source-dir`. The same selection commands work on the full
bundle and the examples.

Hugging Face retrieval additionally requires:

```bash
python -m pip install huggingface_hub
```

After publication, replace `YOUR_ORG/YOUR_DATASET` below with the actual dataset
repository ID. **This is currently a placeholder; no public download is announced.**

```bash
python -m profile_data.organize \
  --repo-id YOUR_ORG/YOUR_DATASET \
  --revision main --cache-dir outputs/profile_cache \
  --family base --family context \
  --group culturax_ko_cleaned --value fairness \
  --min-support base=500 --min-support context=30 \
  --output-dir outputs/selected_profiles
```

The organizer fetches the manifest, selected family/group files, and supplied
catalogs. It verifies file sizes and SHA-256 checksums and records the resolved
remote commit. Value and support filtering happen locally. Repeat `--family`,
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

## Data and rights

Users provide prepared corpora when constructing new profiles. Place local
inputs under `data/` and generated artifacts under `outputs/`; both are ignored
by Git. The supplied [value and context catalogs](resources/original/README.md)
are reusable pipeline inputs, not source-text datasets.

Profiles describe annotated expressions and model-derived scores in selected
corpora, not representative estimates of individual beliefs or national
cultures. Support counts are not counts of independent people, and support
thresholds do not establish annotation accuracy or privacy guarantees.

The aggregate examples and full bundle have **no assigned dataset license** and
remain subject to source-specific rights review before publication. Hosting on
Hugging Face rather than GitHub does not change those requirements. Removing
source text does not itself establish redistribution permission.

See [DATA.md](DATA.md), the [example notice](examples/released_profiles/README.md),
and the full bundle's `NOTICE.md` for the applicable data scope and source notices.
