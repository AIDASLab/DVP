# Workflow

Run commands from the project root. Relative launcher paths are resolved there.
Use `PYTHON=/path/to/python` to select an environment. Choose GPU allocation
outside the scripts; no launcher assigns GPU IDs.

Each launcher accepts `--help` and `--dry-run`. A dry run prints commands without
loading models, downloading weights, or writing files. Normal inference may
download weights when given a remote model ID; use local directories for offline
work. Existing output destinations are refused. There is no automatic resume.

## 1. Prepared text to annotations

Supply a CSV with a text column containing a document, comment, or conversation.
The selected text type controls the annotation framing.

```csv
text
"I think everyone deserves fair treatment."
```

Set a compatible local chat model and a Sentence Transformers-compatible
embedding model. The paths below are placeholders for your own model directories:

```bash
export ANNOTATION_MODEL="/path/to/chat-model"
export EMBEDDING_MODEL="/path/to/embedding-model"
bash scripts/annotate.sh demo data/corpus.csv --dry-run
bash scripts/annotate.sh demo data/corpus.csv
```

Full interface:

```text
annotate.sh ATTRIBUTE INPUT_CSV [MAPPING_JSONL] [TEXT_COLUMN=text] [TEXT_TYPE=plain]
```

Text types are `plain`, `comment`, and `dialogue`. An optional mapping path may
refer to an existing writable dictionary or a new file. The default is
`outputs/mapping/ATTRIBUTE.jsonl`. It is updated in place; use a copy of any
dictionary you need to preserve and avoid concurrent writers.

This runs expressive selection, value extraction, embedding-assisted mapping,
then relation, expression-category, and context-state annotation. The components
can also be called separately with `python -m MODULE --help`.

| Output for attribute `demo` | Path |
| --- | --- |
| Expressive sentences | `outputs/expressive/demo.jsonl` |
| Raw and canonical values | `outputs/values/demo.raw.jsonl`, `outputs/values/demo.jsonl` |
| Relation graphs | `outputs/relations/demo.with_graph.jsonl`, `demo.normalized.jsonl` |
| Expression labels | `outputs/surface/demo.with_surface_regimes.jsonl`, `demo.surface_regimes.normalized.jsonl` |
| Context states | `outputs/context/demo.context_states.jsonl`, `demo.context_states.jsonl.normalized.jsonl` |

A canonical value record has this shape:

```json
{"idx":0,"text_type":"plain","full_text":"I think everyone deserves fair treatment.","target_sentence":"I think everyone deserves fair treatment.","value_items":[{"value_item":"fairness","score":90}]}
```

These examples are synthetic. Annotation adds `value_relation_graph`,
`surface_value_regimes`, or `context_states` fields. Sentence identity and value
names must remain consistent across files. Annotation models may differ in output
quality; inspect a small sample before scaling up.

Key options include `TENSOR_PARALLEL_SIZE`, `ANNOTATION_TEMPERATURE`,
`ANNOTATION_TOP_P`, `ANNOTATION_BATCH_SIZE`, `TAGGING_BATCH_SIZE`,
`MIN_EXPRESSIVE_SCORE`, `MAX_ROWS`, `TOP_K_VALUE_ITEMS` (default 4 for relation
and expression annotation), and `EMBEDDING_DEVICE` (default `cpu`).
`REPRESENTATIVES_JSON` selects the embedding candidate list or definition
dictionary; it defaults to `VALUE_DEFS_JSON`.
Stage-specific token limits and further options are listed by `--help`.
Individual Python commands expose additional controls.

## 2. Contrastive examples and value vectors

Generate positive/negative examples once, or supply an existing compatible
`outputs/contrastive/posneg.jsonl`:

```bash
bash scripts/value_vectors.sh generate
```

Each line contains `value_item`, `label` (`pos` or `neg`), and `sentence`.
Generation counts and sampling are configurable, including
`POS_PER_VALUE_ITEM`, `NEG_PER_VALUE_ITEM`, and `GENERATION_TEMPERATURE`.

Choose an intensity model, valid hidden-state layer indices, and your own source
label. The label below is arbitrary, not a preset:

```bash
export MODEL_NAME="/path/to/causal-model"
export LAYERS="8,16,24"
bash scripts/value_vectors.sh my_model
```

Choose layers that exist in your model. Hidden-state index 0 is the embedding
output; index i is the output after transformer block i. Intensity extraction
requires a compatible Hugging Face causal model exposing these hidden states.

Vector files are written under `outputs/value_vectors/my_model/`:

- `activations/*_activations_config.json`, metadata JSONL, and NPZ shards
- `value.value_vectors.npz`
- `value.value_vectors_index.jsonl`
- `value.value_vectors_stats.json`

Inspect positive/negative counts. By default, `ALLOW_MISSING_NEG_OR_POS=1`
permits a missing side and uses zero for that side's mean; set it to `0` to
require both sides.

## 3. Intensity scores

Use the same model, layer list, pooling, and source label used for the vectors:

```bash
bash scripts/score.sh demo my_model
```

The scorer checks the vector extraction configuration and attaches scores to the
same canonical input rows. Keep sentence metadata and do not modify that input
between scoring and attachment. Attached scores are written to
`outputs/activations/my_model/processed/demo.jsonl`.

Shared options include `DTYPE`, `POOLING`, `LAST_K`, and `DEVICE_MAP`.
Extraction/scoring batch sizes and maximum lengths are separate options.
The default last-token pooling requires right padding. `TRUST_REMOTE_CODE=0`
is the default for activation loading; enable it only for model code you trust.

The scoring arrays and metadata under `outputs/scores/my_model/` have names
such as `demo..scores.npz`, `demo..meta.jsonl`, and `demo..columns.json`.
They are intermediates used by score attachment.

## 4. Profiles

Build profiles from the annotation and attached-score files:

```bash
bash scripts/profiles.sh demo my_model
```

One source is sufficient. To combine multiple compatible sources, generate and
score each under a distinct label, then pass all labels to the profile command.

| Configuration | Default |
| --- | --- |
| `MIN_VALUE_ITEM_COUNT` | 200 |
| `MIN_N_PER_STATE` | 10 |
| `MIN_GROUP_N` | 20 |
| `RELATION_TOP_K`, `RELATION_MIN_COUNT` | 20, 10 |
| `JOINT_TOP_K`, `JOINT_MIN_PARTNER_COUNT` | 20, 3 |
| `DISCRETE_DRAWS`, `CONTINUOUS_DRAWS`, `JOINT_DRAWS` | 2000, 2000, 200 |
| `LAMBDA_PRIOR`, `PROFILE_SEED` | 20, 0 |
| `NORMALIZATION_MODE` | `polarity_quantile` |

For example, `MIN_VALUE_ITEM_COUNT=50 bash scripts/profiles.sh demo my_model`
changes support filtering and passes the corresponding filenames to assembly.
Choose settings appropriate for your data. Small or fully filtered inputs can
produce no profiles.

Base expression profiles remove `Not Invoked`, renormalize remaining counts,
and require support strictly above the selected threshold. Context constructors
use their own sentence-support filters. Continuous aggregation uses a middle
available layer per model by default; `PREFER_LAYER` selects a specific layer.
Normalized scores are averaged across available sources. Continuous base
`n_samples` counts factor-expanded records; `n_unique_sentences` counts distinct
sentence keys.

Final intensity profile files under
`outputs/distributional_profile_dataset/demo/` are:

- `distributional_profile_base_v2.jsonl`
- `distributional_profile_pairwise_ot_v2.jsonl`
- `distributional_profile_all_v2.jsonl`

Base relations are separate at
`outputs/value_relation_distributions/demo.distribution.json`.
Context-conditioned relation outputs are under
`outputs/context_effect_joint_ot_relations/demo/`.

## Catalogs and individual stages

Override `FACTORS_JSON`, `VALUE_ITEMS_JSON`, `VALUE_DEFS_JSON`, or
`REGIMES_JSONL` to use compatible catalogs. Use the same files throughout the
run. See [catalog formats and coverage](../resources/original/README.md).

You can reuse prepared annotations, contrastive examples, or vectors and run only
the necessary Python modules. The shell scripts are convenience commands, not a
required workflow engine. `--dry-run` shows the exact module calls and paths;
each module's `--help` lists its input and configuration arguments.
