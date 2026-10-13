#!/usr/bin/env bash
# Aggregate annotations and selected score sources, then assemble profiles.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
if [[ "${ARGS[0]:-}" == --help ]]; then
  printf '%s\n' 'Usage: bash scripts/profiles.sh ATTRIBUTE SOURCE [SOURCE ...] [--dry-run]' \
    'SOURCE names are arbitrary unique labels from score.sh; one or several sources can be selected explicitly. No model is loaded.' \
    'Builds discrete, continuous, relation and context profiles, then assembly. Existing output destinations are refused.' \
    'Resources: FACTORS_JSON, VALUE_ITEMS_JSON, VALUE_DEFS_JSON, REGIMES_JSONL (shipped catalogs by default).' \
    'Options: MIN_VALUE_ITEM_COUNT=200, MIN_GROUP_N=20, MIN_N_PER_STATE=10 (state-support flags and assembly filter),' \
    '  RELATION_TOP_K=20, RELATION_MIN_COUNT=10, JOINT_TOP_K=20, JOINT_MIN_PARTNER_COUNT=3, LAMBDA_PRIOR=20,' \
    '  DISCRETE_DRAWS=2000, CONTINUOUS_DRAWS=2000, JOINT_DRAWS=200, PROFILE_SEED=0,' \
    '  NORMALIZATION_MODE=polarity_quantile, OUTLIER_QUANTILE_LOW=0.01, OUTLIER_QUANTILE_HIGH=0.99, OUTLIER_ACTION=winsorize.' \
    'PREFER_LAYER optionally selects a layer; otherwise each continuous constructor selects its middle available layer. It does not average layers.' \
    'The discrete base filter is strictly greater than MIN_VALUE_ITEM_COUNT; the other value-support filters are greater than or equal.'
  exit 0
fi
[[ ${#ARGS[@]} -ge 2 ]] || die 'Provide ATTRIBUTE and at least one SOURCE; use --help.'
attr="${ARGS[0]}"; attribute_name "$attr"
names=(); paths=()
for source_name in "${ARGS[@]:1}"; do
  safe_name "$source_name" SOURCE
  for previous in "${names[@]}"; do [[ "$source_name" != "$previous" ]] || die "Duplicate source: $source_name"; done
  names+=("$source_name")
  paths+=("outputs/activations/$source_name/processed/$attr.jsonl")
done
support="${MIN_VALUE_ITEM_COUNT:-200}"; state_support="${MIN_N_PER_STATE:-10}"
positive_integer MIN_VALUE_ITEM_COUNT "$support"
positive_integer MIN_N_PER_STATE "$state_support"
normalization="${NORMALIZATION_MODE:-polarity_quantile}"
case "$normalization" in minmax|zscore|robust_zscore|zscore_tanh|robust_tanh|polarity_quantile|polarity_quantile_tanh) ;; *) die 'Unknown NORMALIZATION_MODE.' ;; esac
normalization_args=(--normalization_mode "$normalization" --outlier_quantile_low "${OUTLIER_QUANTILE_LOW:-0.01}" --outlier_quantile_high "${OUTLIER_QUANTILE_HIGH:-0.99}" --outlier_action "${OUTLIER_ACTION:-winsorize}")
if [[ -n "${PREFER_LAYER:-}" ]]; then normalization_args+=(--prefer_layer "$PREFER_LAYER"); fi
model_names=$(IFS=,; printf '%s' "${names[*]}")
activation_jsonls=$(IFS=,; printf '%s' "${paths[*]}")
canonical="outputs/values/$attr.jsonl"
context="outputs/context/$attr.context_states.jsonl.normalized.jsonl"
surface="outputs/surface/$attr.surface_regimes.normalized.jsonl"
relations="outputs/relations/$attr.normalized.jsonl"; graph="outputs/relations/$attr.with_graph.jsonl"
counted_dir="outputs/distributional_profiles/surface_regimes/$attr"
filtered_dir="outputs/distributional_profiles/surface_regimes_filtered/$attr"
continuous_dir="outputs/distributional_profiles/continuous_distribution/$attr"
distribution_suffix=".surface_regimes.filtered_noNotInv_gt${support}.jsonl"
ot_suffix=".items_by_count${support}.ot_context_effects.csv"
continuous_base_suffix=".items_by_count${support}.group_attribute_value.gaussian_fit.jsonl"
continuous_ot_suffix=".items_by_count${support}.norm_${normalization}.ot_context_effects.continuous.csv"
discrete_ot="outputs/context_effect_ot/$attr$ot_suffix"
continuous_ot="outputs/context_effect_ot_continuous/$attr$continuous_ot_suffix"
relation_profile="outputs/value_relation_distributions/$attr.distribution.json"
joint_dir="outputs/context_effect_joint_ot_relations/$attr"
assembled_dir="outputs/distributional_profile_dataset/$attr"
require_file "$canonical" "$context" "$surface" "$relations" "$graph" "$FACTORS" "$REGIMES" "${paths[@]}"
require_new "$counted_dir" "$filtered_dir" "$continuous_dir" "$discrete_ot" "$continuous_ot" "$relation_profile" "$joint_dir" "$assembled_dir"
coverage "$canonical"
run mkdir -p "$counted_dir" "$joint_dir" outputs/value_relation_distributions outputs/context_effect_ot outputs/context_effect_ot_continuous
run "$VALUE_PYTHON" -m profile_constructor.construct_distributional_surface_regimes \
  --labels_jsonl "$surface" --regimes_jsonl "$REGIMES" --out_jsonl "$counted_dir/$attr.surface_regimes.counted.jsonl"
run "$VALUE_PYTHON" -m profile_constructor.filter_surface_regimes \
  --in_dir "$counted_dir" --out_dir "$filtered_dir" --threshold "$support"
run "$VALUE_PYTHON" -m value_profiler.build_value_relation_distribution \
  --input "$relations" --output "$relation_profile" --attribute "$attr" --top_k "${RELATION_TOP_K:-20}" --min_count "${RELATION_MIN_COUNT:-10}"
run "$VALUE_PYTHON" -m context_parser.context_effect_ot \
  --context_jsonl "$context" --surface_jsonl "$surface" --factors_json "$FACTORS" \
  --surface_regime_defs_jsonl "$REGIMES" --min_value_item_count "$support" --lambda_prior "${LAMBDA_PRIOR:-20}" \
  --draws "${DISCRETE_DRAWS:-2000}" --seed "${PROFILE_SEED:-0}" --min_n_per_state "$state_support" --out_csv "$discrete_ot"
run "$VALUE_PYTHON" -m context_parser.context_effect_joint_ot_relations \
  --context_jsonl "$context" --relation_jsonl "$graph" --factors_json "$FACTORS" \
  --top_k_partners "${JOINT_TOP_K:-20}" --min_partner_count "${JOINT_MIN_PARTNER_COUNT:-3}" \
  --min_anchor_value_count "$support" --lambda_prior "${LAMBDA_PRIOR:-20}" --draws "${JOINT_DRAWS:-200}" --seed "${PROFILE_SEED:-0}" --min_n_per_state "$state_support" \
  --out_csv "$joint_dir/$attr.joint_ot.csv" --out_counts_csv "$joint_dir/$attr.joint_counts.csv" --out_delta_csv "$joint_dir/$attr.joint_deltas.csv"
run "$VALUE_PYTHON" -m profile_constructor.construct_distributional_raw_intensity \
  --attribute_name "$attr" --context_jsonl "$context" --activation_jsonls "$activation_jsonls" --model_names "$model_names" \
  --factors_json "$FACTORS" --group_mode attribute_value "${normalization_args[@]}" --min_value_item_count "$support" --min_group_n "${MIN_GROUP_N:-20}" \
  --out_summary_csv "$continuous_dir/$attr.items_by_count${support}.group_attribute_value.summary.csv" \
  --out_raw_npz "$continuous_dir/$attr.items_by_count${support}.group_attribute_value.raw.npz" \
  --out_fit_jsonl "$continuous_dir/$attr$continuous_base_suffix"
run "$VALUE_PYTHON" -m context_parser.context_effect_ot_raw_score \
  --attribute_name "$attr" --context_jsonl "$context" --activation_jsonls "$activation_jsonls" --model_names "$model_names" \
  --factors_json "$FACTORS" "${normalization_args[@]}" --min_value_item_count "$support" \
  --draws "${CONTINUOUS_DRAWS:-2000}" --seed "${PROFILE_SEED:-0}" --min_n_per_state "$state_support" --out_csv "$continuous_ot"
run "$VALUE_PYTHON" -m value_profiler.distributional_dataset \
  --attributes "$attr" --distribution_dir "$filtered_dir" --distribution_suffix "$distribution_suffix" \
  --ot_dir outputs/context_effect_ot --ot_suffix "$ot_suffix" \
  --continuous_base_dir outputs/distributional_profiles/continuous_distribution --continuous_base_suffix "$continuous_base_suffix" \
  --continuous_ot_dir outputs/context_effect_ot_continuous --continuous_ot_suffix "$continuous_ot_suffix" \
  --min_ot_state_count "$state_support" --factors_json "$FACTORS" --surface_regime_defs_jsonl "$REGIMES" --output_dir "$assembled_dir"
