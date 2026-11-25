#!/usr/bin/env bash
# Score and attach the same canonical input; retain sentence metadata.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
if [[ "${ARGS[0]:-}" == --help ]]; then
  printf '%s\n' 'Usage: bash scripts/score.sh ATTRIBUTE SOURCE [--dry-run]' \
    'Required environment: MODEL_NAME, LAYERS. SOURCE is the label used for matching value vectors, not a fixed model preset.' \
    'Uses outputs/values/ATTRIBUTE.jsonl and outputs/value_vectors/SOURCE; checks extraction config and strict sentence/count alignment.' \
    'Options: POOLING=mean_last_k_tokens, LAST_K=3, DTYPE=auto, DEVICE_MAP=auto, SCORE_MAX_LENGTH=256, SCORE_BATCH_SIZE=16,' \
    '  TRUST_REMOTE_CODE=0 (set 1 only for code you trust), SCORE_QUANT=int16. Cosine scoring is fixed; SCORE_QUANT can be int16 or float16.' \
    'Resources: FACTORS_JSON, VALUE_ITEMS_JSON, VALUE_DEFS_JSON, REGIMES_JSONL. Existing output files are refused.'
  exit 0
fi
[[ ${#ARGS[@]} == 2 ]] || die 'Use --help for arguments.'
attr="${ARGS[0]}"; source_name="${ARGS[1]}"
attribute_name "$attr"; model_source "$source_name"; trust_remote_args
canonical="outputs/values/$attr.jsonl"
score_dir="outputs/scores/$source_name"
score_prefix="$score_dir/$attr."
attached="outputs/activations/$source_name/processed/$attr.jsonl"
require_file "$canonical" "$VECTOR_PREFIX.value_vectors.npz" "$VECTOR_PREFIX.value_vectors_index.jsonl" "$ACTIVATION_CONFIG"
require_new "$score_prefix.scores.npz" "$score_prefix.meta.jsonl" "$score_prefix.columns.json" "$attached" "$attached.errors.jsonl"
coverage "$canonical"
run "$VALUE_PYTHON" -m activation_parser.calculate_activation_intensity \
  --input_jsonls "$canonical" --text_key target_sentence --model_name "$MODEL_NAME" \
  ${TRUST_ARGS[@]+"${TRUST_ARGS[@]}"} --dtype "${DTYPE:-auto}" --device_map "${DEVICE_MAP:-auto}" --layers "$LAYERS" \
  --pooling "${POOLING:-mean_last_k_tokens}" --last_k "${LAST_K:-3}" --max_length "${SCORE_MAX_LENGTH:-256}" --batch_size "${SCORE_BATCH_SIZE:-16}" \
  --value_index_jsonl "$VECTOR_PREFIX.value_vectors_index.jsonl" --value_vectors_npz "$VECTOR_PREFIX.value_vectors.npz" \
  --activation_config "$ACTIVATION_CONFIG" --dedup_value_defs_json "$VALUE_DEFS" \
  --metric cosine --out_dir "$score_dir" --format npz --quant "${SCORE_QUANT:-int16}" --count_rows
run "$VALUE_PYTHON" -m profile_constructor.attach_valueitem_projection_scores \
  --input_jsonl "$canonical" --columns_json "$score_prefix.columns.json" \
  --scores_npz "$score_prefix.scores.npz" --meta_jsonl "$score_prefix.meta.jsonl" --out_jsonl "$attached" \
  --text_key target_sentence --value_items_key value_items --proj_field proj_scores --strict_meta_check --fail_fast
