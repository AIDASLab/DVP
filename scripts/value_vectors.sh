#!/usr/bin/env bash
# Positive/negative examples -> pooled activations -> value directions.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
if [[ "${ARGS[0]:-}" == --help ]]; then
  printf '%s\n' 'Usage: bash scripts/value_vectors.sh generate|SOURCE [--dry-run]' \
    'generate requires ANNOTATION_MODEL. A SOURCE is your own safe label; extraction requires MODEL_NAME and comma-separated LAYERS.' \
    'Generation writes outputs/contrastive/posneg.jsonl; extraction writes outputs/value_vectors/SOURCE. Existing destinations are refused.' \
    'Generation options: TENSOR_PARALLEL_SIZE=1, GENERATION_TEMPERATURE=0.8, GENERATION_TOP_P=0.95, GENERATION_MAX_TOKENS=800,' \
    '  POS_PER_VALUE_ITEM=50, NEG_PER_VALUE_ITEM=50, CHUNK_POS=5, CHUNK_NEG=5, GENERATION_SEED=1234, FACTORS_JSON (shipped catalog).' \
    'Extraction options: POOLING=mean_last_k_tokens, LAST_K=3, DTYPE=auto, EXTRACTION_BATCH_SIZE=1, SHARD_SIZE=50000,' \
    '  EXTRACTION_MAX_LENGTH (unset means no cap), DEVICE_MAP=auto, TRUST_REMOTE_CODE=0 (set 1 only for code you trust).' \
    'ALLOW_MISSING_NEG_OR_POS=1 keeps a direction when one generated side is missing (zero mean for that side); set 0 to require both sides. Inspect the saved vector counts.'
  exit 0
fi
[[ ${#ARGS[@]} == 1 ]] || die 'Use --help for arguments.'
posneg="outputs/contrastive/posneg.jsonl"
if [[ "${ARGS[0]}" == generate ]]; then
  require_env ANNOTATION_MODEL
  require_file "$FACTORS"
  require_new outputs/contrastive
  run mkdir -p outputs/contrastive
  run "$VALUE_PYTHON" -m value_item_parser.generate_posneg_sentences_from_value_items \
    --factors_json "$FACTORS" --output_path "$posneg" --llm_model "$ANNOTATION_MODEL" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE:-1}" --temperature "${GENERATION_TEMPERATURE:-0.8}" \
    --top_p "${GENERATION_TOP_P:-0.95}" --max_new_tokens "${GENERATION_MAX_TOKENS:-800}" --dtype "${DTYPE:-auto}" \
    --pos_per_value_item "${POS_PER_VALUE_ITEM:-50}" --neg_per_value_item "${NEG_PER_VALUE_ITEM:-50}" \
    --chunk_pos "${CHUNK_POS:-5}" --chunk_neg "${CHUNK_NEG:-5}" \
    --seed "${GENERATION_SEED:-1234}" --seed_step 97 --temp_jitter 0.15 --min_unique_ratio 0.85 --max_retries_per_chunk 2 \
    --max_value_items 0 --start_index 0 --end_index 0 --save_per_value_item
else
  model_source "${ARGS[0]}"
  trust_remote_args
  require_file "$posneg"
  require_new "$VECTOR_DIR"
  extraction_args=()
  if [[ -n "${EXTRACTION_MAX_LENGTH:-}" ]]; then extraction_args+=(--max_length "$EXTRACTION_MAX_LENGTH"); fi
  vector_args=()
  case "${ALLOW_MISSING_NEG_OR_POS:-1}" in
    0) ;; 1) vector_args+=(--allow_missing_neg_or_pos) ;; *) die 'ALLOW_MISSING_NEG_OR_POS must be 0 or 1.' ;;
  esac
  run "$VALUE_PYTHON" -m activation_parser.save_activations_from_jsonl \
    --input_jsonl "$posneg" --model_name "$MODEL_NAME" --out_dir "$VECTOR_DIR/activations" \
    --layers "$LAYERS" --pooling "${POOLING:-mean_last_k_tokens}" --last_k "${LAST_K:-3}" \
    --batch_size "${EXTRACTION_BATCH_SIZE:-1}" --shard_size "${SHARD_SIZE:-50000}" \
    --format npz --dtype "${DTYPE:-auto}" "${TRUST_ARGS[@]}" --device_map "${DEVICE_MAP:-auto}" \
    --require_fresh_output "${extraction_args[@]}"
  run "$VALUE_PYTHON" -m activation_parser.save_value_vectors \
    --meta_jsonl "$VECTOR_DIR/activations/${MODEL_TAG}_activations_meta.jsonl" \
    --shard_glob "$VECTOR_DIR/activations/${MODEL_TAG}_activations_shard*.npz" \
    --out_prefix "$VECTOR_PREFIX" "${vector_args[@]}"
fi
