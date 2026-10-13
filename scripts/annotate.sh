#!/usr/bin/env bash
# Prepared CSV -> expressive sentences -> canonical values -> annotations.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
if [[ "${ARGS[0]:-}" == --help ]]; then
  printf '%s\n' 'Usage: bash scripts/annotate.sh ATTRIBUTE INPUT_CSV [MAPPING_JSONL] [TEXT_COLUMN=text] [TEXT_TYPE=plain] [--dry-run]' \
    'Required environment: ANNOTATION_MODEL, EMBEDDING_MODEL. Choose models supported by the respective inference backends.' \
    'Mapping defaults to outputs/mapping/ATTRIBUTE.jsonl; missing files are initialized, existing writable files are updated in place. Do not share a map between concurrent writers.' \
    'Paths are relative to this project. Outputs are fresh-run only. --dry-run prints commands without running Python or creating files.' \
    'Resources: FACTORS_JSON, VALUE_ITEMS_JSON, VALUE_DEFS_JSON, REGIMES_JSONL (shipped catalogs); REPRESENTATIVES_JSON defaults to VALUE_DEFS_JSON.' \
    'Options: TENSOR_PARALLEL_SIZE=1, ANNOTATION_TEMPERATURE=0, ANNOTATION_TOP_P=1, ANNOTATION_BATCH_SIZE=1, TAGGING_BATCH_SIZE=64,' \
    '  EMBEDDING_DEVICE=cpu, EMBEDDING_BATCH_SIZE=64, MAPPING_SEED=42, ANNOTATION_DEVICE=cuda, DTYPE=auto, MAX_ROWS=0,' \
    '  MIN_EXPRESSIVE_SCORE=30, TOP_K_VALUE_ITEMS=4, EXPRESSIVE_MAX_TOKENS=1024, VALUE_MAX_TOKENS=256, RELATION_MAX_TOKENS=768, SURFACE_MAX_TOKENS=256, CONTEXT_MAX_TOKENS=200.'
  exit 0
fi
[[ ${#ARGS[@]} -ge 2 && ${#ARGS[@]} -le 5 ]] || die 'Use --help for arguments.'
require_env ANNOTATION_MODEL EMBEDDING_MODEL
attr="${ARGS[0]}"; input="${ARGS[1]}"; mapping="${ARGS[2]:-outputs/mapping/${ARGS[0]}.jsonl}"
column="${ARGS[3]:-text}"; framing="${ARGS[4]:-plain}"
attribute_name "$attr"
case "$framing" in plain|comment|dialogue) ;; *) die 'TEXT_TYPE must be plain, comment or dialogue.' ;; esac
require_file "$input" "$FACTORS" "$VALUE_ITEMS" "$VALUE_DEFS" "$REGIMES" "$REPRESENTATIVES"
if [[ -e "$mapping" || -L "$mapping" ]]; then
  [[ -f "$mapping" && -w "$mapping" ]] || die 'An existing MAPPING_JSONL must be a writable regular file; it is updated in place.'
fi
expressive="outputs/expressive/$attr.jsonl"
raw="outputs/values/$attr.raw.jsonl"; canonical="outputs/values/$attr.jsonl"
graph="outputs/relations/$attr.with_graph.jsonl"; relations="outputs/relations/$attr.normalized.jsonl"
surface="outputs/surface/$attr.surface_regimes.normalized.jsonl"
surface_full="outputs/surface/$attr.with_surface_regimes.jsonl"
context="outputs/context/$attr.context_states.jsonl"
require_new "$expressive" "$raw" "$canonical" "$graph" "$relations" "$surface" "$surface_full" "$context" "$context.normalized.jsonl" "outputs/mapping_cache/$attr"
annotation_args=(--llm_model "$ANNOTATION_MODEL" --tensor_parallel_size "${TENSOR_PARALLEL_SIZE:-1}" --temperature "${ANNOTATION_TEMPERATURE:-0}" --top_p "${ANNOTATION_TOP_P:-1}")
run mkdir -p outputs/expressive outputs/values outputs/relations outputs/surface outputs/context "outputs/mapping_cache/$attr"
run "$VALUE_PYTHON" -m processing.extract_valued_text \
  --input_csv "$input" --text_col "$column" --text_type "$framing" --output_path "$expressive" \
  "${annotation_args[@]}" --max_new_tokens "${EXPRESSIVE_MAX_TOKENS:-1024}" \
  --min_score "${MIN_EXPRESSIVE_SCORE:-30}" --max_rows "${MAX_ROWS:-0}" --batch_size "${ANNOTATION_BATCH_SIZE:-1}"
run "$VALUE_PYTHON" -m value_item_parser.extract_value_item \
  --input_jsonl "$expressive" --output_jsonl "$raw" "${annotation_args[@]}" \
  --max_new_tokens "${VALUE_MAX_TOKENS:-256}" --batch_size "${ANNOTATION_BATCH_SIZE:-1}" \
  --min_expressive_score "${MIN_EXPRESSIVE_SCORE:-30}" --max_full_text_chars 8000 --max_rows "${MAX_ROWS:-0}"
run "$VALUE_PYTHON" -m value_item_parser.deduplicate_value_items \
  --input_jsonl "$raw" --output_jsonl "$canonical" --mapping_jsonl "$mapping" --rep_json "$REPRESENTATIVES" \
  --allow_embedding --embedding_model "$EMBEDDING_MODEL" --device "${EMBEDDING_DEVICE:-cpu}" \
  --embed_batch_size "${EMBEDDING_BATCH_SIZE:-64}" --seed "${MAPPING_SEED:-42}" \
  --rep_cache_npz "outputs/mapping_cache/$attr/rep_embeddings.npz"
coverage "$canonical"
run "$VALUE_PYTHON" -m relation_parser.extract_value_relationships \
  --input_jsonl "$canonical" --output_jsonl_original "$graph" --output_jsonl "$relations" \
  "${annotation_args[@]}" --max_new_tokens "${RELATION_MAX_TOKENS:-768}" \
  --use_context --context_window_chars 1600 --evidence_max_chars 2200 --top_k_value_items "${TOP_K_VALUE_ITEMS:-4}" \
  --keep_warnings --count_total_first --save_every_rows 5000
run "$VALUE_PYTHON" -m intensity_parser.classify_surface_value_regimes \
  --input_jsonl "$canonical" --regimes_jsonl "$REGIMES" --value_defs_json "$VALUE_DEFS" \
  --output_jsonl_original "$surface_full" --output_jsonl "$surface" "${annotation_args[@]}" \
  --max_new_tokens "${SURFACE_MAX_TOKENS:-256}" --batch_size "${TAGGING_BATCH_SIZE:-64}" --max_rows "${MAX_ROWS:-0}" --top_k_value_items "${TOP_K_VALUE_ITEMS:-4}" --use_context \
  --context_window_chars 1400 --context_max_chars 2000 --target_max_chars 800 --def_max_chars 500 \
  --keep_warnings --count_total_first --save_every_rows 5000
run "$VALUE_PYTHON" -m context_parser.extract_context_states \
  --input_jsonl "$graph" --factors_json "$FACTORS" --output_path "$context" --output_format jsonl \
  "${annotation_args[@]}" --max_new_tokens "${CONTEXT_MAX_TOKENS:-200}" \
  --device "${ANNOTATION_DEVICE:-cuda}" --dtype "${DTYPE:-auto}" --batch_size "${TAGGING_BATCH_SIZE:-64}" \
  --row_chunk_size 2000 --two_stage --save_normalized --keep_warnings --save_every_rows 1000
