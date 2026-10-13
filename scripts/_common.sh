#!/usr/bin/env bash
# Shared paths and checks; processing remains in the Python modules.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
VALUE_PYTHON="${PYTHON:-python3}"
FACTORS="${FACTORS_JSON:-resources/original/context_factors_revised.json}"
VALUE_ITEMS="${VALUE_ITEMS_JSON:-resources/original/unique_value_items.json}"
VALUE_DEFS="${VALUE_DEFS_JSON:-resources/original/unique_value_item_definitions.json}"
REPRESENTATIVES="${REPRESENTATIVES_JSON:-$VALUE_DEFS}"
REGIMES="${REGIMES_JSONL:-resources/original/surface_value_regimes_gpt.jsonl}"
DRY_RUN=0
ARGS=()
for arg in "$@"; do
  if [[ "$arg" == --dry-run ]]; then DRY_RUN=1; else ARGS+=("$arg"); fi
done

die() { printf 'Error: %s\n' "$*" >&2; exit 2; }
safe_name() {
  [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || die "${2:-Name} must start with a letter or digit and contain only letters, digits, underscores or hyphens."
}
attribute_name() { safe_name "$1" ATTRIBUTE; }
require_env() {
  local key
  for key in "$@"; do [[ -n "${!key:-}" ]] || die "Set $key explicitly; see --help."; done
}
positive_integer() { [[ "$2" =~ ^[0-9]+$ && "$2" -gt 0 ]] || die "$1 must be a positive integer."; }
require_new() {
  local path
  for path in "$@"; do
    [[ ! -e "$path" && ! -L "$path" ]] || die "Output already exists: $path. Inspect it; these fresh-run scripts do not resume or overwrite."
  done
}
require_file() {
  local path
  for path in "$@"; do
    if [[ "$DRY_RUN" == 0 && ! -f "$path" ]]; then die "Missing input: $path"; fi
  done
}
run() {
  printf '+ '; printf '%q ' "$@"; printf '\n'
  if [[ "$DRY_RUN" == 0 ]]; then "$@"; fi
}
coverage() {
  run "$VALUE_PYTHON" -m value_item_parser.check_value_coverage \
    --input_jsonl "$1" --value_items_json "$VALUE_ITEMS" \
    --value_defs_json "$VALUE_DEFS" --factors_json "$FACTORS" --regimes_jsonl "$REGIMES"
}
model_source() {
  safe_name "$1" SOURCE
  require_env MODEL_NAME LAYERS
  [[ "$LAYERS" =~ ^[0-9]+(,[0-9]+)*$ ]] || die 'LAYERS must be comma-separated nonnegative layer numbers.'
  case "${POOLING:-mean_last_k_tokens}" in
    mean_all_tokens|last_token|mean_last_k_tokens) ;;
    *) die 'POOLING must produce one vector per sentence: mean_all_tokens, last_token or mean_last_k_tokens.' ;;
  esac
  MODEL_TAG=$(printf '%s' "$MODEL_NAME" | sed -E 's/[^[:alnum:]_.-]+/__/g')
  VECTOR_DIR="outputs/value_vectors/$1"
  VECTOR_PREFIX="$VECTOR_DIR/value"
  ACTIVATION_CONFIG="$VECTOR_DIR/activations/${MODEL_TAG}_activations_config.json"
}
trust_remote_args() {
  TRUST_ARGS=()
  case "${TRUST_REMOTE_CODE:-0}" in
    0) ;;
    1) TRUST_ARGS+=(--trust_remote_code) ;;
    *) die 'TRUST_REMOTE_CODE must be 0 or 1.' ;;
  esac
}
