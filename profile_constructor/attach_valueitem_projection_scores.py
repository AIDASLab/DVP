#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attach per-value projection scores to annotated JSONL records.

Array loading:
- Preload all needed layer arrays from NPZ once (layer_arrays[lid] = (N,K) array)
- Avoid repeated z["layer_x"] lookups inside loops.

Progress:
- tqdm over input rows
- optional --count_rows to pre-count lines for a proper total

Errors:
- writes <out_jsonl>.errors.jsonl
- continues on per-row errors unless --fail_fast
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import zip_longest
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


def stream_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def count_jsonl_lines(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def load_columns_json(path: str) -> Tuple[List[str], str, str, List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    selected = list(obj["selected_value_items"])
    metric = str(obj.get("metric", "cosine"))
    quant = str(obj.get("quant", "float16"))
    layers = [int(x) for x in obj.get("layers", [])]
    if not layers:
        raise ValueError(f"[columns.json] missing/empty 'layers': {path}")
    return selected, metric, quant, layers


def load_meta_index(meta_jsonl: str) -> Dict[int, Tuple[Any, Any]]:
    m: Dict[int, Tuple[Any, Any]] = {}
    for r in stream_jsonl(meta_jsonl):
        rid = int(r["row_id"])
        m[rid] = (r.get("idx", None), r.get("text_type", None))
    return m


def decode_score(x: Any, quant: str) -> float:
    if quant == "int16":
        return float(np.int16(x)) / 10000.0
    return float(x)


def validate_strict_alignment(input_jsonl: str, meta_jsonl: str, text_key: str, expected_rows: int):
    """Check the scorer's native nonempty-row order before creating outputs."""
    def scored_rows():
        for row in stream_jsonl(input_jsonl):
            sent = row.get(text_key, None)
            if sent is not None and str(sent).strip():
                yield row, str(sent).strip()

    missing = object()
    count = 0
    for row_id, pair in enumerate(zip_longest(scored_rows(), stream_jsonl(meta_jsonl), fillvalue=missing)):
        source, meta = pair
        if source is missing or meta is missing:
            raise ValueError(f"Strict alignment row-count mismatch between input and metadata at row_id={row_id}")
        if row_id >= expected_rows:
            raise ValueError(f"Strict alignment has more input/metadata rows than NPZ N={expected_rows}")
        if int(meta.get("row_id", -1)) != row_id:
            raise ValueError(f"Strict metadata must contain each row_id exactly once in order; expected {row_id}")
        row, sentence = source
        if "target_sentence" not in meta or not isinstance(meta["target_sentence"], str):
            raise ValueError("Strict alignment requires full target_sentence metadata; do not score with --drop_text")
        if (row.get("idx", None), row.get("text_type", None), sentence) != (
            meta.get("idx", None), meta.get("text_type", None), meta["target_sentence"]
        ):
            raise ValueError(f"Strict metadata idx/text_type/sentence mismatch at row_id={row_id}")
        count += 1
    if count != expected_rows:
        raise ValueError(f"Strict alignment row-count mismatch: input/meta={count}, NPZ N={expected_rows}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input_jsonl", type=str, required=True)
    ap.add_argument("--scores_npz", type=str, required=True)
    ap.add_argument("--columns_json", type=str, required=True)
    ap.add_argument("--out_jsonl", type=str, required=True)

    ap.add_argument("--text_key", type=str, default="target_sentence")
    ap.add_argument("--value_items_key", type=str, default="value_items")
    ap.add_argument("--proj_field", type=str, default="proj_scores")

    ap.add_argument("--layers", type=str, default=None,
                    help="Comma-separated layer ids to attach. Default: columns_json['layers'].")

    ap.add_argument("--meta_jsonl", type=str, default=None)
    ap.add_argument("--strict_meta_check", action="store_true",
                    help="Before writing, require complete idx/type/sentence/row-count alignment. "
                         "Requires --meta_jsonl produced without --drop_text.")

    ap.add_argument("--skip_missing_value_items", action="store_true",
                    help="If set, silently skip value_items not found in selected_value_items. "
                         "Otherwise attach null for missing ones.")

    # tqdm / robustness
    ap.add_argument("--count_rows", default=True,
                    help="Pre-count input_jsonl lines to give tqdm a total (extra pass).")
    ap.add_argument("--fail_fast", action="store_true",
                    help="Stop immediately on first per-row error (default: log and continue).")
    ap.add_argument("--log_every", type=int, default=5000,
                    help="Update tqdm postfix every N processed rows.")

    args = ap.parse_args()

    selected_items, metric, quant, layers_in_cols = load_columns_json(args.columns_json)
    if len(set(selected_items)) != len(selected_items):
        raise ValueError("columns.json selected_value_items contains duplicate columns")
    if len(set(layers_in_cols)) != len(layers_in_cols):
        raise ValueError("columns.json layers contains duplicate layers")
    item_to_col = {v: i for i, v in enumerate(selected_items)}

    if args.layers is None:
        use_layers = layers_in_cols
    else:
        use_layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
        if not use_layers:
            raise ValueError("--layers provided but parsed empty")

    if not set(use_layers).issubset(layers_in_cols):
        raise ValueError("Requested layers must be listed in columns.json")
    if args.strict_meta_check and not args.meta_jsonl:
        raise ValueError("--strict_meta_check requires --meta_jsonl")

    if not os.path.exists(args.scores_npz):
        raise FileNotFoundError(args.scores_npz)

    # ---- Load NPZ and PRELOAD arrays for all layers (speed) ----
    layer_arrays: Dict[int, np.ndarray] = {}
    with np.load(args.scores_npz, allow_pickle=False) as z:
        expected_keys = {f"layer_{lid}" for lid in layers_in_cols}
        if {key for key in z.files if key.startswith("layer_")} != expected_keys:
            raise ValueError("NPZ layer keys do not match columns.json layers")
        expected_shape = None
        for lid in layers_in_cols:
            arr = z[f"layer_{lid}"]
            if arr.ndim != 2 or arr.shape[1] != len(selected_items):
                raise ValueError(f"layer_{lid} shape={arr.shape}; expected (N, {len(selected_items)})")
            if expected_shape is not None and arr.shape != expected_shape:
                raise ValueError(f"layer_{lid} shape={arr.shape} differs from other layers {expected_shape}")
            expected_shape = arr.shape
            if lid in use_layers:
                layer_arrays[lid] = arr
    N, K = expected_shape

    if args.strict_meta_check:
        validate_strict_alignment(args.input_jsonl, args.meta_jsonl, args.text_key, N)
    meta_map: Optional[Dict[int, Tuple[Any, Any]]] = None
    if args.meta_jsonl:
        meta_map = load_meta_index(args.meta_jsonl)

    # tqdm total
    total = count_jsonl_lines(args.input_jsonl) if args.count_rows else None

    err_path = args.out_jsonl + ".errors.jsonl"

    row_id = 0          # aligns with scoring rows (only non-empty text_key)
    written = 0
    skipped_empty = 0
    errors = 0

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as fout, open(err_path, "w", encoding="utf-8") as ferr:
        pbar = tqdm(stream_jsonl(args.input_jsonl), total=total, desc="AttachProjScores", unit="rows")

        processed = 0
        for r in pbar:
            processed += 1
            try:
                sent = r.get(args.text_key, None)
                if sent is None or str(sent).strip() == "":
                    skipped_empty += 1
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                    continue

                if row_id >= N:
                    raise RuntimeError(
                        f"Row alignment error: row_id={row_id} but npz has N={N}. "
                        f"Input likely differs from the one used to compute scores."
                    )

                if meta_map is not None and row_id in meta_map and args.strict_meta_check:
                    midx, mtype = meta_map[row_id]
                    if (r.get("idx", None), r.get("text_type", None)) != (midx, mtype):
                        raise RuntimeError(
                            f"Meta mismatch at row_id={row_id}: "
                            f"input(idx,text_type)={(r.get('idx',None), r.get('text_type',None))} "
                            f"meta(idx,text_type)={(midx,mtype)}"
                        )

                vitems = r.get(args.value_items_key, None)
                if isinstance(vitems, list):
                    for vi in vitems:
                        if not isinstance(vi, dict):
                            continue
                        name = vi.get("value_item", None)
                        if name is None:
                            continue
                        name = str(name)

                        col = item_to_col.get(name, None)
                        if col is None:
                            if args.skip_missing_value_items:
                                continue
                            vi[args.proj_field] = None
                            continue

                        proj: Dict[str, float] = {}
                        # Fast: layer_arrays[lid] already fetched
                        for lid in use_layers:
                            proj[str(lid)] = decode_score(layer_arrays[lid][row_id, col], quant)
                        vi[args.proj_field] = proj

                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                row_id += 1
                written += 1

            except Exception as e:
                errors += 1
                # log and continue (unless fail_fast)
                ferr.write(json.dumps(
                    {
                        "error": type(e).__name__,
                        "message": str(e),
                        "processed_row": processed,
                        "current_row_id": row_id,
                        "idx": r.get("idx", None),
                        "text_type": r.get("text_type", None),
                    },
                    ensure_ascii=False
                ) + "\n")
                if args.fail_fast:
                    raise

            if processed % args.log_every == 0:
                pbar.set_postfix({
                    "row_id": row_id,
                    "written": written,
                    "skip_empty": skipped_empty,
                    "errors": errors,
                })

    print("[OK] wrote:", args.out_jsonl)
    print("[OK] errors:", err_path)
    print(f"[INFO] metric={metric} quant={quant} layers={use_layers}")
    print(f"[INFO] scored_rows_written={written} | row_id_end={row_id} | npz_N={N}")
    print(f"[INFO] empty_or_missing_text_passed_through={skipped_empty} | errors={errors}")


if __name__ == "__main__":
    main()
