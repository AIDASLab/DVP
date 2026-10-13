#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
save_value_vectors.py

Compute value vectors (mean(pos) - mean(neg)) per value_item per layer from:
  - <model_tag>_activations_meta.jsonl
  - <model_tag>_activations_shardXXXX.npz

Assumptions:
- meta jsonl lines include at least:
    value_item, label ("pos"/"neg"), shard_id, index_in_shard, layers (list)
- npz shard contains keys like "layer_16", "layer_24", ... each shaped (N, H)
  (pooling already applied at extraction time; this script expects (N,H) per layer)

Outputs:
- out_prefix.value_vectors.npz  (layer_<lid> -> (V,H))
- out_prefix.value_vectors_index.jsonl  (value_item -> row_index)
- out_prefix.value_vectors_stats.json   (counts, missing, config)

Example:
python -m activation_parser.save_value_vectors \
  --meta_jsonl outputs/activations/qwen3_32b/Qwen__Qwen3-32B_activations_meta.jsonl \
  --shard_glob "outputs/activations/qwen3_32b/Qwen__Qwen3-32B_activations_shard*.npz" \
  --out_prefix outputs/activations/qwen3_32b/Qwen__Qwen3-32B \
  --label_pos pos --label_neg neg
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Any, Optional

import numpy as np
from tqdm import tqdm


def stream_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def parse_shard_id_from_path(p: str) -> int:
    # expects ...shard0003.npz
    m = re.search(r"shard(\d+)\.npz$", os.path.basename(p))
    if not m:
        raise ValueError(f"Cannot parse shard id from filename: {p}")
    return int(m.group(1))


@dataclass
class MetaRow:
    value_item: str
    label: str
    shard_id: int
    index_in_shard: int


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_jsonl", type=str, required=True)
    ap.add_argument("--shard_glob", type=str, required=True,
                    help='e.g. "/path/model_activations_shard*.npz"')
    ap.add_argument("--out_prefix", type=str, required=True)

    ap.add_argument("--label_pos", type=str, default="pos")
    ap.add_argument("--label_neg", type=str, default="neg")

    ap.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    ap.add_argument("--allow_missing_neg_or_pos", action="store_true",
                    help="If set, still writes vectors; missing side treated as zero mean. "
                         "Stats will report missing counts.")
    args = ap.parse_args()

    out_npz = args.out_prefix + ".value_vectors.npz"
    out_index_jsonl = args.out_prefix + ".value_vectors_index.jsonl"
    out_stats_json = args.out_prefix + ".value_vectors_stats.json"

    shard_paths = sorted(glob.glob(args.shard_glob))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files matched: {args.shard_glob}")

    shard_ids = [parse_shard_id_from_path(p) for p in shard_paths]
    shard_id_to_path = {sid: p for sid, p in zip(shard_ids, shard_paths)}

    # -------- pass 1: read meta, build value_item index, and per-shard row lists ----------
    # shard_id -> list of MetaRow in the exact order index_in_shard
    shard_rows: Dict[int, List[Optional[MetaRow]]] = defaultdict(list)
    value_items_set = set()

    # We will store as an array-like with direct placement by index_in_shard.
    # Need max index_in_shard per shard.
    max_idx_per_shard: Dict[int, int] = defaultdict(lambda: -1)
    meta_count = 0

    for r in tqdm(stream_jsonl(args.meta_jsonl), desc="Reading meta", unit="rows"):
        meta_count += 1
        vi = r.get("value_item")
        lb = r.get("label")
        sid = r.get("shard_id")
        idx = r.get("index_in_shard")
        if vi is None or lb is None or sid is None or idx is None:
            continue
        vi = str(vi)
        lb = str(lb)
        sid = int(sid)
        idx = int(idx)

        value_items_set.add(vi)
        if idx > max_idx_per_shard[sid]:
            max_idx_per_shard[sid] = idx

        # We'll place later once we know the list size, so just temporarily store tuples
        shard_rows[sid].append((idx, MetaRow(value_item=vi, label=lb, shard_id=sid, index_in_shard=idx)))

    # materialize shard_rows into index-addressable lists
    shard_rows_indexed: Dict[int, List[Optional[MetaRow]]] = {}
    for sid, pairs in shard_rows.items():
        size = max_idx_per_shard[sid] + 1
        arr: List[Optional[MetaRow]] = [None] * size
        for idx, mr in pairs:
            if 0 <= idx < size:
                arr[idx] = mr
        shard_rows_indexed[sid] = arr

    value_items = sorted(list(value_items_set))
    V = len(value_items)
    value_item_to_row = {vi: i for i, vi in enumerate(value_items)}

    # -------- determine layers + hidden size by peeking first shard ----------
    peek_path = shard_paths[0]
    with np.load(peek_path) as z:
        layer_keys = sorted([k for k in z.files if k.startswith("layer_")],
                            key=lambda x: int(x.split("_", 1)[1]))
        if not layer_keys:
            raise ValueError(f"No layer_* arrays found in {peek_path}")
        # assume pooled (N,H)
        sample = z[layer_keys[0]]
        if sample.ndim != 2:
            raise ValueError(
                f"Expected pooled activations (N,H) but got shape {sample.shape} for {layer_keys[0]} in {peek_path}"
            )
        H = int(sample.shape[1])
        layer_ids = [int(k.split("_", 1)[1]) for k in layer_keys]

    # -------- accumulators: sums and counts for pos/neg per (layer, value_item) ----------
    # sums_pos[lid] shape (V,H), counts_pos shape (V,)
    sums_pos: Dict[int, np.ndarray] = {lid: np.zeros((V, H), dtype=np.float64) for lid in layer_ids}
    sums_neg: Dict[int, np.ndarray] = {lid: np.zeros((V, H), dtype=np.float64) for lid in layer_ids}
    cnt_pos = np.zeros((V,), dtype=np.int64)
    cnt_neg = np.zeros((V,), dtype=np.int64)

    # -------- pass 2: load shards and accumulate ----------
    for sid in tqdm(sorted(shard_ids), desc="Accumulating shards", unit="shard"):
        shard_path = shard_id_to_path.get(sid)
        if shard_path is None:
            continue
        meta_list = shard_rows_indexed.get(sid)
        if meta_list is None:
            # shard exists but no meta rows
            continue

        with np.load(shard_path) as z:
            # ensure all needed layers exist
            for lid in layer_ids:
                k = f"layer_{lid}"
                if k not in z.files:
                    raise ValueError(f"Missing {k} in shard {shard_path}")

            # iterate rows by index_in_shard
            # We load each layer array once for speed
            layer_arrays = {lid: z[f"layer_{lid}"] for lid in layer_ids}
            N = next(iter(layer_arrays.values())).shape[0]

            if len(meta_list) != N:
                # meta might not perfectly match due to filtering; handle safely:
                # iterate min length
                M = min(len(meta_list), N)
            else:
                M = N

            for j in range(M):
                mr = meta_list[j]
                if mr is None:
                    continue
                vi = mr.value_item
                lb = mr.label
                row = value_item_to_row.get(vi)
                if row is None:
                    continue

                if lb == args.label_pos:
                    cnt_pos[row] += 1
                    for lid in layer_ids:
                        sums_pos[lid][row] += layer_arrays[lid][j].astype(np.float64, copy=False)
                elif lb == args.label_neg:
                    cnt_neg[row] += 1
                    for lid in layer_ids:
                        sums_neg[lid][row] += layer_arrays[lid][j].astype(np.float64, copy=False)
                else:
                    # ignore other labels
                    continue

    # -------- compute means and value vectors ----------
    eps = 1e-12
    means_pos: Dict[int, np.ndarray] = {}
    means_neg: Dict[int, np.ndarray] = {}
    value_vecs: Dict[int, np.ndarray] = {}

    # For missing counts, division by zero guarded
    for lid in layer_ids:
        mp = sums_pos[lid] / np.maximum(cnt_pos[:, None], 1)
        mn = sums_neg[lid] / np.maximum(cnt_neg[:, None], 1)
        means_pos[lid] = mp
        means_neg[lid] = mn
        vv = mp - mn

        # if strict, require both sides for every item
        if not args.allow_missing_neg_or_pos:
            missing = (cnt_pos == 0) | (cnt_neg == 0)
            if missing.any():
                # fail with a clear message listing a few
                bad_items = [value_items[i] for i in np.where(missing)[0][:20]]
                raise RuntimeError(
                    f"Some value_items have missing pos/neg examples (pos==0 or neg==0). "
                    f"Set --allow_missing_neg_or_pos to proceed.\n"
                    f"Examples (up to 20): {bad_items}"
                )
            # else ok

        # cast final dtype
        if args.dtype == "float32":
            value_vecs[lid] = vv.astype(np.float32)
        else:
            value_vecs[lid] = vv.astype(np.float64)

    # -------- save outputs ----------
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)

    # NPZ with layer_<lid> arrays
    np.savez_compressed(out_npz, **{f"layer_{lid}": arr for lid, arr in value_vecs.items()})

    # index jsonl: one line per value_item
    with open(out_index_jsonl, "w", encoding="utf-8") as f:
        for i, vi in enumerate(value_items):
            f.write(json.dumps({"row_index": i, "value_item": vi}, ensure_ascii=False) + "\n")

    stats = {
        "meta_jsonl": os.path.abspath(args.meta_jsonl),
        "shard_glob": args.shard_glob,
        "num_meta_rows_read": meta_count,
        "num_shards_found": len(shard_paths),
        "shard_ids": sorted(shard_ids),
        "num_value_items": V,
        "hidden_dim": H,
        "layers": layer_ids,
        "label_pos": args.label_pos,
        "label_neg": args.label_neg,
        "count_pos_total": int(cnt_pos.sum()),
        "count_neg_total": int(cnt_neg.sum()),
        "num_value_items_missing_pos": int((cnt_pos == 0).sum()),
        "num_value_items_missing_neg": int((cnt_neg == 0).sum()),
        "num_value_items_missing_either": int(((cnt_pos == 0) | (cnt_neg == 0)).sum()),
        "out_npz": os.path.abspath(out_npz),
        "out_index_jsonl": os.path.abspath(out_index_jsonl),
        "dtype": args.dtype,
        "allow_missing_neg_or_pos": bool(args.allow_missing_neg_or_pos),
    }
    with open(out_stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved value vectors npz: {out_npz}")
    print(f"[OK] saved index jsonl:      {out_index_jsonl}")
    print(f"[OK] saved stats json:       {out_stats_json}")
    print(f"[INFO] missing either pos/neg: {stats['num_value_items_missing_either']} / {V}")


if __name__ == "__main__":
    main()
