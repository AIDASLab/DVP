#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calculate_activation_intensity.py

Compute value-intensity projections for target_sentence activations.

Key: avoid gigantic wide CSV.
Recommended output:
  --format npz
    <base>.meta.jsonl        (one per example)
    <base>.columns.json      (value_items column order)
    <base>.scores.npz        (layer->(N,K) numeric matrix, float16/int16)
OR fallback:
  --format csv_gz            (wide csv but gzip-compressed)

Selected K:
  unique_value_item_definitions.json (~1500) intersected with value_vectors_index.jsonl

Activations:
  uses activation_parser/activation_extractor.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
from typing import Any, Dict, Iterable, List, Tuple, Optional

import numpy as np
from tqdm import tqdm


# -----------------------------
# IO helpers
# -----------------------------
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


def load_value_index_jsonl(path: str) -> List[str]:
    items: List[Tuple[int, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            items.append((int(r["row_index"]), str(r["value_item"])))
    items.sort(key=lambda x: x[0])
    return [vi for _, vi in items]


def load_value_definitions_json(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict in {path}, got {type(obj)}")
    return {str(k): str(v) for k, v in obj.items()}


def parse_layers(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def resolve_input_paths(inputs: List[str]) -> List[str]:
    out: List[str] = []
    for p in inputs:
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.endswith(".jsonl"):
                    out.append(os.path.join(p, name))
        else:
            out.append(p)
    seen = set()
    uniq: List[str] = []
    for p in out:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            uniq.append(ap)
    return uniq


def base_name_no_ext(path: str) -> str:
    b = os.path.basename(path)
    return b[:-5] if b.endswith(".jsonl") else b


def sanitize_header(s: str, max_len: int = 120) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\-\.\: ]+", "_", s)
    s = s.replace(" ", "_")
    return s[:max_len] if len(s) > max_len else s


# -----------------------------
# math
# -----------------------------
def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def compute_scores(act: np.ndarray, vv: np.ndarray, metric: str) -> np.ndarray:
    # act: (B,H), vv: (K,H) (pre-normalized if cosine)
    if metric == "dot":
        return act @ vv.T
    if metric == "cosine":
        act_n = l2_normalize(act, axis=1)
        return act_n @ vv.T
    raise ValueError(f"Unknown metric: {metric}")


def quantize_scores(x: np.ndarray, quant: str) -> np.ndarray:
    """
    quant:
      - "float16": store as float16
      - "int16": store as int16 after scaling by 10000 (4 decimals)
    """
    if quant == "float16":
        return x.astype(np.float16)
    if quant == "int16":
        # cosine is in [-1,1], dot could be larger; int16 may overflow if dot is big.
        # Scale cosine scores from [-1, 1] to [-10000, 10000].
        y = np.rint(np.clip(x, -1.0, 1.0) * 10000.0).astype(np.int16)
        return y
    raise ValueError(f"Unknown quant: {quant}")


# -----------------------------
# core per file
# -----------------------------
def project_one_file_npz(
    input_jsonl: str,
    out_dir: str,
    text_key: str,
    layers: List[int],
    bundle,
    vv_by_layer: Dict[int, np.ndarray],   # lid -> (K,H)
    selected_items: List[str],            # K
    metric: str,
    pooling: str,
    last_k: int,
    max_length: int,
    batch_size: int,
    total_rows: Optional[int],
    drop_text: bool,
    quant: str,
):
    if __package__:
        from .activation_extractor import extract_activations_batch
    else:
        from activation_extractor import extract_activations_batch

    base = base_name_no_ext(input_jsonl)
    meta_path = os.path.join(out_dir, base + ".meta.jsonl")
    cols_path = os.path.join(out_dir, base + ".columns.json")
    scores_path = os.path.join(out_dir, base + ".scores.npz")

    os.makedirs(out_dir, exist_ok=True)

    # write columns once (order matters)
    with open(cols_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_value_items": selected_items,
                "metric": metric,
                "quant": quant,
                "layers": layers,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Build per-layer score arrays in chunks to limit peak memory use.
    # For simplicity and good performance: accumulate as list of chunks (each chunk is (chunkN,K) in float16/int16)
    # then concat once per layer at end. This uses RAM proportional to one layer's scores, not CSV-size.
    layer_chunks: Dict[int, List[np.ndarray]] = {lid: [] for lid in layers}

    buf_rows: List[Dict[str, Any]] = []
    buf_texts: List[str] = []

    n_written = 0

    with open(meta_path, "w", encoding="utf-8") as fmeta:
        pbar = tqdm(
            stream_jsonl(input_jsonl),
            total=total_rows,
            desc=f"Projecting(NPZ) {os.path.basename(input_jsonl)}",
            unit="rows",
        )

        def flush_batch():
            nonlocal buf_rows, buf_texts, n_written
            if not buf_rows:
                return

            acts_t = extract_activations_batch(
                bundle=bundle,
                texts=buf_texts,
                layer_list=layers,
                pooling=pooling,
                last_k=last_k,
                max_length=max_length,
                add_special_tokens=False,
            )
            acts_np: Dict[int, np.ndarray] = {lid: t.numpy() for lid, t in acts_t.items()}  # (B,H)

            B = len(buf_rows)

            # meta: one row per example (not per layer)
            for b in range(B):
                r = buf_rows[b]
                meta = {
                    "row_id": n_written + b,
                    "idx": r.get("idx", None),
                    "text_type": r.get("text_type", None),
                }
                if not drop_text:
                    meta["target_sentence"] = buf_texts[b]
                fmeta.write(json.dumps(meta, ensure_ascii=False) + "\n")

            # scores per layer
            for lid in layers:
                scores = compute_scores(acts_np[lid], vv_by_layer[lid], metric)  # (B,K)
                layer_chunks[lid].append(quantize_scores(scores, quant))

            n_written += B
            buf_rows, buf_texts = [], []

        for r in pbar:
            sent = r.get(text_key, None)
            if sent is None:
                continue
            sent = str(sent).strip()
            if not sent:
                continue

            buf_rows.append(r)
            buf_texts.append(sent)

            if len(buf_rows) >= batch_size:
                flush_batch()

        if buf_rows:
            flush_batch()

    # finalize: concat per layer and save to npz
    to_save: Dict[str, np.ndarray] = {}
    for lid in layers:
        if not layer_chunks[lid]:
            arr = np.zeros((0, len(selected_items)), dtype=np.float16 if quant == "float16" else np.int16)
        else:
            arr = np.concatenate(layer_chunks[lid], axis=0)
        to_save[f"layer_{lid}"] = arr

    np.savez_compressed(scores_path, **to_save)

    print(f"[OK] saved:")
    print(f"  meta:   {meta_path}")
    print(f"  cols:   {cols_path}")
    print(f"  scores: {scores_path}")
    print(f"[INFO] N={n_written}, K={len(selected_items)}, layers={layers}, quant={quant}")


def project_one_file_csv_gz(
    input_jsonl: str,
    out_dir: str,
    text_key: str,
    layers: List[int],
    bundle,
    vv_by_layer: Dict[int, np.ndarray],   # lid -> (K,H)
    value_cols: List[str],                # K, already sanitized "score__..."
    metric: str,
    pooling: str,
    last_k: int,
    max_length: int,
    batch_size: int,
    total_rows: Optional[int],
    round_digits: int,
    drop_text: bool,
):
    if __package__:
        from .activation_extractor import extract_activations_batch
    else:
        from activation_extractor import extract_activations_batch

    base = base_name_no_ext(input_jsonl)
    out_csv_gz = os.path.join(out_dir, base + ".csv.gz")
    os.makedirs(out_dir, exist_ok=True)

    header = ["idx", "text_type", "layer"] + ([] if drop_text else ["target_sentence"]) + value_cols
    K = len(value_cols)

    buf_rows: List[Dict[str, Any]] = []
    buf_texts: List[str] = []

    def fmt(x: float) -> str:
        r = round(float(x), round_digits)
        if r == 0.0:
            return "0"
        s = f"{r:.{round_digits}f}".rstrip("0").rstrip(".")
        return s if s != "-0" else "0"

    def flush_batch(writer: csv.DictWriter):
        nonlocal buf_rows, buf_texts
        if not buf_rows:
            return

        acts_t = extract_activations_batch(
            bundle=bundle,
            texts=buf_texts,
            layer_list=layers,
            pooling=pooling,
            last_k=last_k,
            max_length=max_length,
            add_special_tokens=False,
        )
        acts_np: Dict[int, np.ndarray] = {lid: t.numpy() for lid, t in acts_t.items()}

        B = len(buf_rows)
        for lid in layers:
            scores = compute_scores(acts_np[lid], vv_by_layer[lid], metric)  # (B,K)
            for b in range(B):
                r = buf_rows[b]
                out: Dict[str, Any] = {
                    "idx": r.get("idx", None),
                    "text_type": r.get("text_type", None),
                    "layer": lid,
                }
                if not drop_text:
                    out["target_sentence"] = buf_texts[b]
                row_scores = scores[b]
                for j in range(K):
                    out[value_cols[j]] = fmt(float(row_scores[j]))
                writer.writerow(out)

        buf_rows, buf_texts = [], []

    with gzip.open(out_csv_gz, "wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        pbar = tqdm(
            stream_jsonl(input_jsonl),
            total=total_rows,
            desc=f"Projecting(CSV.GZ,K={K}) {os.path.basename(input_jsonl)}",
            unit="rows",
        )
        for r in pbar:
            sent = r.get(text_key, None)
            if sent is None:
                continue
            sent = str(sent).strip()
            if not sent:
                continue
            buf_rows.append(r)
            buf_texts.append(sent)
            if len(buf_rows) >= batch_size:
                flush_batch(writer)
        if buf_rows:
            flush_batch(writer)

    print(f"[OK] saved: {out_csv_gz}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input_jsonls", type=str, nargs="+", required=True,
                    help="JSONL paths and/or directories (collect *.jsonl).")
    ap.add_argument("--text_key", type=str, default="target_sentence")

    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--device_map", type=str, default="auto")

    ap.add_argument("--layers", type=str, required=True)
    ap.add_argument("--pooling", type=str, default="mean_last_k_tokens",
                    choices=["mean_all_tokens", "last_token", "mean_last_k_tokens"])
    ap.add_argument("--last_k", type=int, default=3)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)

    ap.add_argument("--value_vectors_npz", type=str, required=True)
    ap.add_argument("--value_index_jsonl", type=str, required=True)
    ap.add_argument("--dedup_value_defs_json", type=str, required=True)
    ap.add_argument("--activation_config", type=str, default=None,
                    help="Original *_activations_config.json used to produce the value-vector shards. "
                         "Validate model, layers, pooling and last_k before loading the model.")

    ap.add_argument("--metric", type=str, default="cosine", choices=["cosine", "dot"])

    ap.add_argument("--out_dir", type=str, required=True)

    # output control
    ap.add_argument("--format", type=str, default="npz", choices=["npz", "csv_gz"],
                    help="npz is strongly recommended for size.")
    ap.add_argument("--drop_text", action="store_true",
                    help="Do not store target_sentence in outputs (saves a lot).")
    ap.add_argument("--quant", type=str, default="float16", choices=["float16", "int16"],
                    help="Only for --format npz. int16 stores cosine*10000.")

    # rounding for csv_gz only
    ap.add_argument("--round_digits", type=int, default=4)

    # ETA
    ap.add_argument("--num_rows", type=int, default=None,
                    help="(Per-file) total rows if you already know it. If provided, used for all files.")
    ap.add_argument("--count_rows", action="store_true",
                    help="Pre-count JSONL lines for accurate tqdm ETA (extra pass per file).")

    args = ap.parse_args()

    input_paths = resolve_input_paths(args.input_jsonls)
    if not input_paths:
        raise ValueError("No input JSONL files found.")

    layers = parse_layers(args.layers)
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("--layers must contain distinct layer ids")
    if args.activation_config:
        with open(args.activation_config, "r", encoding="utf-8") as f:
            activation_config = json.load(f)
        for key, requested in (("model_name", args.model_name), ("layers", layers),
                               ("pooling", args.pooling), ("last_k", args.last_k)):
            if activation_config.get(key) != requested:
                raise ValueError(
                    f"Activation config mismatch for {key}: "
                    f"vector extraction={activation_config.get(key)!r}, scoring={requested!r}"
                )

    # full value index
    full_value_items = load_value_index_jsonl(args.value_index_jsonl)
    if len(set(full_value_items)) != len(full_value_items):
        raise ValueError("Value vector index contains duplicate value_items")
    full_item_to_index = {vi: i for i, vi in enumerate(full_value_items)}
    full_V = len(full_value_items)

    # dedup (~1500)
    dedup_defs = load_value_definitions_json(args.dedup_value_defs_json)
    dedup_items = list(dedup_defs.keys())

    selected_items = [vi for vi in dedup_items if vi in full_item_to_index]
    if not selected_items:
        raise ValueError("No overlap between dedup list and value vector index.")
    sel_idx = np.array([full_item_to_index[vi] for vi in selected_items], dtype=np.int64)
    K = len(selected_items)

    # load vv and slice once
    vv_by_layer: Dict[int, np.ndarray] = {}
    hidden_dim = None
    with np.load(args.value_vectors_npz) as z:
        for lid in layers:
            key = f"layer_{lid}"
            if key not in z.files:
                raise ValueError(f"Missing {key} in {args.value_vectors_npz}")
            vv_full = z[key].astype(np.float32, copy=False)  # (V,H)
            if vv_full.ndim != 2 or vv_full.shape[1] == 0:
                raise ValueError(f"{key} must have shape (V,H) with H > 0; got {vv_full.shape}")
            if vv_full.shape[0] != full_V:
                raise ValueError(f"{key} V mismatch: expected {full_V}, got {vv_full.shape[0]}")
            if hidden_dim is not None and vv_full.shape[1] != hidden_dim:
                raise ValueError(f"{key} hidden dimension {vv_full.shape[1]} differs from other layers {hidden_dim}")
            hidden_dim = vv_full.shape[1]
            vv = vv_full[sel_idx]  # (K,H)
            if args.metric == "cosine":
                vv = l2_normalize(vv, axis=1)
            vv_by_layer[lid] = vv

    # Model loading follows native-file checks, so mismatched inputs fail cheaply.
    if __package__:
        from .activation_extractor import load_model_and_tokenizer
    else:
        from activation_extractor import load_model_and_tokenizer

    bundle = load_model_and_tokenizer(
        model_name=args.model_name,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        device_map=args.device_map,
    )
    model_hidden_dim = getattr(bundle.model.config, "hidden_size", None)
    if model_hidden_dim is not None and model_hidden_dim != hidden_dim:
        raise ValueError(f"Model hidden_size={model_hidden_dim} does not match value vectors H={hidden_dim}")
    if (args.activation_config and args.pooling in ("last_token", "mean_last_k_tokens")
            and getattr(bundle.tokenizer, "padding_side", "right") != "right"):
        raise ValueError("Safe scoring requires right padding for last-token pooling")
    os.makedirs(args.out_dir, exist_ok=True)

    # headers for csv_gz
    value_cols = [f"score__{sanitize_header(v)}" for v in selected_items]

    print(f"[INFO] format={args.format} | inputs={len(input_paths)} | layers={layers} | K={K} | metric={args.metric}")
    if args.format == "npz":
        print(f"[INFO] npz quant={args.quant} | drop_text={args.drop_text}")

    for in_path in input_paths:
        total = args.num_rows
        if total is None and args.count_rows:
            total = count_jsonl_lines(in_path)

        if args.format == "npz":
            project_one_file_npz(
                input_jsonl=in_path,
                out_dir=args.out_dir,
                text_key=args.text_key,
                layers=layers,
                bundle=bundle,
                vv_by_layer=vv_by_layer,
                selected_items=selected_items,
                metric=args.metric,
                pooling=args.pooling,
                last_k=args.last_k,
                max_length=args.max_length,
                batch_size=args.batch_size,
                total_rows=total,
                drop_text=args.drop_text,
                quant=args.quant,
            )
        else:
            project_one_file_csv_gz(
                input_jsonl=in_path,
                out_dir=args.out_dir,
                text_key=args.text_key,
                layers=layers,
                bundle=bundle,
                vv_by_layer=vv_by_layer,
                value_cols=value_cols,
                metric=args.metric,
                pooling=args.pooling,
                last_k=args.last_k,
                max_length=args.max_length,
                batch_size=args.batch_size,
                total_rows=total,
                round_digits=args.round_digits,
                drop_text=args.drop_text,
            )


if __name__ == "__main__":
    main()
