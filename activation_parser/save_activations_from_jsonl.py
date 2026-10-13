#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
save_activations_from_jsonl.py

Load (value_item, label, sentence, ...) JSONL, extract layer activations, save.

Recommended output:
- NPZ shards for activations
- metadata JSONL with per-row pointers

Example:
python save_activations_from_jsonl.py \
  --input_jsonl /path/posneg_sentences.jsonl \
  --model_name Qwen/Qwen3-32B \
  --out_dir /path/outputs/activations_qwen3 \
  --layers 10,20,30 \
  --pooling mean_all_tokens \
  --batch_size 8 \
  --shard_size 2000 \
  --dtype bfloat16
"""

from __future__ import annotations

import os
import json
import re
import argparse
from typing import Dict, Any, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm


def sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^\w\-\.]+", "__", model_name)


def parse_layers_arg(layers_arg: Optional[str], num_layers: int) -> Optional[List[int]]:
    """
    None => all [0..L]
    "10,20,30"
    "0:-1" (slice style, end exclusive; supports negatives)
    "25" single
    """
    if layers_arg is None:
        return None

    s = layers_arg.strip()
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip() != ""]

    if ":" in s:
        a, b = s.split(":", 1)
        start = int(a) if a.strip() != "" else 0
        end = int(b) if b.strip() != "" else (num_layers + 1)
        # handle negatives like python
        if start < 0:
            start = (num_layers + 1) + start
        if end < 0:
            end = (num_layers + 1) + end
        start = max(0, start)
        end = min(num_layers + 1, end)
        return list(range(start, end))

    return [int(s)]


def stream_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_npz_shard(npz_path: str, shard_layers: Dict[int, np.ndarray]):
    # keys: layer_<id>
    np.savez_compressed(npz_path, **{f"layer_{k}": v for k, v in shard_layers.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", type=str, required=True)
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--require_fresh_output", action="store_true",
                    help="Reject a nonempty output directory before loading the model. "
                         "Extraction appends metadata and does not safely resume.")

    ap.add_argument("--layers", type=str, default=None,
                    help="None=all, '10,20,30', or slice '0:-1' (end exclusive).")
    ap.add_argument("--pooling", type=str, default="mean_all_tokens",
                    choices=["mean_all_tokens", "last_token", "mean_last_k_tokens", "all_tokens"])
    ap.add_argument("--last_k", type=int, default=3)
    ap.add_argument("--max_length", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--shard_size", type=int, default=2000,
                    help="How many rows per NPZ shard (ignored if --format jsonl).")

    ap.add_argument("--format", type=str, default="npz", choices=["npz", "jsonl"],
                    help="npz recommended; jsonl is huge (debug only).")

    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--device_map", type=str, default="auto")

    args = ap.parse_args()

    if args.require_fresh_output and os.path.exists(args.out_dir):
        if not os.path.isdir(args.out_dir):
            raise FileExistsError(f"Output path is not a directory: {args.out_dir}")
        with os.scandir(args.out_dir) as entries:
            if next(entries, None) is not None:
                raise FileExistsError(
                    f"--require_fresh_output needs an empty or new directory: {args.out_dir}. "
                    "Choose a new directory; extraction is not resumable."
                )

    if __package__:
        from .activation_extractor import load_model_and_tokenizer, extract_activations_batch
    else:
        from activation_extractor import load_model_and_tokenizer, extract_activations_batch

    # load model
    bundle = load_model_and_tokenizer(
        args.model_name,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        device_map=args.device_map,
    )
    if (args.require_fresh_output and args.pooling in ("last_token", "mean_last_k_tokens")
            and getattr(bundle.tokenizer, "padding_side", "right") != "right"):
        raise ValueError("Safe extraction requires right padding for last-token pooling")
    L = bundle.num_layers
    layer_list = parse_layers_arg(args.layers, L)
    os.makedirs(args.out_dir, exist_ok=True)

    model_tag = sanitize_model_name(args.model_name)
    meta_path = os.path.join(args.out_dir, f"{model_tag}_activations_meta.jsonl")
    cfg_path = os.path.join(args.out_dir, f"{model_tag}_activations_config.json")

    # save config once
    cfg = {
        "input_jsonl": os.path.abspath(args.input_jsonl),
        "model_name": args.model_name,
        "num_hidden_layers": L,
        "layers": layer_list if layer_list is not None else list(range(L + 1)),
        "pooling": args.pooling,
        "last_k": args.last_k,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "format": args.format,
        "shard_size": args.shard_size,
        "dtype": args.dtype,
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # iterate rows
    buffer_rows: List[Dict[str, Any]] = []
    buffer_texts: List[str] = []

    # shard buffers (npz mode)
    shard_layers_buf: Dict[int, List[np.ndarray]] = {}
    shard_row_meta: List[Dict[str, Any]] = []
    shard_id = 0
    global_i = 0

    def flush_batch(batch_rows: List[Dict[str, Any]], batch_texts: List[str]):
        nonlocal shard_layers_buf, shard_row_meta, shard_id, global_i

        acts = extract_activations_batch(
            bundle,
            batch_texts,
            layer_list=layer_list,
            pooling=args.pooling,
            last_k=args.last_k,
            max_length=args.max_length,
            add_special_tokens=False,
        )
        # acts[layer] -> torch tensor on CPU float32
        # convert to numpy
        acts_np: Dict[int, np.ndarray] = {lid: t.numpy() for lid, t in acts.items()}

        if args.format == "jsonl":
            out_rows = []
            for bi, row in enumerate(batch_rows):
                rec = dict(row)
                rec["_activation_pooling"] = args.pooling
                rec["_layers"] = sorted(list(acts_np.keys()))
                rec["_activations"] = {str(lid): acts_np[lid][bi].tolist() for lid in acts_np.keys()}
                out_rows.append(rec)
            write_jsonl(os.path.join(args.out_dir, f"{model_tag}_activations.jsonl"), out_rows)
            return

        # NPZ mode: append each sample to shard buffers
        for lid, arr in acts_np.items():
            if lid not in shard_layers_buf:
                shard_layers_buf[lid] = []
            shard_layers_buf[lid].append(arr)  # (B,H) or (B,T,H) per batch; arr contains B

        # create meta rows pointing to shard + local index
        B = len(batch_rows)
        # local index within shard after appending: we need to know base offset
        # compute current shard size BEFORE adding this batch:
        current_size = len(shard_row_meta)
        for bi in range(B):
            row = batch_rows[bi]
            shard_row_meta.append({
                "global_index": global_i,
                "value_item": row.get("value_item"),
                "label": row.get("label"),
                "gen_id": row.get("gen_id"),
                "sentence": row.get("sentence"),
                "run_id": row.get("run_id"),
                "seed": row.get("seed"),
                "temperature": row.get("temperature"),
                "top_p": row.get("top_p"),
                "value_item_index": row.get("value_item_index"),
                "pooling": args.pooling,
                "layers": (layer_list if layer_list is not None else list(range(L + 1))),
                "shard_id": shard_id,
                "index_in_shard": current_size + bi,
            })
            global_i += 1

        # flush shard if big enough
        if len(shard_row_meta) >= args.shard_size:
            flush_shard()

    def flush_shard():
        nonlocal shard_layers_buf, shard_row_meta, shard_id

        if not shard_row_meta:
            return

        shard_path = os.path.join(args.out_dir, f"{model_tag}_activations_shard{shard_id:04d}.npz")

        # stack buffers into arrays
        to_save: Dict[int, np.ndarray] = {}
        for lid, chunks in shard_layers_buf.items():
            # chunks is list of arrays each shaped (B,...) ; concatenate on axis 0
            to_save[lid] = np.concatenate(chunks, axis=0).astype(np.float32)

        save_npz_shard(shard_path, to_save)

        # append meta
        write_jsonl(meta_path, shard_row_meta)

        # reset
        shard_layers_buf = {}
        shard_row_meta = []
        shard_id += 1

    # main loop
    pbar = tqdm(stream_jsonl(args.input_jsonl), desc="Extracting", unit="rows")
    for row in pbar:
        # minimal required fields
        sent = row.get("sentence", None)
        if sent is None:
            continue

        buffer_rows.append(row)
        buffer_texts.append(str(sent))

        if len(buffer_rows) >= args.batch_size:
            flush_batch(buffer_rows, buffer_texts)
            buffer_rows = []
            buffer_texts = []

    # tail batch
    if buffer_rows:
        flush_batch(buffer_rows, buffer_texts)

    # tail shard
    if args.format == "npz":
        flush_shard()

    print(f"[OK] saved config: {cfg_path}")
    if args.format == "npz":
        print(f"[OK] saved meta jsonl: {meta_path}")
        print(f"[OK] saved shards under: {args.out_dir}")
    else:
        print(f"[OK] saved jsonl activations under: {args.out_dir}")


if __name__ == "__main__":
    main()
