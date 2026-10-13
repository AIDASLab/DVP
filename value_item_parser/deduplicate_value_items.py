#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deduplicate_value_items.py

Supports BOTH:
  (A) folder -> folder (batch)
  (B) single file -> single file

Mapping logic:
1) If value_item exists in mapping_jsonl:
     - pick one of its representatives (random choice)
2) Else (if --allow_embedding):
     - embed value_item
     - nearest neighbor among representative list (rep_json)
     - replace value_item with that representative
     - also add (value_item -> [representative]) into mapping_jsonl (persisted)

Input JSONL format expects:
  - key: "value_items" as list of dicts like {"value_item": "...", "score": ...}
Other fields are preserved as-is.

Examples:
  # Folder mode
  python deduplicate_value_items.py \
    --input_dir /path/in_dir --output_dir /path/out_dir --allow_embedding

  # Single file mode
  python deduplicate_value_items.py \
    --input_jsonl /path/in.jsonl --output_jsonl /path/out.jsonl --allow_embedding
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm


# -----------------------
# Helpers
# -----------------------
def norm_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    s = " ".join(s.split())
    return s

def load_rep_items(rep_path: Path) -> List[str]:
    """
    Supports BOTH:
      - JSON list: ["a", "b", ...]
      - JSON dict: {"a": "...definition...", "b": "...", ...}  (keys are reps)
    Returns a de-duplicated, normalized list preserving order.
    """
    obj = read_json(rep_path)

    if isinstance(obj, list):
        items = [norm_str(x) for x in obj]
    elif isinstance(obj, dict):
        items = [norm_str(k) for k in obj.keys()]
    else:
        raise ValueError(
            f"rep_json must be either a JSON list or JSON dict, got {type(obj)}: {rep_path}"
        )

    items = merge_unique_keep_order(items)
    if len(items) == 0:
        raise ValueError(f"Representative list is empty after normalization: {rep_path}")
    return items


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, records_iter):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in records_iter:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def merge_unique_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        x = norm_str(x)
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


# -----------------------
# Mapping load/save
# -----------------------
def load_mapping_jsonl(mapping_path: Path) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    if not mapping_path.exists():
        return mapping

    for obj in iter_jsonl(mapping_path):
        vi = norm_str(obj.get("value_item"))
        if not vi:
            continue

        reps = obj.get("representatives", obj.get("represents", obj.get("rep_list", [])))
        if not isinstance(reps, list):
            reps = []
        reps = merge_unique_keep_order([norm_str(r) for r in reps])

        obj2 = dict(obj)
        obj2["value_item"] = vi
        obj2["representatives"] = reps
        mapping[vi] = obj2

    return mapping


def save_mapping_jsonl(mapping: Dict[str, Dict[str, Any]], mapping_path: Path):
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    items = sorted(mapping.items(), key=lambda kv: kv[0])
    with mapping_path.open("w", encoding="utf-8") as f:
        for _, rec in items:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# -----------------------
# Embedding (SentenceTransformers)
# -----------------------
def load_embedder(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    return model


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, eps, None)


def embed_texts(model, texts: List[str], batch_size: int = 32) -> np.ndarray:
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    if embs.dtype != np.float32:
        embs = embs.astype(np.float32)
    return embs


def load_or_build_rep_cache(
    rep_items: List[str],
    cache_path: Path,
    model,
    batch_size: int,
) -> Tuple[List[str], np.ndarray]:
    """
    Cache format:
      npz with fields:
        - reps: object array of strings
        - embs: float32 [N, D]
    """
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        cached_reps = [norm_str(x) for x in data["reps"].tolist()]
        cached_embs = data["embs"]
        if cached_reps == rep_items and isinstance(cached_embs, np.ndarray) and cached_embs.shape[0] == len(rep_items):
            return cached_reps, cached_embs

    embs = embed_texts(model, rep_items, batch_size=batch_size)
    embs = l2_normalize(embs).astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, reps=np.array(rep_items, dtype=object), embs=embs)
    return rep_items, embs


def nearest_rep(query_emb: np.ndarray, rep_embs: np.ndarray) -> int:
    # cosine sim because both normalized
    sims = rep_embs @ query_emb
    return int(np.argmax(sims))


# -----------------------
# Core mapping logic
# -----------------------
class ValueItemDeduplicator:
    def __init__(
        self,
        rep_items: List[str],
        mapping: Dict[str, Dict[str, Any]],
        rep_embs: Optional[np.ndarray] = None,
        embedder=None,
        seed: int = 42,
        embed_batch_size: int = 32,
        allow_embedding: bool = True,
    ):
        self.rep_items = rep_items
        self.mapping = mapping
        self.rep_embs = rep_embs
        self.embedder = embedder
        self.seed = seed
        self.embed_batch_size = embed_batch_size
        self.allow_embedding = allow_embedding
        self._new_query_cache: Dict[str, str] = {}
        random.seed(seed)

    def map_value_item(self, vi: str) -> Tuple[str, str]:
        """
        Returns:
          (mapped_value_item, mode)
          mode in {"existing_mapping", "embedding", "embedding_cached", "no_change"}
        """
        vi = norm_str(vi)
        if not vi:
            return vi, "no_change"

        # 1) existing mapping?
        if vi in self.mapping:
            reps = self.mapping[vi].get("representatives", [])
            if isinstance(reps, list) and len(reps) > 0:
                chosen = norm_str(random.choice(reps)) or norm_str(reps[0])
                if chosen:
                    return chosen, "existing_mapping"

        # 2) embedding-based
        if not self.allow_embedding:
            return vi, "no_change"

        if vi in self._new_query_cache:
            return self._new_query_cache[vi], "embedding_cached"

        if self.embedder is None or self.rep_embs is None:
            raise RuntimeError("Embedding requested but embedder/rep_embs are not initialized.")

        q_emb = embed_texts(self.embedder, [vi], batch_size=1)
        q_emb = l2_normalize(q_emb)[0]
        idx = nearest_rep(q_emb, self.rep_embs)
        chosen = self.rep_items[idx]

        self._new_query_cache[vi] = chosen

        # write back to mapping in-memory
        self.mapping[vi] = {
            "value_item": vi,
            "representatives": [chosen],
            "meta": {
                "created_by": "embedding_nearest_rep",
                "embedding_model": getattr(self.embedder, "model_card", None) or "",
            },
        }
        return chosen, "embedding"


def rewrite_one_jsonl(
    input_jsonl: Path,
    output_jsonl: Path,
    dedup: ValueItemDeduplicator,
    debug: bool = False,
    debug_max_items: int = 200,
) -> Dict[str, int]:
    stats = {
        "n_rows": 0,
        "n_rows_missing_value_items": 0,
        "n_rows_bad_value_items_type": 0,
        "n_rows_empty_value_items": 0,
        "n_value_items_seen": 0,
        "n_value_item_objs_bad_type": 0,
        "n_value_item_missing_key": 0,
        "n_mapped_existing": 0,
        "n_mapped_embedding": 0,
        "n_mapped_embedding_cached": 0,
        "n_kept_no_change": 0,
    }

    dbg_count = 0

    def dprint(msg: str):
        nonlocal dbg_count
        if not debug:
            return
        if dbg_count >= debug_max_items:
            return
        print(msg, flush=True)
        dbg_count += 1

    def out_iter():
        for row in iter_jsonl(input_jsonl):
            stats["n_rows"] += 1
            vis = row.get("value_items", None)

            if vis is None:
                stats["n_rows_missing_value_items"] += 1
                yield row
                continue

            if not isinstance(vis, list):
                stats["n_rows_bad_value_items_type"] += 1
                yield row
                continue

            if len(vis) == 0:
                stats["n_rows_empty_value_items"] += 1
                yield row
                continue

            new_vis = []
            for obj in vis:
                if not isinstance(obj, dict):
                    stats["n_value_item_objs_bad_type"] += 1
                    new_vis.append(obj)
                    continue

                vi_raw = obj.get("value_item", None)
                vi = norm_str(vi_raw)
                if not vi:
                    stats["n_value_item_missing_key"] += 1
                    new_vis.append(obj)
                    continue

                stats["n_value_items_seen"] += 1

                mapped, mode = dedup.map_value_item(vi)
                if mode == "existing_mapping":
                    stats["n_mapped_existing"] += 1
                elif mode == "embedding":
                    stats["n_mapped_embedding"] += 1
                elif mode == "embedding_cached":
                    stats["n_mapped_embedding_cached"] += 1
                else:
                    stats["n_kept_no_change"] += 1

                if debug:
                    dprint(f"[{input_jsonl.name}] {vi!r} -> {mapped!r} ({mode})")

                obj2 = dict(obj)
                obj2["value_item"] = mapped
                new_vis.append(obj2)

            row2 = dict(row)
            row2["value_items"] = new_vis
            yield row2

    write_jsonl(output_jsonl, out_iter())
    return stats


def main():
    ap = argparse.ArgumentParser()

    # ---- Input/Output: folder OR single file ----
    ap.add_argument("--input_dir", type=str, default=None, help="Folder containing JSONL files to rewrite.")
    ap.add_argument("--output_dir", type=str, default=None, help="Folder to write rewritten JSONL files (same filenames).")
    ap.add_argument("--input_jsonl", type=str, default=None, help="Single input JSONL file path.")
    ap.add_argument("--output_jsonl", type=str, default=None, help="Single output JSONL file path.")

    # ---- Dictionaries ----
    ap.add_argument(
        "--rep_json",
        type=str,
        default="resources/original/unique_value_items.json",
        help="JSON list of representative value items.",
    )
    ap.add_argument(
        "--mapping_jsonl",
        type=str,
        default="data/value_item_mapping_dictionary.jsonl",
        help="JSONL mapping dictionary: value_item -> representatives list (updated in-place).",
    )

    # ---- Embedding ----
    ap.add_argument("--embedding_model", type=str, default="Qwen/Qwen3-Embedding-8B",
                    help="SentenceTransformer embedding model name/path.")
    ap.add_argument("--device", type=str, default="cuda", help="cuda / cpu (or cuda:0).")
    ap.add_argument("--embed_batch_size", type=int, default=32,
                    help="Batch size for representative embedding computation.")
    ap.add_argument(
        "--rep_cache_npz",
        type=str,
        default="data/rep_embeddings_qwen3_embedding_8b.npz",
        help="Where to cache representative embeddings.",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed (used when multiple representatives exist).")

    # Folder mode file pattern
    ap.add_argument("--pattern", type=str, default="*.jsonl", help="Glob pattern for input files in input_dir.")

    ap.add_argument(
        "--allow_embedding",
        action="store_true",
        help="If set, unknown value items are mapped via embedding NN. If not set, unknowns are left unchanged.",
    )

    # Debug
    ap.add_argument("--debug", action="store_true", help="Print mapping decisions.")
    ap.add_argument("--debug_max_items", type=int, default=200, help="Max debug lines.")

    args = ap.parse_args()

    # ---- Validate mode selection ----
    folder_mode = (args.input_dir is not None) or (args.output_dir is not None)
    file_mode = (args.input_jsonl is not None) or (args.output_jsonl is not None)

    if folder_mode and file_mode:
        raise ValueError("Choose ONE mode: (input_dir/output_dir) OR (input_jsonl/output_jsonl), not both.")

    if not folder_mode and not file_mode:
        raise ValueError("Specify either folder mode (--input_dir --output_dir) or file mode (--input_jsonl --output_jsonl).")

    if folder_mode:
        if args.input_dir is None or args.output_dir is None:
            raise ValueError("Folder mode requires BOTH --input_dir and --output_dir.")
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"input_dir not found: {input_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        if args.input_jsonl is None:
            raise ValueError("File mode requires --input_jsonl.")
        input_jsonl = Path(args.input_jsonl)
        if not input_jsonl.exists():
            raise FileNotFoundError(f"input_jsonl not found: {input_jsonl}")

        if args.output_jsonl is None:
            # sensible default: add suffix before extension
            output_jsonl = input_jsonl.with_name(input_jsonl.stem + "_dedup.jsonl")
        else:
            output_jsonl = Path(args.output_jsonl)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rep_json = Path(args.rep_json)
    mapping_jsonl = Path(args.mapping_jsonl)
    rep_cache_npz = Path(args.rep_cache_npz)

    if not rep_json.exists():
        raise FileNotFoundError(f"rep_json not found: {rep_json}")

    # 1) load representatives (supports list OR dict-with-definitions)
    rep_items = load_rep_items(rep_json)

    # 2) load mapping
    mapping = load_mapping_jsonl(mapping_jsonl)

    # 3) load embedder + rep embeddings cache (only if allow_embedding)
    embedder = None
    rep_embs = None
    if args.allow_embedding:
        embedder = load_embedder(args.embedding_model, device=args.device)
        rep_items, rep_embs = load_or_build_rep_cache(
            rep_items=rep_items,
            cache_path=rep_cache_npz,
            model=embedder,
            batch_size=args.embed_batch_size,
        )

    dedup = ValueItemDeduplicator(
        rep_items=rep_items,
        mapping=mapping,
        rep_embs=rep_embs,
        embedder=embedder,
        seed=args.seed,
        embed_batch_size=args.embed_batch_size,
        allow_embedding=args.allow_embedding,
    )

    all_stats = {}

    # 4) process input(s)
    if folder_mode:
        input_files = sorted(Path(args.input_dir).glob(args.pattern))
        if len(input_files) == 0:
            raise FileNotFoundError(f"No files matched {args.pattern} in {args.input_dir}")

        for fp in tqdm(input_files, desc="Files"):
            out_fp = Path(args.output_dir) / fp.name
            stats = rewrite_one_jsonl(
                input_jsonl=fp,
                output_jsonl=out_fp,
                dedup=dedup,
                debug=args.debug,
                debug_max_items=args.debug_max_items,
            )
            all_stats[fp.name] = stats
    else:
        stats = rewrite_one_jsonl(
            input_jsonl=input_jsonl,
            output_jsonl=output_jsonl,
            dedup=dedup,
            debug=args.debug,
            debug_max_items=args.debug_max_items,
        )
        all_stats[input_jsonl.name] = stats

    # 5) normalize & save updated mapping
    for vi, rec in list(mapping.items()):
        rec["value_item"] = norm_str(rec.get("value_item", vi))
        reps = rec.get("representatives", [])
        if not isinstance(reps, list):
            reps = []
        rec["representatives"] = merge_unique_keep_order([norm_str(r) for r in reps])

    save_mapping_jsonl(mapping, mapping_jsonl)

    # ---- Print summary ----
    if folder_mode:
        print("[OK] Mode: folder")
        print("[OK] Input dir:", str(Path(args.input_dir)))
        print("[OK] Output dir:", str(Path(args.output_dir)))
    else:
        print("[OK] Mode: file")
        print("[OK] Input file:", str(input_jsonl))
        print("[OK] Output file:", str(output_jsonl))

    print("[OK] Updated mapping JSONL:", str(mapping_jsonl))
    if args.allow_embedding:
        print("[OK] Representative embedding cache:", str(rep_cache_npz))

    total = {
        "files": len(all_stats),
        "rows": 0,
        "value_items_seen": 0,
        "mapped_existing": 0,
        "mapped_embedding": 0,
        "mapped_embedding_cached": 0,
    }
    for _, st in all_stats.items():
        total["rows"] += st["n_rows"]
        total["value_items_seen"] += st["n_value_items_seen"]
        total["mapped_existing"] += st["n_mapped_existing"]
        total["mapped_embedding"] += st["n_mapped_embedding"]
        total["mapped_embedding_cached"] += st["n_mapped_embedding_cached"]

    print("[Stats summary]")
    print(json.dumps(total, ensure_ascii=False))


if __name__ == "__main__":
    main()
