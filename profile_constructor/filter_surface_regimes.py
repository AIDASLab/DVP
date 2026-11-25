#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filter_surface_regimes.py

Filtering behavior:
- Exclude the non-invocation category ("Not Invoked") from:
    (a) threshold sum calculation
    (b) output categories list (removed)
    (c) proportions (computed over non-"Not Invoked" counts)

Keep row if:
  sum(counts over categories excluding "Not Invoked") > threshold

Adds:
  - per-category "proportion" (over the same non-"Not Invoked" total)
  - "total_count" (the non-"Not Invoked" total used for filtering)

The non-invocation category is the last category of each value item, as in
context_parser.context_effect_ot. Most catalog entries name it "Not Invoked",
but some use value-specific labels (e.g. "Progress Not Invoked"), so it is
identified by position rather than by name.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional


def _ensure_dir(path: str) -> None:
    os.makedirs(path or ".", exist_ok=True)


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise ValueError(f"JSON parse error in {path} at line {lineno}: {e}") from e


def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as w:
        for obj in rows:
            w.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _is_counted_profile_file(name: str) -> bool:
    return name.endswith(".surface_regimes.counted.jsonl")


def _to_int(x: Any) -> int:
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)):
        return int(x)
    return 0


def process_file(
    in_path: str,
    out_path: str,
    threshold: int,
    round_digits: Optional[int],
) -> Dict[str, int]:
    kept = 0
    total = 0

    def gen():
        nonlocal kept, total
        for obj in _iter_jsonl(in_path):
            total += 1
            cats = obj.get("categories")
            if not isinstance(cats, list):
                continue

            # 1) total over invoked categories (all but the last, non-invocation one)
            invoked_cats = [c for c in cats if isinstance(c, dict)][:-1]
            total_non_notinv = sum(_to_int(c.get("count", 0)) for c in invoked_cats)

            # 2) filter by total_non_notinv
            if total_non_notinv <= threshold:
                continue

            # 3) build new categories list WITHOUT "Not Invoked", add proportions
            new_cats: List[Any] = []
            for c in invoked_cats:
                cnt_i = _to_int(c.get("count", 0))
                prop = (cnt_i / total_non_notinv) if total_non_notinv > 0 else 0.0
                if round_digits is not None:
                    prop = round(prop, round_digits)

                c2 = dict(c)
                c2["proportion"] = prop
                new_cats.append(c2)

            obj2 = dict(obj)
            obj2["categories"] = new_cats
            obj2["total_count"] = total_non_notinv  # total excluding "Not Invoked"

            kept += 1
            yield obj2

    _write_jsonl(out_path, gen())
    return {"total_rows": total, "kept_rows": kept}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="outputs/distributional_profiles/surface_regimes", help="Folder containing *.surface_regimes.counted.jsonl")
    ap.add_argument(
        "--out_dir",
        default="outputs/distributional_profiles/surface_regimes_filtered",
        help="Output folder",
    )
    ap.add_argument("--threshold", type=int, default=200, help="Keep rows where NON-'Not Invoked' sum(counts) > threshold")
    ap.add_argument(
        "--round_digits",
        type=int,
        default=4,
        help="Round proportions to this many digits (set -1 to disable rounding)",
    )
    args = ap.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    out_dir = os.path.abspath(args.out_dir)
    _ensure_dir(out_dir)

    round_digits = None if args.round_digits < 0 else args.round_digits

    files = sorted([fn for fn in os.listdir(in_dir) if _is_counted_profile_file(fn)])
    if not files:
        raise FileNotFoundError(f"No *.surface_regimes.counted.jsonl files found in: {in_dir}")

    grand_total = 0
    grand_kept = 0

    for fn in files:
        in_path = os.path.join(in_dir, fn)
        out_name = fn.replace(".counted.jsonl", f".filtered_noNotInv_gt{args.threshold}.jsonl")
        out_path = os.path.join(out_dir, out_name)

        stats = process_file(in_path, out_path, threshold=args.threshold, round_digits=round_digits)
        grand_total += stats["total_rows"]
        grand_kept += stats["kept_rows"]
        print(f"[OK] {fn} -> {out_name}  kept {stats['kept_rows']}/{stats['total_rows']}")

    print(f"[DONE] total kept {grand_kept}/{grand_total} rows across {len(files)} files")
    print(f"[DONE] outputs in: {out_dir}")


if __name__ == "__main__":
    main()
