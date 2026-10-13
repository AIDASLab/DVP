#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
construct_distributional_surface_regimes.py

For a single normalized surface regime label JSONL:
  - counts (value_item -> category -> occurrences) from "surface_value_regimes"
  - merges counts into the master regimes definition JSONL (surface_value_regimes_gpt.jsonl)
  - writes output JSONL with "count" added per category entry

Example:
python -m profile_constructor.construct_distributional_surface_regimes \
  --labels_jsonl outputs/surface_value_regime_labels/example.surface_regimes.normalized.jsonl \
  --regimes_jsonl resources/original/surface_value_regimes_gpt.jsonl \
  --out_jsonl outputs/distributional_profiles/surface_regimes/example.surface_regimes.counted.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, DefaultDict


def _ensure_dir(path: str) -> None:
    os.makedirs(path or ".", exist_ok=True)


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise ValueError(f"JSON parse error in {path} at line {lineno}: {e}") from e


def count_from_labels(labels_jsonl: str) -> Dict[str, Dict[str, int]]:
    """
    Returns:
      counts[value_item][category] = occurrences
    Counting is based on labels_jsonl rows' "surface_value_regimes".
    """
    counts: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))

    for obj in _iter_jsonl(labels_jsonl):
        svr = obj.get("surface_value_regimes", None)
        if not isinstance(svr, dict):
            continue

        # Use surface_value_regimes keys (already unique per row) to avoid duplicates in value_items list.
        for value_item, info in svr.items():
            if not isinstance(info, dict):
                continue
            cat = info.get("category", None)
            if not isinstance(cat, str) or not cat.strip():
                continue
            counts[value_item][cat] += 1

    # convert nested defaultdict -> dict
    return {vi: dict(cat_map) for vi, cat_map in counts.items()}


def merge_counts_into_regimes(
    regimes_jsonl: str,
    counts: Dict[str, Dict[str, int]],
    out_jsonl: str,
    source_labels_jsonl: str,
) -> None:
    """
    Writes regimes entries with added per-category "count".
    Keeps original fields; just augments each category object.
    """
    _ensure_dir(os.path.dirname(out_jsonl))

    with open(out_jsonl, "w", encoding="utf-8") as w:
        for obj in _iter_jsonl(regimes_jsonl):
            value_item = obj.get("value_item", None)
            categories = obj.get("categories", None)

            # If malformed, write as-is
            if not isinstance(value_item, str) or not isinstance(categories, list):
                w.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            vi_counts = counts.get(value_item, {})
            total_mentions = 0

            new_categories = []
            for cat_obj in categories:
                if not isinstance(cat_obj, dict):
                    new_categories.append(cat_obj)
                    continue

                name = cat_obj.get("name", None)
                c = 0
                if isinstance(name, str):
                    c = int(vi_counts.get(name, 0))
                    total_mentions += c

                cat_obj2 = dict(cat_obj)
                cat_obj2["count"] = c
                new_categories.append(cat_obj2)

            obj2 = dict(obj)
            obj2["categories"] = new_categories

            # optional helpful metadata (won't break downstream)
            meta = obj2.get("meta")
            if not isinstance(meta, dict):
                meta = {} if meta is None else {"_orig_meta": meta}
            meta2 = dict(meta)
            meta2["count_source_labels_jsonl"] = source_labels_jsonl
            meta2["total_mentions_in_labels"] = total_mentions
            obj2["meta"] = meta2

            w.write(json.dumps(obj2, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_jsonl", required=True, help="One *.surface_regimes.normalized.jsonl file")
    ap.add_argument(
        "--regimes_jsonl",
        required=True,
        help="Master regimes definition JSONL (surface_value_regimes_gpt.jsonl)",
    )
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL with counts injected")
    args = ap.parse_args()

    counts = count_from_labels(args.labels_jsonl)
    merge_counts_into_regimes(
        regimes_jsonl=args.regimes_jsonl,
        counts=counts,
        out_jsonl=args.out_jsonl,
        source_labels_jsonl=args.labels_jsonl,
    )


if __name__ == "__main__":
    main()
