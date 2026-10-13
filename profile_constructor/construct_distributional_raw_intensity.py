#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Save continuous score distributions from activation/context data
without OT, using the same loading + averaging procedure.

Main idea:
  - Read multiple activation JSONLs (one per model)
  - Pick a middle proj_scores layer per model
  - Extract per (sentence, value_item) scalar score
  - Normalize scores per (model, value_item) using a configurable scheme
  - Average normalized scores across models
  - Join with context_jsonl by (idx, normalized target_sentence/target_text)
  - Keep only value items whose unique text/sentence count >= threshold
  - Extract score samples
  - Group samples by one of:
      * attribute_only
      * attribute_value
      * attribute_value_factor
      * attribute_value_factor_state
  - For each group:
      * save raw sample array
      * fit Gaussian (mean, std)
      * compute Gaussian-likeness diagnostics

Outputs:
  1) summary CSV
  2) raw NPZ
  3) fitted JSONL
  4) optional plots
  5) optional value-item count CSV
  6) optional normalization stats CSV

Recommended default for polarity-sensitive directional scores:
  --normalization_mode polarity_quantile
  --outlier_quantile_low 0.01
  --outlier_quantile_high 0.99
  --outlier_action winsorize
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
from tqdm import tqdm


# -------------------------
# Helpers: IO / keys
# -------------------------
def read_jsonl(path: str, desc: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=desc, unit="lines"):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def get_sentence_key(rec: Dict[str, Any]) -> Tuple[Any, str]:
    idx = rec.get("idx")
    txt = rec.get("target_sentence") or rec.get("target_text") or ""
    return (idx, normalize_text(txt))


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return None
    except Exception:
        return None


def make_group_id(parts: Tuple[str, ...]) -> str:
    raw = "||".join(parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


# -------------------------
# Factor universe
# -------------------------
def load_factor_universe_from_factors_json(factors_json_path: str) -> Dict[str, List[str]]:
    with open(factors_json_path, "r", encoding="utf-8") as f:
        factors_data = json.load(f)

    items = (factors_data or {}).get("items") or {}
    factor_universe: Dict[str, List[str]] = {}

    for v, obj in items.items():
        facs = []
        for fac in (obj or {}).get("factors") or []:
            name = fac.get("factor")
            if isinstance(name, str) and name.strip():
                facs.append(name.strip())
        if facs:
            factor_universe[v] = sorted(set(facs))

    return factor_universe


def infer_factor_universe_from_context_rows(context_rows: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    fac_sets: Dict[str, Set[str]] = collections.defaultdict(set)
    for cr in context_rows:
        ctx_states = cr.get("context_states") or {}
        if not isinstance(ctx_states, dict):
            continue
        for v, v_ctx in ctx_states.items():
            if not isinstance(v, str) or not v.strip():
                continue
            if not isinstance(v_ctx, dict):
                continue
            for factor in v_ctx.keys():
                if isinstance(factor, str) and factor.strip():
                    fac_sets[v.strip()].add(factor.strip())
    return {v: sorted(list(s)) for v, s in fac_sets.items()}


# -------------------------
# Activation score extraction
# -------------------------
def parse_layer_keys(proj_scores: Dict[str, Any]) -> List[int]:
    keys = []
    for k in (proj_scores or {}).keys():
        try:
            keys.append(int(k))
        except Exception:
            continue
    keys.sort()
    return keys


def pick_middle_layer_key(all_layer_keys: List[int]) -> Optional[int]:
    if not all_layer_keys:
        return None
    mid_idx = (len(all_layer_keys) - 1) // 2
    return all_layer_keys[mid_idx]


def build_model_score_index(
    activation_rows: List[Dict[str, Any]],
    model_name: str,
    prefer_layer: Optional[int] = None,
) -> Tuple[Dict[Tuple[Any, str], Dict[str, float]], Optional[int]]:
    layer_key_counts: collections.Counter = collections.Counter()
    for r in activation_rows:
        vis = r.get("value_items") or []
        if not isinstance(vis, list):
            continue
        for vi in vis:
            if not isinstance(vi, dict):
                continue
            ps = vi.get("proj_scores")
            if not isinstance(ps, dict) or not ps:
                continue
            ks = parse_layer_keys(ps)
            for k in ks:
                layer_key_counts[k] += 1

    all_keys = sorted(layer_key_counts.keys())
    if prefer_layer is not None and prefer_layer in layer_key_counts:
        chosen = prefer_layer
    else:
        chosen = pick_middle_layer_key(all_keys)

    if chosen is None:
        return {}, None

    scores_by_key: Dict[Tuple[Any, str], Dict[str, float]] = {}
    missing_layer = 0
    total_items = 0

    for r in tqdm(activation_rows, desc=f"Extract scores ({model_name})", unit="rows"):
        key = get_sentence_key(r)
        vis = r.get("value_items") or []
        if not isinstance(vis, list):
            continue

        for vi in vis:
            if not isinstance(vi, dict):
                continue
            v = vi.get("value_item")
            if not isinstance(v, str) or not v.strip():
                continue
            v = v.strip()

            ps = vi.get("proj_scores")
            if not isinstance(ps, dict) or not ps:
                continue

            total_items += 1
            sval = ps.get(str(chosen))
            if sval is None:
                missing_layer += 1
                continue

            sval_f = safe_float(sval)
            if sval_f is None:
                continue

            if key not in scores_by_key:
                scores_by_key[key] = {}
            scores_by_key[key][v] = sval_f

    if total_items > 0:
        print(
            f"[info] {model_name}: chosen_layer={chosen} "
            f"(available_keys={len(all_keys)}), missing_layer={missing_layer}/{total_items} "
            f"({(missing_layer/max(1,total_items))*100:.2f}%)"
        )
    else:
        print(f"[warn] {model_name}: no value_items/proj_scores found")

    return scores_by_key, chosen


# -------------------------
# Normalization helpers
# -------------------------
def minmax_to_minus1_1(x: float, xmin: float, xmax: float) -> float:
    if not np.isfinite(x) or not np.isfinite(xmin) or not np.isfinite(xmax):
        return float("nan")
    if xmax <= xmin:
        return 0.0
    return float(2.0 * (x - xmin) / (xmax - xmin) - 1.0)


def mean_std_center_scale(xs: List[float], eps: float = 1e-8) -> Tuple[float, float]:
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0

    mu = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
    if not np.isfinite(std) or std < eps:
        std = 1.0
    return mu, std


def robust_center_scale(xs: List[float], eps: float = 1e-8) -> Tuple[float, float]:
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0

    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = 1.4826 * mad

    if not np.isfinite(scale) or scale < eps:
        std = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
        if not np.isfinite(std) or std < eps:
            scale = 1.0
        else:
            scale = std

    return med, scale


def zscore_clip_tanh(
    x: float,
    center: float,
    scale: float,
    clip_z: float = 4.0,
    tanh_temperature: float = 2.0,
) -> float:
    if not np.isfinite(x) or not np.isfinite(center) or not np.isfinite(scale) or scale <= 0:
        return float("nan")
    z = (x - center) / scale
    z = float(np.clip(z, -clip_z, clip_z))
    return float(np.tanh(z / tanh_temperature))


def compute_quantile_bounds(
    xs: List[float],
    q_low: float,
    q_high: float,
) -> Tuple[float, float]:
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return -1.0, 1.0

    lo = float(np.quantile(arr, q_low))
    hi = float(np.quantile(arr, q_high))

    if not np.isfinite(lo):
        lo = float(np.min(arr))
    if not np.isfinite(hi):
        hi = float(np.max(arr))
    if lo > hi:
        lo, hi = hi, lo

    return lo, hi


def apply_outlier_action(
    x: float,
    lower_bound: float,
    upper_bound: float,
    outlier_action: str,
) -> Optional[float]:
    if not np.isfinite(x):
        return None

    if outlier_action == "winsorize":
        return float(np.clip(x, lower_bound, upper_bound))
    elif outlier_action == "drop":
        if x < lower_bound or x > upper_bound:
            return None
        return float(x)
    else:
        raise ValueError(f"Unknown outlier_action: {outlier_action}")


def compute_shared_symmetric_scale(
    lower_bound: float,
    upper_bound: float,
    eps: float = 1e-8,
) -> float:
    scale = max(abs(lower_bound), abs(upper_bound))
    if not np.isfinite(scale) or scale < eps:
        scale = 1.0
    return float(scale)


def polarity_preserving_quantile_normalize(
    x: float,
    shared_scale: float,
    tanh_temperature: Optional[float] = None,
) -> float:
    """
    Shared-scale normalization:
      - keeps 0 fixed
      - preserves polarity
      - uses same denominator for both negative and positive sides

    If tanh_temperature is None:
      x_norm = clip(x / shared_scale, -1, 1)
    else:
      x_norm = tanh((x / shared_scale) / tanh_temperature)
    """
    if not np.isfinite(x):
        return float("nan")

    shared_scale = max(float(shared_scale), 1e-8)
    z = x / shared_scale

    if tanh_temperature is None:
        return float(np.clip(z, -1.0, 1.0))
    return float(np.tanh(z / tanh_temperature))


def build_normalization_stats(
    model_score_maps: Dict[str, Dict[Tuple[Any, str], Dict[str, float]]],
    normalization_mode: str = "robust_zscore",
    outlier_quantile_low: float = 0.01,
    outlier_quantile_high: float = 0.99,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Returns:
      per_model_stats[model_name][value_item] = {
          "center": ...,
          "scale": ...,
          "raw_min": ...,
          "raw_max": ...,
          "raw_mean": ...,
          "raw_std": ...,
          "raw_median": ...,
          "raw_mad": ...,
          "q_low": ...,
          "q_high": ...,
          "shared_scale": ...,
          "n": ...
      }
    """
    per_model_stats: Dict[str, Dict[str, Dict[str, float]]] = {}

    for mname, score_map in model_score_maps.items():
        by_value: Dict[str, List[float]] = collections.defaultdict(list)
        for _, vmap in score_map.items():
            for v, x in vmap.items():
                if np.isfinite(x):
                    by_value[v].append(float(x))

        stats_for_model: Dict[str, Dict[str, float]] = {}
        for v, xs in by_value.items():
            arr = np.asarray(xs, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue

            raw_min = float(np.min(arr))
            raw_max = float(np.max(arr))
            raw_mean = float(np.mean(arr))
            raw_std = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
            raw_median = float(np.median(arr))
            raw_mad = float(np.median(np.abs(arr - raw_median)))

            q_low, q_high = compute_quantile_bounds(
                xs=xs,
                q_low=outlier_quantile_low,
                q_high=outlier_quantile_high,
            )
            shared_scale = compute_shared_symmetric_scale(q_low, q_high)

            if normalization_mode in {"robust_zscore", "robust_tanh"}:
                center, scale = robust_center_scale(xs)
            elif normalization_mode in {"zscore", "zscore_tanh"}:
                center, scale = mean_std_center_scale(xs)
            elif normalization_mode == "minmax":
                center, scale = raw_min, raw_max
            elif normalization_mode in {"polarity_quantile", "polarity_quantile_tanh"}:
                center, scale = 0.0, shared_scale
            else:
                raise ValueError(f"Unknown normalization_mode: {normalization_mode}")

            stats_for_model[v] = {
                "center": float(center),
                "scale": float(scale),
                "raw_min": raw_min,
                "raw_max": raw_max,
                "raw_mean": raw_mean,
                "raw_std": raw_std,
                "raw_median": raw_median,
                "raw_mad": raw_mad,
                "q_low": float(q_low),
                "q_high": float(q_high),
                "shared_scale": float(shared_scale),
                "n": int(arr.size),
            }

        per_model_stats[mname] = stats_for_model

    return per_model_stats


def build_normalized_targets(
    model_score_maps: Dict[str, Dict[Tuple[Any, str], Dict[str, float]]],
    normalization_mode: str = "robust_zscore",
    clip_z: float = 4.0,
    tanh_temperature: float = 2.0,
    outlier_quantile_low: float = 0.01,
    outlier_quantile_high: float = 0.99,
    outlier_action: str = "winsorize",
    return_stats: bool = False,
):
    """
    Normalize raw scores per (model, value_item), then average across models
    for each (sent_key, value_item).

    normalization_mode:
      - minmax
      - zscore
      - robust_zscore
      - zscore_tanh
      - robust_tanh
      - polarity_quantile
      - polarity_quantile_tanh

    polarity_quantile:
      - keeps zero fixed at zero
      - preserves raw polarity
      - removes/clips outliers using quantile bounds
      - uses ONE shared scale for both sides:
            shared_scale = max(abs(q_low), abs(q_high))
      - therefore one side may not reach ±1 if distribution is asymmetric
    """
    per_model_stats = build_normalization_stats(
        model_score_maps=model_score_maps,
        normalization_mode=normalization_mode,
        outlier_quantile_low=outlier_quantile_low,
        outlier_quantile_high=outlier_quantile_high,
    )

    agg: Dict[Tuple[Any, str], Dict[str, List[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )

    for mname, score_map in model_score_maps.items():
        stats_map = per_model_stats.get(mname, {})
        for sk, vmap in score_map.items():
            for v, x in vmap.items():
                if v not in stats_map or not np.isfinite(x):
                    continue

                stat = stats_map[v]

                if normalization_mode == "minmax":
                    xmin = stat["raw_min"]
                    xmax = stat["raw_max"]
                    xn = minmax_to_minus1_1(x, xmin, xmax)

                elif normalization_mode == "zscore":
                    mu = stat["center"]
                    std = stat["scale"]
                    xn = float(np.clip((x - mu) / std, -clip_z, clip_z))

                elif normalization_mode == "robust_zscore":
                    med = stat["center"]
                    scale = stat["scale"]
                    xn = float(np.clip((x - med) / scale, -clip_z, clip_z))

                elif normalization_mode == "zscore_tanh":
                    mu = stat["center"]
                    std = stat["scale"]
                    xn = zscore_clip_tanh(
                        x=x,
                        center=mu,
                        scale=std,
                        clip_z=clip_z,
                        tanh_temperature=tanh_temperature,
                    )

                elif normalization_mode == "robust_tanh":
                    med = stat["center"]
                    scale = stat["scale"]
                    xn = zscore_clip_tanh(
                        x=x,
                        center=med,
                        scale=scale,
                        clip_z=clip_z,
                        tanh_temperature=tanh_temperature,
                    )

                elif normalization_mode in {"polarity_quantile", "polarity_quantile_tanh"}:
                    q_low = stat["q_low"]
                    q_high = stat["q_high"]

                    x_adj = apply_outlier_action(
                        x=x,
                        lower_bound=q_low,
                        upper_bound=q_high,
                        outlier_action=outlier_action,
                    )
                    if x_adj is None:
                        continue

                    shared_scale = stat["shared_scale"]

                    if normalization_mode == "polarity_quantile":
                        xn = polarity_preserving_quantile_normalize(
                            x=x_adj,
                            shared_scale=shared_scale,
                            tanh_temperature=None,
                        )
                    else:
                        xn = polarity_preserving_quantile_normalize(
                            x=x_adj,
                            shared_scale=shared_scale,
                            tanh_temperature=tanh_temperature,
                        )

                else:
                    raise ValueError(f"Unknown normalization_mode: {normalization_mode}")

                if np.isfinite(xn):
                    agg[sk][v].append(float(xn))

    target_by_key: Dict[Tuple[Any, str], Dict[str, float]] = {}
    for sk, vlists in agg.items():
        outv = {}
        for v, xs in vlists.items():
            if xs:
                outv[v] = float(np.mean(xs))
        if outv:
            target_by_key[sk] = outv

    print(f"[info] built target scores for sentences: {len(target_by_key)}")
    print(f"[info] normalization_mode={normalization_mode}")
    print(f"[info] outlier_quantile_low={outlier_quantile_low}")
    print(f"[info] outlier_quantile_high={outlier_quantile_high}")
    print(f"[info] outlier_action={outlier_action}")

    if return_stats:
        return target_by_key, per_model_stats
    return target_by_key


def write_normalization_stats_csv(
    out_csv: str,
    attribute_name: str,
    per_model_stats: Dict[str, Dict[str, Dict[str, float]]],
    kept_value_items: Optional[Set[str]] = None,
) -> None:
    rows = []
    for model_name, value_stats in per_model_stats.items():
        for value_item, stat in value_stats.items():
            if kept_value_items is not None and value_item not in kept_value_items:
                continue
            rows.append({
                "attribute": attribute_name,
                "model_name": model_name,
                "value_item": value_item,
                **stat,
            })

    if not rows:
        print(f"[warn] no normalization stats rows to write: {out_csv}")
        return

    df = pd.DataFrame(rows).sort_values(
        ["model_name", "value_item"],
        ascending=[True, True],
    )
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[done] wrote normalization stats CSV: {out_csv}")


# -------------------------
# Sentence-level value-item filtering
# -------------------------
def compute_value_item_sentence_counts(
    target_by_key: Dict[Tuple[Any, str], Dict[str, float]]
) -> Dict[str, int]:
    sent_sets: Dict[str, Set[Tuple[Any, str]]] = collections.defaultdict(set)
    for sent_key, value_map in target_by_key.items():
        for value_item in value_map.keys():
            if isinstance(value_item, str) and value_item.strip():
                sent_sets[value_item.strip()].add(sent_key)
    return {v: len(ss) for v, ss in sent_sets.items()}


def filter_target_by_value_item_count(
    target_by_key: Dict[Tuple[Any, str], Dict[str, float]],
    min_value_item_count: int,
) -> Tuple[Dict[Tuple[Any, str], Dict[str, float]], Dict[str, int], Set[str]]:
    if min_value_item_count <= 0:
        counts = compute_value_item_sentence_counts(target_by_key)
        keep = set(counts.keys())
        return target_by_key, counts, keep

    counts = compute_value_item_sentence_counts(target_by_key)
    keep = {v for v, c in counts.items() if c >= min_value_item_count}

    filtered_target_by_key: Dict[Tuple[Any, str], Dict[str, float]] = {}
    for sent_key, value_map in target_by_key.items():
        kept_map = {
            v: score for v, score in value_map.items()
            if v in keep
        }
        if kept_map:
            filtered_target_by_key[sent_key] = kept_map

    print(f"[info] value_items before sentence-count filter: {len(counts)}")
    print(f"[info] value_items kept (count >= {min_value_item_count}): {len(keep)}")
    print(f"[info] value_items dropped: {len(counts) - len(keep)}")
    print(f"[info] sentences with kept value_items: {len(filtered_target_by_key)}")

    return filtered_target_by_key, counts, keep


def write_value_item_counts_csv(
    out_csv: str,
    attribute_name: str,
    counts: Dict[str, int],
    kept_value_items: Set[str],
) -> None:
    rows = []
    for value_item, count in counts.items():
        rows.append({
            "attribute": attribute_name,
            "value_item": value_item,
            "text_count": int(count),
            "kept": int(value_item in kept_value_items),
        })
    df = pd.DataFrame(rows).sort_values(["kept", "text_count", "value_item"], ascending=[False, False, True])
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[done] wrote value-item count CSV: {out_csv}")


# -------------------------
# Data model
# -------------------------
@dataclass(frozen=True)
class Sample:
    attribute: str
    value_item: str
    factor: str
    state: str
    score: float
    sent_key: Tuple[Any, str]


# -------------------------
# Extraction (context + target scores)
# -------------------------
def extract_samples_from_targets(
    attribute_name: str,
    context_rows: List[Dict[str, Any]],
    target_by_key: Dict[Tuple[Any, str], Dict[str, float]],
    factor_universe: Dict[str, List[str]],
) -> List[Sample]:
    samples: List[Sample] = []
    n_total_context = 0
    n_missing_join = 0
    n_missing_factor_universe = 0

    for cr in tqdm(context_rows, desc="Extract samples", unit="rows"):
        n_total_context += 1
        key = get_sentence_key(cr)

        score_map = target_by_key.get(key)
        if score_map is None:
            n_missing_join += 1
            continue

        ctx_states = cr.get("context_states") or {}
        if not isinstance(ctx_states, dict):
            ctx_states = {}

        for v, score in score_map.items():
            if not isinstance(v, str) or not v.strip():
                continue
            v = v.strip()
            if not np.isfinite(score):
                continue

            facs = factor_universe.get(v)
            if not facs:
                n_missing_factor_universe += 1
                continue

            v_ctx = ctx_states.get(v)
            if not isinstance(v_ctx, dict):
                v_ctx = {}

            for factor in facs:
                obj = v_ctx.get(factor)
                st = None
                if isinstance(obj, dict):
                    st = obj.get("selected_state")
                if not isinstance(st, str) or not st.strip():
                    st = "NONE"
                st = st.strip()

                samples.append(Sample(
                    attribute=attribute_name,
                    value_item=v,
                    factor=factor,
                    state=st,
                    score=float(score),
                    sent_key=key,
                ))

    print(f"[info] context rows: {n_total_context}")
    print(f"[info] missing target join: {n_missing_join} ({(n_missing_join/max(1,n_total_context))*100:.2f}%)")
    print(f"[info] missing factor universe (value_item has no factors): {n_missing_factor_universe}")
    print(f"[info] extracted samples: {len(samples)}")
    return samples


# -------------------------
# Grouping
# -------------------------
def get_group_key(sample: Sample, group_mode: str) -> Tuple[str, ...]:
    if group_mode == "attribute_only":
        return (sample.attribute,)
    elif group_mode == "attribute_value":
        return (sample.attribute, sample.value_item)
    elif group_mode == "attribute_value_factor":
        return (sample.attribute, sample.value_item, sample.factor)
    elif group_mode == "attribute_value_factor_state":
        return (sample.attribute, sample.value_item, sample.factor, sample.state)
    else:
        raise ValueError(f"Unknown group_mode: {group_mode}")


def parse_group_key(group_key: Tuple[str, ...], group_mode: str) -> Dict[str, str]:
    if group_mode == "attribute_only":
        return {
            "attribute": group_key[0],
            "value_item": "",
            "factor": "",
            "state": "",
        }
    elif group_mode == "attribute_value":
        return {
            "attribute": group_key[0],
            "value_item": group_key[1],
            "factor": "",
            "state": "",
        }
    elif group_mode == "attribute_value_factor":
        return {
            "attribute": group_key[0],
            "value_item": group_key[1],
            "factor": group_key[2],
            "state": "",
        }
    elif group_mode == "attribute_value_factor_state":
        return {
            "attribute": group_key[0],
            "value_item": group_key[1],
            "factor": group_key[2],
            "state": group_key[3],
        }
    else:
        raise ValueError(f"Unknown group_mode: {group_mode}")


# -------------------------
# Gaussian diagnostics
# -------------------------
def normal_cdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return (x >= mu).astype(np.float64)
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + np.vectorize(math.erf)(z))


def empirical_cdf(sorted_x: np.ndarray) -> np.ndarray:
    n = sorted_x.size
    return np.arange(1, n + 1, dtype=np.float64) / n


def ks_distance_to_fitted_normal(x: np.ndarray, mu: float, sigma: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return float("nan")
    xs = np.sort(x)
    ecdf = empirical_cdf(xs)
    ncdf = normal_cdf(xs, mu, sigma)
    return float(np.max(np.abs(ecdf - ncdf)))


def compute_sample_stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size

    if n == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "std": math.nan,
            "min": math.nan,
            "max": math.nan,
            "q01": math.nan,
            "q05": math.nan,
            "q10": math.nan,
            "q25": math.nan,
            "q50": math.nan,
            "q75": math.nan,
            "q90": math.nan,
            "q95": math.nan,
            "q99": math.nan,
            "skewness": math.nan,
            "excess_kurtosis": math.nan,
            "jb_stat": math.nan,
            "ks_to_fitted_normal": math.nan,
            "gaussian_like_flag": 0,
        }

    mu = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if n >= 2 else 0.0

    centered = x - mu
    m2 = float(np.mean(centered ** 2)) if n >= 1 else math.nan
    m3 = float(np.mean(centered ** 3)) if n >= 1 else math.nan
    m4 = float(np.mean(centered ** 4)) if n >= 1 else math.nan

    if m2 > 0:
        skew = float(m3 / (m2 ** 1.5))
        ex_kurt = float(m4 / (m2 ** 2) - 3.0)
    else:
        skew = 0.0
        ex_kurt = 0.0

    if n >= 2:
        jb = float((n / 6.0) * (skew ** 2 + 0.25 * (ex_kurt ** 2)))
    else:
        jb = math.nan

    ks = ks_distance_to_fitted_normal(x, mu, std if std > 0 else 1e-8)

    gaussian_like = int(
        (n >= 20) and
        (abs(skew) <= 0.5) and
        (abs(ex_kurt) <= 1.0) and
        (ks <= 0.12)
    )

    return {
        "n": int(n),
        "mean": mu,
        "std": std,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "q01": float(np.quantile(x, 0.01)),
        "q05": float(np.quantile(x, 0.05)),
        "q10": float(np.quantile(x, 0.10)),
        "q25": float(np.quantile(x, 0.25)),
        "q50": float(np.quantile(x, 0.50)),
        "q75": float(np.quantile(x, 0.75)),
        "q90": float(np.quantile(x, 0.90)),
        "q95": float(np.quantile(x, 0.95)),
        "q99": float(np.quantile(x, 0.99)),
        "skewness": skew,
        "excess_kurtosis": ex_kurt,
        "jb_stat": jb,
        "ks_to_fitted_normal": ks,
        "gaussian_like_flag": gaussian_like,
    }


# -------------------------
# Plotting (optional)
# -------------------------
def maybe_make_plot(
    xs: np.ndarray,
    title: str,
    out_png: str,
    bins: int = 50,
) -> None:
    import matplotlib.pyplot as plt

    xs = np.asarray(xs, dtype=np.float64)
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return

    mu = float(np.mean(xs))
    std = float(np.std(xs, ddof=1)) if xs.size >= 2 else 0.0

    plt.figure(figsize=(6, 4))
    plt.hist(xs, bins=bins, density=True)

    if std > 1e-8:
        left = float(np.min(xs))
        right = float(np.max(xs))
        if left == right:
            left -= 1.0
            right += 1.0
        grid = np.linspace(left, right, 400)
        pdf = (1.0 / (std * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((grid - mu) / std) ** 2)
        plt.plot(grid, pdf)

    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


# -------------------------
# Save distributions
# -------------------------
def save_raw_npz(path: str, grouped_scores: Dict[Tuple[str, ...], List[float]], group_mode: str) -> None:
    arrays = {}
    manifest = []
    for group_key, xs in grouped_scores.items():
        meta = parse_group_key(group_key, group_mode)
        gid = make_group_id(group_key)
        arr_name = f"group_{gid}"
        arrays[arr_name] = np.asarray(xs, dtype=np.float32)
        manifest.append({
            "group_id": gid,
            "array_name": arr_name,
            **meta,
            "n": int(len(xs)),
        })

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **arrays)

    manifest_path = path + ".manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[info] wrote raw NPZ: {path}")
    print(f"[info] wrote raw NPZ manifest: {manifest_path}")


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--attribute_name", required=True)
    ap.add_argument("--context_jsonl", required=True)

    ap.add_argument(
        "--activation_jsonls",
        required=True,
        help="Comma-separated list of activation JSONL paths (one per model).",
    )
    ap.add_argument(
        "--model_names",
        default=None,
        help="Optional comma-separated names aligned with --activation_jsonls. If omitted, uses basename.",
    )
    ap.add_argument(
        "--prefer_layer",
        type=int,
        default=None,
        help="Optional: force a specific layer key if present; otherwise middle layer is used per model.",
    )

    ap.add_argument("--factors_json", default=None)

    ap.add_argument(
        "--group_mode",
        default="attribute_value",
        choices=[
            "attribute_only",
            "attribute_value",
            "attribute_value_factor",
            "attribute_value_factor_state",
        ],
    )
    ap.add_argument("--min_value_item_count", type=int, default=200)
    ap.add_argument("--min_group_n", type=int, default=20)

    ap.add_argument(
        "--normalization_mode",
        default="polarity_quantile",
        choices=[
            "minmax",
            "zscore",
            "robust_zscore",
            "zscore_tanh",
            "robust_tanh",
            "polarity_quantile",
            "polarity_quantile_tanh",
        ],
    )
    ap.add_argument("--clip_z", type=float, default=4.0)
    ap.add_argument("--tanh_temperature", type=float, default=2.0)

    ap.add_argument("--outlier_quantile_low", type=float, default=0.01)
    ap.add_argument("--outlier_quantile_high", type=float, default=0.99)
    ap.add_argument(
        "--outlier_action",
        default="winsorize",
        choices=["winsorize", "drop"],
        help="For polarity_quantile modes: winsorize clips tails, drop removes tail values.",
    )

    ap.add_argument("--out_summary_csv", required=True)
    ap.add_argument("--out_raw_npz", required=True)
    ap.add_argument("--out_fit_jsonl", required=True)
    ap.add_argument("--out_value_item_count_csv", default=None)
    ap.add_argument("--out_normalization_stats_csv", default=None)
    ap.add_argument("--plot_dir", default=None)

    args = ap.parse_args()

    if not (0.0 <= args.outlier_quantile_low < args.outlier_quantile_high <= 1.0):
        raise ValueError("Require 0 <= outlier_quantile_low < outlier_quantile_high <= 1")

    act_paths = [p.strip() for p in args.activation_jsonls.split(",") if p.strip()]
    if not act_paths:
        raise ValueError("Empty --activation_jsonls")

    if args.model_names:
        model_names = [x.strip() for x in args.model_names.split(",")]
        if len(model_names) != len(act_paths):
            raise ValueError("--model_names must have same length as --activation_jsonls")
    else:
        model_names = [os.path.basename(p).replace(".jsonl", "") for p in act_paths]

    if any(not name for name in model_names) or len(set(model_names)) != len(model_names):
        raise ValueError("Model names must be nonempty and unique; supply --model_names for files with identical basenames.")

    print("[stage] load context jsonl")
    context_rows = read_jsonl(args.context_jsonl, desc="Load context JSONL")

    print("[stage] load activation jsonls")
    model_rows: Dict[str, List[Dict[str, Any]]] = {}
    for mname, path in zip(model_names, act_paths):
        model_rows[mname] = read_jsonl(path, desc=f"Load activations ({mname})")

    print("[stage] extract per-model middle-layer raw scores")
    model_score_maps: Dict[str, Dict[Tuple[Any, str], Dict[str, float]]] = {}
    chosen_layers: Dict[str, Optional[int]] = {}

    for mname in model_names:
        s_map, chosen = build_model_score_index(
            activation_rows=model_rows[mname],
            model_name=mname,
            prefer_layer=args.prefer_layer,
        )
        model_score_maps[mname] = s_map
        chosen_layers[mname] = chosen

    if all(chosen_layers[m] is None for m in model_names):
        raise RuntimeError("No model had usable proj_scores layer keys.")

    print("[stage] build normalized target scores (avg across models)")
    target_result = build_normalized_targets(
        model_score_maps=model_score_maps,
        normalization_mode=args.normalization_mode,
        clip_z=args.clip_z,
        tanh_temperature=args.tanh_temperature,
        outlier_quantile_low=args.outlier_quantile_low,
        outlier_quantile_high=args.outlier_quantile_high,
        outlier_action=args.outlier_action,
        return_stats=True,
    )
    target_by_key, per_model_stats = target_result

    if not target_by_key:
        raise RuntimeError("No target scores built. Check joins / proj_scores availability.")

    print("[stage] filter target scores by sentence-level value-item count")
    target_by_key, value_item_counts, kept_value_items = filter_target_by_value_item_count(
        target_by_key=target_by_key,
        min_value_item_count=args.min_value_item_count,
    )
    if not target_by_key:
        raise RuntimeError("No target scores left after value-item count filtering.")

    if args.out_value_item_count_csv:
        write_value_item_counts_csv(
            out_csv=args.out_value_item_count_csv,
            attribute_name=args.attribute_name,
            counts=value_item_counts,
            kept_value_items=kept_value_items,
        )

    if args.out_normalization_stats_csv:
        write_normalization_stats_csv(
            out_csv=args.out_normalization_stats_csv,
            attribute_name=args.attribute_name,
            per_model_stats=per_model_stats,
            kept_value_items=kept_value_items,
        )

    print("[stage] factor universe")
    if args.factors_json:
        factor_universe = load_factor_universe_from_factors_json(args.factors_json)
        print(f"[info] factor universe loaded from factors_json: {len(factor_universe)} value_items")
    else:
        factor_universe = infer_factor_universe_from_context_rows(context_rows)
        print(f"[info] factor universe inferred from context_rows: {len(factor_universe)} value_items")

    print("[stage] restrict factor universe to kept value items")
    factor_universe = {v: facs for v, facs in factor_universe.items() if v in kept_value_items}
    print(f"[info] factor universe after keep filter: {len(factor_universe)} value_items")

    print("[stage] extract samples from (context_states + filtered target scores)")
    samples = extract_samples_from_targets(
        attribute_name=args.attribute_name,
        context_rows=context_rows,
        target_by_key=target_by_key,
        factor_universe=factor_universe,
    )
    if not samples:
        raise RuntimeError("No samples extracted.")

    print(f"[stage] group samples: mode={args.group_mode}")
    grouped_scores: Dict[Tuple[str, ...], List[float]] = collections.defaultdict(list)
    grouped_sentkeys: Dict[Tuple[str, ...], Set[Tuple[Any, str]]] = collections.defaultdict(set)

    for s in tqdm(samples, desc="Accumulate grouped scores", unit="samples"):
        gk = get_group_key(s, args.group_mode)
        grouped_scores[gk].append(float(s.score))
        grouped_sentkeys[gk].add(s.sent_key)

    print(f"[info] groups before min_group_n filter: {len(grouped_scores)}")

    grouped_scores = {
        gk: xs for gk, xs in grouped_scores.items()
        if len(xs) >= args.min_group_n
    }
    grouped_sentkeys = {
        gk: grouped_sentkeys[gk] for gk in grouped_scores.keys()
    }

    print(f"[info] groups kept (min_group_n={args.min_group_n}): {len(grouped_scores)}")
    if not grouped_scores:
        raise RuntimeError("No groups left after --min_group_n filtering.")

    print("[stage] save raw arrays")
    save_raw_npz(args.out_raw_npz, grouped_scores, args.group_mode)

    print("[stage] compute Gaussian fit + diagnostics")
    summary_rows = []
    fit_rows = []

    for group_key, xs in tqdm(grouped_scores.items(), desc="Summarize groups", unit="groups"):
        meta = parse_group_key(group_key, args.group_mode)
        gid = make_group_id(group_key)

        arr = np.asarray(xs, dtype=np.float64)
        arr = arr[np.isfinite(arr)]

        stats = compute_sample_stats(arr)
        row = {
            "group_id": gid,
            **meta,
            "group_mode": args.group_mode,
            "normalization_mode": args.normalization_mode,
            "clip_z": args.clip_z,
            "tanh_temperature": args.tanh_temperature,
            "outlier_quantile_low": args.outlier_quantile_low,
            "outlier_quantile_high": args.outlier_quantile_high,
            "outlier_action": args.outlier_action,
            "n_samples": int(arr.size),
            "n_unique_sentences": int(len(grouped_sentkeys[group_key])),
            **stats,
        }
        summary_rows.append(row)

        fit_rows.append({
            "group_id": gid,
            **meta,
            "group_mode": args.group_mode,
            "normalization_mode": args.normalization_mode,
            "clip_z": args.clip_z,
            "tanh_temperature": args.tanh_temperature,
            "outlier_quantile_low": args.outlier_quantile_low,
            "outlier_quantile_high": args.outlier_quantile_high,
            "outlier_action": args.outlier_action,
            "distribution_type": "gaussian",
            "mean": stats["mean"],
            "std": stats["std"],
            "support_min": float(np.min(arr)) if arr.size > 0 else math.nan,
            "support_max": float(np.max(arr)) if arr.size > 0 else math.nan,
            "n_samples": int(arr.size),
            "n_unique_sentences": int(len(grouped_sentkeys[group_key])),
            "gaussian_like_flag": int(stats["gaussian_like_flag"]),
            "ks_to_fitted_normal": stats["ks_to_fitted_normal"],
            "skewness": stats["skewness"],
            "excess_kurtosis": stats["excess_kurtosis"],
            "jb_stat": stats["jb_stat"],
            "q05": stats["q05"],
            "q50": stats["q50"],
            "q95": stats["q95"],
        })

        if args.plot_dir:
            title_parts = [meta["attribute"]]
            if meta["value_item"]:
                title_parts.append(meta["value_item"])
            if meta["factor"]:
                title_parts.append(meta["factor"])
            if meta["state"]:
                title_parts.append(meta["state"])
            title_parts.append(f"norm={args.normalization_mode}")
            title = " | ".join(title_parts)

            out_png = os.path.join(args.plot_dir, f"{gid}.png")
            maybe_make_plot(arr, title=title, out_png=out_png)

    df = pd.DataFrame(summary_rows)
    df = df.sort_values(
        ["attribute", "value_item", "factor", "state", "n_samples"],
        ascending=[True, True, True, True, False],
    )

    os.makedirs(os.path.dirname(args.out_summary_csv) or ".", exist_ok=True)
    df.to_csv(args.out_summary_csv, index=False)
    print(f"[done] wrote summary CSV: {args.out_summary_csv} (rows={len(df)})")

    write_jsonl(args.out_fit_jsonl, fit_rows)
    print(f"[done] wrote fit JSONL: {args.out_fit_jsonl} (rows={len(fit_rows)})")

    print("[info] chosen middle layers per model:")
    for m in model_names:
        print(f"  - {m}: layer={chosen_layers[m]}")


if __name__ == "__main__":
    main()
