#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OT-based context effect modeling using CONTINUOUS projection scores (NO BINNING),
with the same normalization and filtering logic as continuous base aggregation.

Supports two modes:

[A] Direct OT from saved raw NPZ
    - Requires NPZ created with:
        --group_mode attribute_value_factor_state
    - Uses NPZ + NPZ.manifest.json
    - Fastest path if arrays were already saved correctly

[B] Rebuild from context_jsonl + activation_jsonls
    - Reuses the same:
        * sentence join
        * per-(model,value_item) normalization
        * polarity-preserving quantile normalization
        * value-item count filtering
        * factor-universe handling
    - Then computes OT from grouped continuous samples

Outputs per (attribute, value_item, factor, state_a, state_b):
  - empirical W1 OT
  - bootstrap mean / CI for W1
  - empirical mean shift
  - bootstrap mean / CI for mean shift
  - per-state sample counts
  - optional debug state score summary CSV
  - optional normalization stats CSV
  - optional value-item count CSV

Notes:
  - No discretization
  - No Dirichlet posterior
  - Bootstrap used for uncertainty
"""

from __future__ import annotations

import argparse
import collections
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
    attribute_name: Optional[str],
    per_model_stats: Dict[str, Dict[str, Dict[str, float]]],
    kept_value_items: Optional[Set[str]] = None,
) -> None:
    rows = []
    for model_name, value_stats in per_model_stats.items():
        for value_item, stat in value_stats.items():
            if kept_value_items is not None and value_item not in kept_value_items:
                continue
            row = {
                "model_name": model_name,
                "value_item": value_item,
                **stat,
            }
            if attribute_name is not None:
                row["attribute"] = attribute_name
            rows.append(row)

    if not rows:
        print(f"[warn] no normalization stats rows to write: {out_csv}")
        return

    df = pd.DataFrame(rows)
    sort_cols = ["model_name", "value_item"]
    if "attribute" in df.columns:
        sort_cols = ["attribute"] + sort_cols

    df = df.sort_values(sort_cols, ascending=[True] * len(sort_cols))
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
        kept_map = {v: score for v, score in value_map.items() if v in keep}
        if kept_map:
            filtered_target_by_key[sent_key] = kept_map

    print(f"[info] value_items before sentence-count filter: {len(counts)}")
    print(f"[info] value_items kept (count >= {min_value_item_count}): {len(keep)}")
    print(f"[info] value_items dropped: {len(counts) - len(keep)}")
    print(f"[info] sentences with kept value_items: {len(filtered_target_by_key)}")

    return filtered_target_by_key, counts, keep


def write_value_item_counts_csv(
    out_csv: str,
    attribute_name: Optional[str],
    counts: Dict[str, int],
    kept_value_items: Set[str],
) -> None:
    rows = []
    for value_item, count in counts.items():
        row = {
            "value_item": value_item,
            "text_count": int(count),
            "kept": int(value_item in kept_value_items),
        }
        if attribute_name is not None:
            row["attribute"] = attribute_name
        rows.append(row)

    df = pd.DataFrame(rows)
    sort_cols = ["kept", "text_count", "value_item"]
    ascending = [False, False, True]
    if "attribute" in df.columns:
        sort_cols = ["attribute"] + sort_cols
        ascending = [True] + ascending

    df = df.sort_values(sort_cols, ascending=ascending)
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
    sent_key: Optional[Tuple[Any, str]]


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
# Load state-level arrays from saved raw NPZ
# -------------------------
def load_samples_from_raw_npz(
    raw_npz_path: str,
    attribute_name: Optional[str] = None,
) -> List[Sample]:
    manifest_path = raw_npz_path + ".manifest.json"
    if not os.path.exists(raw_npz_path):
        raise FileNotFoundError(f"raw NPZ not found: {raw_npz_path}")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    data = np.load(raw_npz_path, allow_pickle=False)

    samples: List[Sample] = []
    n_groups = 0
    n_skipped = 0

    for item in manifest:
        arr_name = item.get("array_name")
        value_item = item.get("value_item", "")
        factor = item.get("factor", "")
        state = item.get("state", "")
        attr = item.get("attribute", attribute_name or "")

        # Must be state-level grouped arrays
        if not value_item or not factor or state == "":
            n_skipped += 1
            continue

        if attribute_name is not None and attr != attribute_name:
            continue

        if arr_name not in data:
            n_skipped += 1
            continue

        arr = np.asarray(data[arr_name], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue

        n_groups += 1
        for x in arr.tolist():
            samples.append(Sample(
                attribute=attr,
                value_item=value_item,
                factor=factor,
                state=state,
                score=float(x),
                sent_key=None,
            ))

    print(f"[info] loaded NPZ state groups: {n_groups}")
    print(f"[info] skipped manifest items: {n_skipped}")
    print(f"[info] loaded NPZ samples: {len(samples)}")

    if not samples:
        raise RuntimeError(
            "No usable state-level samples found in NPZ. "
            "This usually means the NPZ was not saved with --group_mode attribute_value_factor_state."
        )

    return samples


# -------------------------
# Continuous 1D OT + Bootstrap
# -------------------------
def w1_continuous_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n = a.size
    m = b.size
    if n == 0 or m == 0:
        return float("nan")

    a_sorted = np.sort(a)
    b_sorted = np.sort(b)

    if n == m:
        return float(np.mean(np.abs(a_sorted - b_sorted)))

    K = int(max(n, m))
    qs = (np.arange(1, K + 1, dtype=np.float64) - 0.5) / K

    def quantiles(sorted_x: np.ndarray, qs_: np.ndarray) -> np.ndarray:
        N = sorted_x.size
        pos = qs_ * (N - 1)
        lo = np.floor(pos).astype(np.int64)
        hi = np.clip(lo + 1, 0, N - 1)
        w = pos - lo
        return (1.0 - w) * sorted_x[lo] + w * sorted_x[hi]

    qa = quantiles(a_sorted, qs)
    qb = quantiles(b_sorted, qs)
    return float(np.mean(np.abs(qa - qb)))


def bootstrap_continuous_ot(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    draws: int = 2000,
) -> Dict[str, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n = a.size
    m = b.size
    if n == 0 or m == 0:
        return {
            "ot_mean": math.nan, "ot_q025": math.nan, "ot_q975": math.nan,
            "mean_shift": math.nan, "mean_shift_q025": math.nan, "mean_shift_q975": math.nan,
        }

    ots = np.empty((draws,), dtype=np.float64)
    shifts = np.empty((draws,), dtype=np.float64)

    for i in range(draws):
        sa = a[rng.integers(0, n, size=n)]
        sb = b[rng.integers(0, m, size=m)]
        ots[i] = w1_continuous_1d(sa, sb)
        shifts[i] = float(np.mean(sa) - np.mean(sb))

    return {
        "ot_mean": float(np.mean(ots)),
        "ot_q025": float(np.quantile(ots, 0.025)),
        "ot_q975": float(np.quantile(ots, 0.975)),
        "mean_shift": float(np.mean(shifts)),
        "mean_shift_q025": float(np.quantile(shifts, 0.025)),
        "mean_shift_q975": float(np.quantile(shifts, 0.975)),
    }


# -------------------------
# Optional debug summaries
# -------------------------
def write_state_score_summary_csv(
    out_csv: str,
    scores: Dict[Tuple[str, str, str, str], List[float]],
) -> None:
    rows = []
    for (attr, v, f, st), xs in scores.items():
        arr = np.asarray(xs, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        rows.append({
            "attribute": attr,
            "value_item": v,
            "factor": f,
            "state": st,
            "N": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size >= 2 else 0.0,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "q05": float(np.quantile(arr, 0.05)),
            "q50": float(np.quantile(arr, 0.50)),
            "q95": float(np.quantile(arr, 0.95)),
        })

    if not rows:
        print(f"[warn] no state-score rows to write: {out_csv}")
        return

    df = pd.DataFrame(rows).sort_values(["attribute", "value_item", "factor", "state"])
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[done] wrote debug state-score summary CSV: {out_csv}")


# -------------------------
# Build samples from activations + context
# -------------------------
def build_samples_from_source(args) -> Tuple[List[Sample], Dict[str, Optional[int]], Optional[Dict[str, Dict[str, Dict[str, float]]]], Optional[Dict[str, int]], Optional[Set[str]]]:
    if not args.context_jsonl:
        raise ValueError("--context_jsonl is required when not using --raw_npz")
    if not args.activation_jsonls:
        raise ValueError("--activation_jsonls is required when not using --raw_npz")
    if not args.attribute_name:
        raise ValueError("--attribute_name is required when not using --raw_npz")

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

    return samples, chosen_layers, per_model_stats, value_item_counts, kept_value_items


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    # Source selection
    ap.add_argument("--raw_npz", default=None, help="Optional: saved raw NPZ from the score-saving script.")
    ap.add_argument("--attribute_name", default=None, help="Required when rebuilding from source. Optional filter when using --raw_npz.")

    # Rebuild-from-source inputs
    ap.add_argument("--context_jsonl", default=None)
    ap.add_argument(
        "--activation_jsonls",
        default=None,
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

    # Filtering / normalization (used only when rebuilding from source)
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
    )
    ap.add_argument("--min_value_item_count", type=int, default=200)

    # OT options
    ap.add_argument("--draws", type=int, default=2000, help="Bootstrap draws for CI.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_n_per_state", type=int, default=10, help="Flag low-support states.")
    ap.add_argument(
        "--drop_low_state_pairs",
        action="store_true",
        help="If set, pairs with either state having N < min_n_per_state are skipped entirely.",
    )

    # Outputs
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_scores_csv", default=None, help="Optional debug summary per (attribute,value_item,factor,state).")
    ap.add_argument("--out_normalization_stats_csv", default=None, help="Optional, only when rebuilding from source.")
    ap.add_argument("--out_value_item_count_csv", default=None, help="Optional, only when rebuilding from source.")

    args = ap.parse_args()

    if not (0.0 <= args.outlier_quantile_low < args.outlier_quantile_high <= 1.0):
        raise ValueError("Require 0 <= outlier_quantile_low < outlier_quantile_high <= 1")

    # -------------------------
    # Build or load samples
    # -------------------------
    if args.raw_npz:
        print("[stage] load samples from saved raw NPZ")
        samples = load_samples_from_raw_npz(
            raw_npz_path=args.raw_npz,
            attribute_name=args.attribute_name,
        )
        chosen_layers = {}
        per_model_stats = None
        value_item_counts = None
        kept_value_items = None
    else:
        samples, chosen_layers, per_model_stats, value_item_counts, kept_value_items = build_samples_from_source(args)

        if args.out_normalization_stats_csv and per_model_stats is not None:
            write_normalization_stats_csv(
                out_csv=args.out_normalization_stats_csv,
                attribute_name=args.attribute_name,
                per_model_stats=per_model_stats,
                kept_value_items=kept_value_items,
            )

        if args.out_value_item_count_csv and value_item_counts is not None and kept_value_items is not None:
            write_value_item_counts_csv(
                out_csv=args.out_value_item_count_csv,
                attribute_name=args.attribute_name,
                counts=value_item_counts,
                kept_value_items=kept_value_items,
            )

    # -------------------------
    # Group per (attribute, value_item, factor, state)
    # -------------------------
    print("[stage] collect continuous score lists per (attribute,value_item,factor,state)")
    scores: Dict[Tuple[str, str, str, str], List[float]] = collections.defaultdict(list)
    for s in tqdm(samples, desc="Accumulate scores", unit="samples"):
        scores[(s.attribute, s.value_item, s.factor, s.state)].append(float(s.score))

    if args.out_scores_csv:
        write_state_score_summary_csv(args.out_scores_csv, scores)

    # -------------------------
    # Compute OT
    # -------------------------
    print("[stage] compute continuous OT + bootstrap CI")
    rng = np.random.default_rng(args.seed)
    rows_out = []

    by_avf: Dict[Tuple[str, str, str], List[str]] = collections.defaultdict(list)
    for (attr, v, f, st) in scores.keys():
        by_avf[(attr, v, f)].append(st)

    for (attr, v, f), states in tqdm(list(by_avf.items()), desc="Per (attribute,value_item,factor)", unit="avf"):
        states = sorted(set(states))
        if len(states) < 2:
            continue

        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                sA = states[i]
                sB = states[j]

                a = np.asarray(scores[(attr, v, f, sA)], dtype=np.float64)
                b = np.asarray(scores[(attr, v, f, sB)], dtype=np.float64)
                a = a[np.isfinite(a)]
                b = b[np.isfinite(b)]

                nA = int(a.size)
                nB = int(b.size)
                if nA == 0 or nB == 0:
                    continue

                low_n_flag = int((nA < args.min_n_per_state) or (nB < args.min_n_per_state))
                if args.drop_low_state_pairs and low_n_flag:
                    continue

                summ = bootstrap_continuous_ot(
                    a=a,
                    b=b,
                    rng=rng,
                    draws=args.draws,
                )

                ot_emp = w1_continuous_1d(a, b)
                ms_emp = float(a.mean() - b.mean())

                rows_out.append({
                    "attribute": attr,
                    "value_item": v,
                    "factor": f,
                    "state_a": sA,
                    "state_b": sB,
                    "N_a": nA,
                    "N_b": nB,
                    "low_n_flag": low_n_flag,
                    "draws": args.draws,
                    "seed": args.seed,
                    "normalization_mode": args.normalization_mode if not args.raw_npz else "from_raw_npz",
                    "outlier_quantile_low": args.outlier_quantile_low if not args.raw_npz else math.nan,
                    "outlier_quantile_high": args.outlier_quantile_high if not args.raw_npz else math.nan,
                    "outlier_action": args.outlier_action if not args.raw_npz else "",
                    "ot_empirical": float(ot_emp),
                    "mean_shift_empirical": float(ms_emp),
                    "ot_mean": summ["ot_mean"],
                    "ot_q025": summ["ot_q025"],
                    "ot_q975": summ["ot_q975"],
                    "ot_ci_width": (
                        summ["ot_q975"] - summ["ot_q025"]
                        if (not math.isnan(summ["ot_q975"]) and not math.isnan(summ["ot_q025"]))
                        else math.nan
                    ),
                    "mean_shift": summ["mean_shift"],
                    "mean_shift_q025": summ["mean_shift_q025"],
                    "mean_shift_q975": summ["mean_shift_q975"],
                    "mean_a": float(a.mean()),
                    "mean_b": float(b.mean()),
                    "std_a": float(a.std(ddof=1)) if a.size >= 2 else 0.0,
                    "std_b": float(b.std(ddof=1)) if b.size >= 2 else 0.0,
                    "q05_a": float(np.quantile(a, 0.05)),
                    "q50_a": float(np.quantile(a, 0.50)),
                    "q95_a": float(np.quantile(a, 0.95)),
                    "q05_b": float(np.quantile(b, 0.05)),
                    "q50_b": float(np.quantile(b, 0.50)),
                    "q95_b": float(np.quantile(b, 0.95)),
                })

    df = pd.DataFrame(rows_out)
    if len(df) == 0:
        raise RuntimeError("No OT rows produced.")

    df = df.sort_values(
        ["attribute", "value_item", "factor", "ot_mean"],
        ascending=[True, True, True, False],
    )

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[done] wrote: {args.out_csv} (rows={len(df)})")

    if chosen_layers:
        print("[info] chosen middle layers per model:")
        for m, layer in chosen_layers.items():
            print(f"  - {m}: layer={layer}")


if __name__ == "__main__":
    main()
