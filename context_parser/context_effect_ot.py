#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OT-based context effect modeling (ordinal intensity bins) with:
  - per-value-item ordered category mapping
  - correct NONE-state assignment
  - NEW: per-state-pair counterfactual distribution shift recording (bin-wise Δ with CI)

For each (value_item, factor, state_a, state_b), we output:
  - OT (W1) posterior mean + CI
  - mean shift posterior mean + CI
  - NEW: posterior mean distributions pA, pB and delta = pB - pA per bin (+ CI)
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


def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def get_sentence_key(rec: Dict[str, Any]) -> Tuple[Any, str]:
    idx = rec.get("idx")
    txt = rec.get("target_sentence") or rec.get("target_text") or ""
    return (idx, normalize_text(txt))


def norm_cat(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


# -------------------------
# Load ordered category definitions (PER value_item)
# -------------------------
def load_ordered_category_defs(surface_regime_defs_jsonl: str) -> Dict[str, List[str]]:
    """
    Returns:
      value_item -> ordered list of category names (length typically 5).
    """
    defs = read_jsonl(surface_regime_defs_jsonl, desc="Load surface regime defs JSONL")
    out: Dict[str, List[str]] = {}
    for r in defs:
        v = r.get("value_item")
        cats = r.get("categories")
        if not isinstance(v, str) or not v.strip():
            continue
        if not isinstance(cats, list) or not cats:
            continue
        names = []
        for c in cats:
            if isinstance(c, dict):
                nm = c.get("name")
                if isinstance(nm, str) and nm.strip():
                    names.append(nm.strip())
        if names:
            out[v.strip()] = names
    return out


def build_category_to_bin_per_value_item(
    ordered_defs: Dict[str, List[str]],
) -> Dict[str, Dict[str, int]]:
    """
    For each value_item, map category name -> ordinal bin by ORDER.
    Bin ids are 1..len(categories). Last bin = "Not Invoked/irrelevant" (by position).
    """
    mapping: Dict[str, Dict[str, int]] = {}
    for v, ordered_names in ordered_defs.items():
        m: Dict[str, int] = {}
        for i, name in enumerate(ordered_names, start=1):
            m[norm_cat(name)] = i
        mapping[v] = m
    return mapping


# -------------------------
# OT (Wasserstein-1) on ordinal bins
# -------------------------
def w1_ordinal(p: np.ndarray, q: np.ndarray) -> float:
    if p.shape != q.shape:
        raise ValueError("p and q must have same shape")
    cp = np.cumsum(p)
    cq = np.cumsum(q)
    return float(np.sum(np.abs(cp[:-1] - cq[:-1])))


def expected_bin(p: np.ndarray) -> float:
    k = np.arange(1, p.shape[0] + 1, dtype=np.float64)
    return float(np.dot(k, p))


# -------------------------
# Data model
# -------------------------
@dataclass(frozen=True)
class Sample:
    value_item: str
    factor: str
    state: str
    bin_y: int
    category_name: str
    sent_key: Tuple[Any, str]


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
# Extraction
# -------------------------
def extract_samples(
    context_rows: List[Dict[str, Any]],
    surface_rows: List[Dict[str, Any]],
    cat2bin_per_v: Dict[str, Dict[str, int]],
    ordered_defs: Dict[str, List[str]],
    factor_universe: Dict[str, List[str]],
    skip_not_invoked_value_items: bool = True,
) -> List[Sample]:
    surface_by_key: Dict[Tuple[Any, str], Dict[str, Any]] = {}
    for r in tqdm(surface_rows, desc="Index surface rows", unit="rows"):
        surface_by_key[get_sentence_key(r)] = r

    samples: List[Sample] = []
    n_missing_join = 0
    n_total_context = 0
    n_missing_factor_universe = 0
    n_missing_catmap = 0
    n_skipped_notinvoked = 0

    for cr in tqdm(context_rows, desc="Extract samples", unit="rows"):
        n_total_context += 1
        key = get_sentence_key(cr)
        sr = surface_by_key.get(key)
        if sr is None:
            n_missing_join += 1
            continue

        svr = sr.get("surface_value_regimes") or {}
        ctx_states = cr.get("context_states") or {}
        if not isinstance(svr, dict):
            continue
        if not isinstance(ctx_states, dict):
            ctx_states = {}

        for v, reg in svr.items():
            if not isinstance(v, str) or not v.strip():
                continue
            v = v.strip()
            if not isinstance(reg, dict):
                continue

            cat = reg.get("category")
            if not isinstance(cat, str) or not cat.strip():
                continue
            cat_name = cat.strip()

            m = cat2bin_per_v.get(v)
            ordered = ordered_defs.get(v)
            if m is None or ordered is None:
                n_missing_catmap += 1
                continue

            y = m.get(norm_cat(cat_name))
            if y is None:
                # robust fallback
                ncat = norm_cat(cat_name)
                matched = None
                for i, cname in enumerate(ordered, start=1):
                    if norm_cat(cname) == ncat:
                        matched = i
                        break
                if matched is None:
                    matched = len(ordered)
                y = matched

            # skip "not invoked" = last category by definition order
            if skip_not_invoked_value_items and y == len(ordered):
                n_skipped_notinvoked += 1
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
                    value_item=v,
                    factor=factor,
                    state=st,
                    bin_y=int(y),
                    category_name=cat_name,
                    sent_key=key,
                ))

    print(f"[info] context rows: {n_total_context}")
    print(f"[info] missing surface join: {n_missing_join} ({(n_missing_join/max(1,n_total_context))*100:.2f}%)")
    print(f"[info] missing cat mapping (value_item not in defs): {n_missing_catmap}")
    print(f"[info] missing factor universe (value_item has no factors): {n_missing_factor_universe}")
    print(f"[info] skipped not-invoked samples: {n_skipped_notinvoked}")
    print(f"[info] extracted samples: {len(samples)}")
    return samples


# -------------------------
# Filtering
# -------------------------
def filter_value_items_by_count(samples: List[Sample], min_unique_sentences: int) -> List[Sample]:
    if min_unique_sentences <= 0:
        return samples

    sent_sets: Dict[str, set] = collections.defaultdict(set)
    for s in samples:
        sent_sets[s.value_item].add(s.sent_key)

    counts = {v: len(ss) for v, ss in sent_sets.items()}
    keep = {v for v, c in counts.items() if c >= min_unique_sentences}

    print(f"[info] value_items before filter: {len(counts)}")
    print(f"[info] value_items kept (min_unique_sentences={min_unique_sentences}): {len(keep)}")
    print(f"[info] value_items dropped: {len(counts) - len(keep)}")

    filtered = [s for s in samples if s.value_item in keep]
    print(f"[info] samples after value_item filter: {len(filtered)}")
    return filtered


# -------------------------
# Posterior OT + NEW distributional counterfactual summaries
# -------------------------
def dirichlet_posterior_summary(
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    lam: float,
    rng: np.random.Generator,
    n_draws: int = 2000,
) -> Dict[str, Any]:
    """
    Returns a dict with:
      - OT mean + CI
      - mean-shift mean + CI (E[bin] difference)
      - NEW: pA_mean, pB_mean (vectors)
      - NEW: delta_mean, delta_q025, delta_q975 per bin where delta = pB - pA
    """
    pooled = counts_a + counts_b
    total = float(np.sum(pooled))
    if total <= 0:
        return {
            "ot_mean": math.nan, "ot_q025": math.nan, "ot_q975": math.nan,
            "mean_shift": math.nan, "mean_shift_q025": math.nan, "mean_shift_q975": math.nan,
            "pA_mean": None, "pB_mean": None,
            "delta_mean": None, "delta_q025": None, "delta_q975": None,
        }

    mu = pooled / total
    alpha = lam * mu

    a_a = counts_a + alpha
    a_b = counts_b + alpha

    pA = rng.dirichlet(a_a, size=n_draws)
    pB = rng.dirichlet(a_b, size=n_draws)

    ots = np.empty((n_draws,), dtype=np.float64)
    shifts = np.empty((n_draws,), dtype=np.float64)

    # NEW: per-draw deltas
    deltas = pB - pA  # shape (draws, K)

    for i in range(n_draws):
        ots[i] = w1_ordinal(pA[i], pB[i])
        shifts[i] = expected_bin(pA[i]) - expected_bin(pB[i])

    out = {
        "ot_mean": float(np.mean(ots)),
        "ot_q025": float(np.quantile(ots, 0.025)),
        "ot_q975": float(np.quantile(ots, 0.975)),
        "mean_shift": float(np.mean(shifts)),
        "mean_shift_q025": float(np.quantile(shifts, 0.025)),
        "mean_shift_q975": float(np.quantile(shifts, 0.975)),
        "pA_mean": np.mean(pA, axis=0),
        "pB_mean": np.mean(pB, axis=0),
        "delta_mean": np.mean(deltas, axis=0),
        "delta_q025": np.quantile(deltas, 0.025, axis=0),
        "delta_q975": np.quantile(deltas, 0.975, axis=0),
    }
    return out


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--context_jsonl", required=True)
    ap.add_argument("--surface_jsonl", required=True)
    ap.add_argument("--factors_json", default=None)

    ap.add_argument("--surface_regime_defs_jsonl", required=True)

    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_counts_csv", default=None)

    ap.add_argument("--lambda_prior", type=float, default=20.0)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--skip_not_invoked", action="store_true", default=True)
    ap.add_argument("--no_skip_not_invoked", dest="skip_not_invoked", action="store_false")

    ap.add_argument("--min_n_per_state", type=int, default=5)
    ap.add_argument("--min_value_item_count", type=int, default=200)

    # NEW: include bin-wise distribution deltas in output CSV
    ap.add_argument("--write_bin_deltas", action="store_true", default=True)
    ap.add_argument("--no_write_bin_deltas", dest="write_bin_deltas", action="store_false")

    args = ap.parse_args()

    print("[stage] load jsonl")
    context_rows = read_jsonl(args.context_jsonl, desc="Load context JSONL")
    surface_rows = read_jsonl(args.surface_jsonl, desc="Load surface JSONL")

    print("[stage] load ordered surface regime definitions")
    ordered_defs = load_ordered_category_defs(args.surface_regime_defs_jsonl)
    cat2bin_per_v = build_category_to_bin_per_value_item(ordered_defs)
    print(f"[info] loaded ordered defs for value_items: {len(ordered_defs)}")

    print("[stage] factor universe")
    if args.factors_json:
        factor_universe = load_factor_universe_from_factors_json(args.factors_json)
        print(f"[info] factor universe loaded from factors_json: {len(factor_universe)} value_items")
    else:
        factor_universe = infer_factor_universe_from_context_rows(context_rows)
        print(f"[info] factor universe inferred from context_rows: {len(factor_universe)} value_items")

    print("[stage] extract samples")
    samples = extract_samples(
        context_rows=context_rows,
        surface_rows=surface_rows,
        cat2bin_per_v=cat2bin_per_v,
        ordered_defs=ordered_defs,
        factor_universe=factor_universe,
        skip_not_invoked_value_items=args.skip_not_invoked,
    )
    if not samples:
        raise RuntimeError("No samples extracted.")

    print("[stage] filter value items by count")
    samples = filter_value_items_by_count(samples, args.min_value_item_count)
    if not samples:
        raise RuntimeError("All samples filtered out by --min_value_item_count.")

    max_bin = max(s.bin_y for s in samples)
    K = max_bin
    print(f"[info] using K={K} bins (invoked bins only)")

    print("[stage] build counts")
    counts: Dict[Tuple[str, str, str], np.ndarray] = {}
    Ns: Dict[Tuple[str, str, str], int] = collections.Counter()

    for s in tqdm(samples, desc="Count samples", unit="samples"):
        key = (s.value_item, s.factor, s.state)
        if key not in counts:
            counts[key] = np.zeros((K,), dtype=np.float64)
        if 1 <= s.bin_y <= K:
            counts[key][s.bin_y - 1] += 1.0
            Ns[key] += 1

    if args.out_counts_csv:
        rows_counts = []
        for (v, f, st), arr in counts.items():
            row = {"value_item": v, "factor": f, "state": st, "N": int(arr.sum())}
            for b in range(K):
                row[f"bin_{b+1}"] = float(arr[b])
            rows_counts.append(row)
        df_counts = pd.DataFrame(rows_counts).sort_values(["value_item", "factor", "state"])
        os.makedirs(os.path.dirname(args.out_counts_csv) or ".", exist_ok=True)
        df_counts.to_csv(args.out_counts_csv, index=False)
        print(f"[info] wrote debug counts: {args.out_counts_csv}")

    print("[stage] compute OT + distribution deltas")
    rng = np.random.default_rng(args.seed)
    rows_out = []

    by_vf: Dict[Tuple[str, str], List[str]] = collections.defaultdict(list)
    for (v, f, st) in counts.keys():
        by_vf[(v, f)].append(st)

    for (v, f), states in tqdm(list(by_vf.items()), desc="Per (value_item,factor)", unit="vf"):
        states = sorted(set(states))
        if len(states) < 2:
            continue

        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                sA = states[i]
                sB = states[j]

                cA = counts[(v, f, sA)]
                cB = counts[(v, f, sB)]
                nA = int(Ns[(v, f, sA)])
                nB = int(Ns[(v, f, sB)])

                if nA == 0 or nB == 0:
                    continue

                summ = dirichlet_posterior_summary(
                    counts_a=cA,
                    counts_b=cB,
                    lam=args.lambda_prior,
                    rng=rng,
                    n_draws=args.draws,
                )

                row = {
                    "value_item": v,
                    "factor": f,
                    "state_a": sA,
                    "state_b": sB,
                    "N_a": nA,
                    "N_b": nB,
                    "low_n_flag": int((nA < args.min_n_per_state) or (nB < args.min_n_per_state)),
                    "K_bins": K,
                    "lambda_prior": args.lambda_prior,
                    "draws": args.draws,
                    "seed": args.seed,
                    "ot_mean": summ["ot_mean"],
                    "ot_q025": summ["ot_q025"],
                    "ot_q975": summ["ot_q975"],
                    "ot_ci_width": (summ["ot_q975"] - summ["ot_q025"]) if (not math.isnan(summ["ot_q975"]) and not math.isnan(summ["ot_q025"])) else math.nan,
                    "mean_shift": summ["mean_shift"],
                    "mean_shift_q025": summ["mean_shift_q025"],
                    "mean_shift_q975": summ["mean_shift_q975"],
                }

                if args.write_bin_deltas and summ["pA_mean"] is not None:
                    pA_mean = summ["pA_mean"]
                    pB_mean = summ["pB_mean"]
                    d_mean = summ["delta_mean"]
                    d_q025 = summ["delta_q025"]
                    d_q975 = summ["delta_q975"]

                    for b in range(K):
                        k = b + 1
                        row[f"pA_bin{k}_mean"] = float(pA_mean[b])
                        row[f"pB_bin{k}_mean"] = float(pB_mean[b])
                        row[f"delta_bin{k}_mean"] = float(d_mean[b])     # pB - pA
                        row[f"delta_bin{k}_q025"] = float(d_q025[b])
                        row[f"delta_bin{k}_q975"] = float(d_q975[b])

                rows_out.append(row)

    df = pd.DataFrame(rows_out)
    if len(df) == 0:
        raise RuntimeError("No OT rows produced.")

    df = df.sort_values(["value_item", "factor", "ot_mean"], ascending=[True, True, False])

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[done] wrote: {args.out_csv} (rows={len(df)})")


if __name__ == "__main__":
    main()
