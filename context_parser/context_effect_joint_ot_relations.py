#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fast joint context-effect modeling for value-relation graphs.

Compared with the LP-based version, this script is much faster because:
1) it computes posterior mean distributions by Dirichlet sampling
2) it computes OT ONLY ONCE per state-pair, on posterior means
3) it prunes support per state-pair
4) it uses a fast structured OT approximation instead of generic LP

Joint object:
    P(partner_value=v_j, relation=r | anchor=v_i, factor=f, state=s)

Outputs:
  1) out_csv        : per-(anchor, factor, state_a, state_b) summary
  2) out_counts_csv : long-form joint counts by (anchor, factor, state, partner, relation)
  3) out_delta_csv  : long-form posterior mean deltas per joint cell: pB - pA

Metrics:
  - joint_ot_fast_mean : fast structured OT on posterior means
  - joint_jsd_mean     : posterior mean JSD across Dirichlet draws
  - value_jsd_mean     : posterior mean JSD on partner-value marginal
  - relation_jsd_mean  : posterior mean JSD on relation marginal

The fast OT is:
  OT_fast = OT_within_partner_relations + OT_across_partners

where
- within-partner mass is moved across relation labels with relation costs
- leftover mass is moved across partners with base cost alpha_value

This is not identical to full LP OT, but is much faster and still aligned
with the same semantics.
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


VALID_STATUSES = {"ACTIVATED", "EXPRESSED", "RELATED"}

DEFAULT_RELATION_COSTS = {
    ("PREREQUISITE", "PREREQUISITE"): 0.0,
    ("MEANS_END", "MEANS_END"): 0.0,
    ("JUSTIFIES", "JUSTIFIES"): 0.0,
    ("SYNERGY", "SYNERGY"): 0.0,
    ("CONFLICT", "CONFLICT"): 0.0,
    ("TRADEOFF", "TRADEOFF"): 0.0,
    ("CONSTRAINT", "CONSTRAINT"): 0.0,

    ("PREREQUISITE", "MEANS_END"): 0.35,
    ("MEANS_END", "PREREQUISITE"): 0.35,

    ("PREREQUISITE", "JUSTIFIES"): 0.50,
    ("JUSTIFIES", "PREREQUISITE"): 0.50,

    ("MEANS_END", "JUSTIFIES"): 0.45,
    ("JUSTIFIES", "MEANS_END"): 0.45,

    ("SYNERGY", "PREREQUISITE"): 0.55,
    ("PREREQUISITE", "SYNERGY"): 0.55,

    ("SYNERGY", "MEANS_END"): 0.50,
    ("MEANS_END", "SYNERGY"): 0.50,

    ("SYNERGY", "JUSTIFIES"): 0.60,
    ("JUSTIFIES", "SYNERGY"): 0.60,

    ("CONFLICT", "TRADEOFF"): 0.35,
    ("TRADEOFF", "CONFLICT"): 0.35,

    ("CONFLICT", "CONSTRAINT"): 0.45,
    ("CONSTRAINT", "CONFLICT"): 0.45,

    ("TRADEOFF", "CONSTRAINT"): 0.45,
    ("CONSTRAINT", "TRADEOFF"): 0.45,
}


@dataclass(frozen=True)
class JointEvent:
    anchor_value: str
    factor: str
    state: str
    partner_value: str
    relation: str
    sent_key: Tuple[Any, str]


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


def load_relation_costs(path: Optional[str]) -> Dict[Tuple[str, str], float]:
    if path is None:
        return dict(DEFAULT_RELATION_COSTS)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = {}
    for k, v in data.items():
        if "|||" not in k:
            continue
        a, b = k.split("|||", 1)
        out[(a.strip(), b.strip())] = float(v)
    return out


def relation_distance(r1: str, r2: str, rel_costs: Dict[Tuple[str, str], float]) -> float:
    if r1 == r2:
        return 0.0
    if (r1, r2) in rel_costs:
        return rel_costs[(r1, r2)]
    if (r2, r1) in rel_costs:
        return rel_costs[(r2, r1)]

    positive = {"PREREQUISITE", "MEANS_END", "JUSTIFIES", "SYNERGY"}
    negative = {"CONFLICT", "TRADEOFF", "CONSTRAINT"}

    if (r1 in positive and r2 in positive) or (r1 in negative and r2 in negative):
        return 0.65
    return 1.0


def safe_normalize(x: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.isfinite(x), x, 0.0)
    x = np.clip(x, 0.0, None)

    s = float(np.sum(x))
    if s <= eps:
        return np.ones_like(x, dtype=np.float64) / len(x)

    x = x / s
    x = x / float(np.sum(x))
    return x


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = safe_normalize(p)
    q = safe_normalize(q)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        a = np.clip(a, eps, 1.0)
        b = np.clip(b, eps, 1.0)
        return float(np.sum(a * np.log(a / b)))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def extract_joint_events(
    context_rows: List[Dict[str, Any]],
    relation_rows: List[Dict[str, Any]],
    factor_universe: Dict[str, List[str]],
    valid_statuses: Set[str],
) -> List[JointEvent]:
    context_by_key: Dict[Tuple[Any, str], Dict[str, Any]] = {}
    for r in tqdm(context_rows, desc="Index context rows", unit="rows"):
        context_by_key[get_sentence_key(r)] = r

    events: List[JointEvent] = []
    n_missing_join = 0
    n_total_relation = 0
    n_missing_factor_universe = 0
    n_valid_edges = 0

    for rr in tqdm(relation_rows, desc="Extract joint events", unit="rows"):
        n_total_relation += 1
        key = get_sentence_key(rr)
        cr = context_by_key.get(key)
        if cr is None:
            n_missing_join += 1
            continue

        graph = rr.get("value_relation_graph") or {}
        values = graph.get("values") or []
        edges = graph.get("edges") or []
        ctx_states = cr.get("context_states") or {}
        if not isinstance(ctx_states, dict):
            ctx_states = {}

        valid_values = set()
        for v in values:
            value_item = v.get("value_item")
            status = v.get("status")
            if isinstance(value_item, str) and value_item.strip() and status in valid_statuses:
                valid_values.add(value_item.strip())

        for e in edges:
            src = e.get("source")
            tgt = e.get("target")
            rel = e.get("relation")

            if not (isinstance(src, str) and isinstance(tgt, str) and isinstance(rel, str)):
                continue

            src = src.strip()
            tgt = tgt.strip()
            rel = rel.strip()

            if not src or not tgt or not rel:
                continue
            if src == tgt:
                continue
            if src not in valid_values or tgt not in valid_values:
                continue

            n_valid_edges += 1

            facs = factor_universe.get(src)
            if not facs:
                n_missing_factor_universe += 1
                continue

            src_ctx = ctx_states.get(src)
            if not isinstance(src_ctx, dict):
                src_ctx = {}

            for factor in facs:
                obj = src_ctx.get(factor)
                st = None
                if isinstance(obj, dict):
                    st = obj.get("selected_state")
                if not isinstance(st, str) or not st.strip():
                    st = "NONE"
                st = st.strip()

                events.append(
                    JointEvent(
                        anchor_value=src,
                        factor=factor,
                        state=st,
                        partner_value=tgt,
                        relation=rel,
                        sent_key=key,
                    )
                )

    print(f"[info] relation rows: {n_total_relation}")
    print(f"[info] missing context join: {n_missing_join} ({(n_missing_join / max(1, n_total_relation))*100:.2f}%)")
    print(f"[info] valid edges processed: {n_valid_edges}")
    print(f"[info] missing factor universe for anchor: {n_missing_factor_universe}")
    print(f"[info] extracted joint events: {len(events)}")
    return events


def filter_anchor_values_by_count(
    events: List[JointEvent],
    min_unique_sentences: int,
) -> List[JointEvent]:
    if min_unique_sentences <= 0:
        return events

    sent_sets: Dict[str, set] = collections.defaultdict(set)
    for ev in events:
        sent_sets[ev.anchor_value].add(ev.sent_key)

    counts = {v: len(ss) for v, ss in sent_sets.items()}
    keep = {v for v, c in counts.items() if c >= min_unique_sentences}

    print(f"[info] anchor values before filter: {len(counts)}")
    print(f"[info] anchor values kept (min_unique_sentences={min_unique_sentences}): {len(keep)}")
    print(f"[info] anchor values dropped: {len(counts) - len(keep)}")

    filtered = [ev for ev in events if ev.anchor_value in keep]
    print(f"[info] events after anchor filter: {len(filtered)}")
    return filtered


def posterior_summary_and_deltas(
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    rng: np.random.Generator,
    lam: float,
    n_draws: int,
) -> Dict[str, Any]:
    pooled = counts_a + counts_b
    total = float(np.sum(pooled))
    if total <= 0:
        return {
            "pA_mean": None,
            "pB_mean": None,
            "delta_mean": None,
            "delta_q025": None,
            "delta_q975": None,
            "joint_jsd_mean": math.nan,
            "joint_jsd_q025": math.nan,
            "joint_jsd_q975": math.nan,
        }

    mu = pooled / total
    alpha = lam * mu

    a_a = counts_a + alpha
    a_b = counts_b + alpha

    pA = rng.dirichlet(a_a, size=n_draws)
    pB = rng.dirichlet(a_b, size=n_draws)

    jsds = np.empty((n_draws,), dtype=np.float64)
    for i in range(n_draws):
        jsds[i] = js_divergence(pA[i], pB[i])

    deltas = pB - pA

    return {
        "pA_mean": np.mean(pA, axis=0),
        "pB_mean": np.mean(pB, axis=0),
        "delta_mean": np.mean(deltas, axis=0),
        "delta_q025": np.quantile(deltas, 0.025, axis=0),
        "delta_q975": np.quantile(deltas, 0.975, axis=0),
        "joint_jsd_mean": float(np.mean(jsds)),
        "joint_jsd_q025": float(np.quantile(jsds, 0.025)),
        "joint_jsd_q975": float(np.quantile(jsds, 0.975)),
    }


def build_support_metadata(
    support: List[Tuple[str, str]]
) -> Tuple[List[str], List[str], Dict[str, List[int]], Dict[str, List[int]]]:
    partners = sorted({p for p, _ in support})
    relations = sorted({r for _, r in support})

    idxs_by_partner: Dict[str, List[int]] = collections.defaultdict(list)
    idxs_by_relation: Dict[str, List[int]] = collections.defaultdict(list)

    for idx, (p, r) in enumerate(support):
        idxs_by_partner[p].append(idx)
        idxs_by_relation[r].append(idx)

    return partners, relations, idxs_by_partner, idxs_by_relation


def marginalize_by_groups(vec: np.ndarray, groups: List[str], idxs_by_group: Dict[str, List[int]]) -> np.ndarray:
    out = np.zeros((len(groups),), dtype=np.float64)
    for i, g in enumerate(groups):
        idxs = idxs_by_group[g]
        out[i] = float(np.sum(vec[idxs]))
    return safe_normalize(out)


def min_cost_match_same_partner(
    p_rel: np.ndarray,
    q_rel: np.ndarray,
    rel_labels: List[str],
    rel_costs: Dict[Tuple[str, str], float],
    beta_relation: float,
) -> float:
    """
    Fast transportation cost within one partner over relation labels.

    Greedy exact for tiny supports is enough here because relation label count is small.
    """
    p = safe_normalize(p_rel.copy())
    q = safe_normalize(q_rel.copy())
    m1 = float(np.sum(p_rel))
    m2 = float(np.sum(q_rel))
    mass = min(m1, m2)
    if mass <= 0:
        return 0.0

    # work on raw masses, not normalized probabilities
    p = p_rel.copy()
    q = q_rel.copy()

    cost = 0.0

    # same relation first
    for i in range(len(rel_labels)):
        moved = min(p[i], q[i])
        p[i] -= moved
        q[i] -= moved

    # then cheapest cross-relation moves
    pairs = []
    for i, r1 in enumerate(rel_labels):
        for j, r2 in enumerate(rel_labels):
            if i == j:
                continue
            c = beta_relation * relation_distance(r1, r2, rel_costs)
            pairs.append((c, i, j))
    pairs.sort(key=lambda x: x[0])

    for c, i, j in pairs:
        if p[i] <= 0 or q[j] <= 0:
            continue
        moved = min(p[i], q[j])
        p[i] -= moved
        q[j] -= moved
        cost += moved * c

    return float(cost)


def fast_structured_joint_ot(
    p_joint: np.ndarray,
    q_joint: np.ndarray,
    support: List[Tuple[str, str]],
    rel_costs: Dict[Tuple[str, str], float],
    alpha_value: float,
    beta_relation: float,
) -> Tuple[float, float, float]:
    """
    Structured fast OT approximation:
      total = within-partner relation cost + across-partner cost

    across-partner cost uses leftover partner marginals after same-partner matching.
    Since value distance is 0 if same partner else 1, moving mass across partners
    costs alpha_value regardless of destination partner.
    """
    p_joint = safe_normalize(p_joint)
    q_joint = safe_normalize(q_joint)

    partners, relations, idxs_by_partner, _ = build_support_metadata(support)

    support_index = {cell: i for i, cell in enumerate(support)}
    rel_to_idx = {r: i for i, r in enumerate(relations)}

    within_cost = 0.0
    matched_within_total = 0.0

    partner_p_total = {}
    partner_q_total = {}

    for partner in partners:
        idxs = idxs_by_partner[partner]
        p_mass = float(np.sum(p_joint[idxs]))
        q_mass = float(np.sum(q_joint[idxs]))
        partner_p_total[partner] = p_mass
        partner_q_total[partner] = q_mass

        if p_mass <= 0 and q_mass <= 0:
            continue

        p_rel = np.zeros((len(relations),), dtype=np.float64)
        q_rel = np.zeros((len(relations),), dtype=np.float64)

        for rel in relations:
            idx = support_index.get((partner, rel))
            if idx is not None:
                p_rel[rel_to_idx[rel]] = p_joint[idx]
                q_rel[rel_to_idx[rel]] = q_joint[idx]

        matched_mass = min(p_mass, q_mass)
        matched_within_total += matched_mass

        if matched_mass > 0:
            within_cost += min_cost_match_same_partner(
                p_rel=p_rel,
                q_rel=q_rel,
                rel_labels=relations,
                rel_costs=rel_costs,
                beta_relation=beta_relation,
            )

    # leftover mass that must cross partners
    # because cross-partner value distance is constant (=1), OT reduces to total unmatched mass / 2 * alpha_value
    p_partner = np.array([partner_p_total[p] for p in partners], dtype=np.float64)
    q_partner = np.array([partner_q_total[p] for p in partners], dtype=np.float64)

    leftover = 0.5 * float(np.sum(np.abs(p_partner - q_partner)))
    across_cost = alpha_value * leftover

    total_cost = within_cost + across_cost
    return float(total_cost), float(within_cost), float(across_cost)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--context_jsonl", required=True)
    ap.add_argument("--relation_jsonl", required=True)
    ap.add_argument("--factors_json", default=None)

    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_counts_csv", required=True)
    ap.add_argument("--out_delta_csv", required=True)

    ap.add_argument("--lambda_prior", type=float, default=20.0)
    ap.add_argument("--draws", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--min_n_per_state", type=int, default=5)
    ap.add_argument("--min_anchor_value_count", type=int, default=200)
    ap.add_argument("--min_partner_count", type=int, default=5)
    ap.add_argument("--top_k_partners", type=int, default=10)

    ap.add_argument("--alpha_value", type=float, default=1.0)
    ap.add_argument("--beta_relation", type=float, default=0.5)
    ap.add_argument("--relation_cost_json", type=str, default=None)

    args = ap.parse_args()

    print("[stage] load inputs")
    context_rows = read_jsonl(args.context_jsonl, desc="Load context JSONL")
    relation_rows = read_jsonl(args.relation_jsonl, desc="Load relation JSONL")

    print("[stage] factor universe")
    if args.factors_json:
        factor_universe = load_factor_universe_from_factors_json(args.factors_json)
        print(f"[info] factor universe loaded from factors_json: {len(factor_universe)} value_items")
    else:
        factor_universe = infer_factor_universe_from_context_rows(context_rows)
        print(f"[info] factor universe inferred from context_rows: {len(factor_universe)} value_items")

    rel_costs = load_relation_costs(args.relation_cost_json)

    print("[stage] extract joint events")
    events = extract_joint_events(
        context_rows=context_rows,
        relation_rows=relation_rows,
        factor_universe=factor_universe,
        valid_statuses=VALID_STATUSES,
    )
    if not events:
        raise RuntimeError("No joint events extracted.")

    print("[stage] filter anchor values by count")
    events = filter_anchor_values_by_count(events, args.min_anchor_value_count)
    if not events:
        raise RuntimeError("All joint events filtered out by --min_anchor_value_count.")

    print("[stage] pooled support selection")
    pooled_partner_counts: Dict[Tuple[str, str], collections.Counter] = collections.defaultdict(collections.Counter)
    pooled_joint_counts: Dict[Tuple[str, str], collections.Counter] = collections.defaultdict(collections.Counter)

    for ev in tqdm(events, desc="Accumulate pooled counts", unit="events"):
        key = (ev.anchor_value, ev.factor)
        pooled_partner_counts[key][ev.partner_value] += 1
        pooled_joint_counts[key][(ev.partner_value, ev.relation)] += 1

    allowed_support: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}

    for key, partner_counter in tqdm(pooled_partner_counts.items(), desc="Select support", unit="vf"):
        filtered_partners = [(p, c) for p, c in partner_counter.items() if c >= args.min_partner_count]
        filtered_partners = sorted(filtered_partners, key=lambda x: (-x[1], x[0]))[:args.top_k_partners]
        kept_partner_set = {p for p, _ in filtered_partners}

        support_cells = []
        joint_counter = pooled_joint_counts[key]
        for (partner, rel), c in joint_counter.items():
            if partner in kept_partner_set:
                support_cells.append(((partner, rel), c))

        support_cells = sorted(support_cells, key=lambda x: (-x[1], x[0][0], x[0][1]))
        allowed_support[key] = [cell for cell, _ in support_cells]

    print("[stage] build state-conditioned joint counts")
    counts: Dict[Tuple[str, str, str], collections.Counter] = collections.defaultdict(collections.Counter)
    Ns: Dict[Tuple[str, str, str], int] = collections.Counter()

    support_sets = {k: set(v) for k, v in allowed_support.items()}

    for ev in tqdm(events, desc="Count state-conditioned events", unit="events"):
        vf = (ev.anchor_value, ev.factor)
        support_set = support_sets.get(vf, set())
        cell = (ev.partner_value, ev.relation)
        if cell not in support_set:
            continue

        key = (ev.anchor_value, ev.factor, ev.state)
        counts[key][cell] += 1
        Ns[key] += 1

    rows_counts = []
    for (anchor, factor, state), ctr in counts.items():
        total = sum(ctr.values())
        for (partner, rel), c in sorted(ctr.items(), key=lambda x: (-x[1], x[0][0], x[0][1])):
            rows_counts.append({
                "anchor_value": anchor,
                "factor": factor,
                "state": state,
                "partner_value": partner,
                "relation": rel,
                "count": int(c),
                "empirical_prob": float(c / total) if total > 0 else math.nan,
                "support_N_state": int(total),
            })

    df_counts = pd.DataFrame(rows_counts)
    if len(df_counts) == 0:
        raise RuntimeError("No counts remained after support filtering.")

    os.makedirs(os.path.dirname(args.out_counts_csv) or ".", exist_ok=True)
    df_counts.sort_values(
        ["anchor_value", "factor", "state", "count"],
        ascending=[True, True, True, False],
    ).to_csv(args.out_counts_csv, index=False)
    print(f"[done] wrote counts: {args.out_counts_csv} (rows={len(df_counts)})")

    print("[stage] compute fast joint summaries")
    rng = np.random.default_rng(args.seed)
    rows_out = []
    rows_delta = []

    by_vf_states: Dict[Tuple[str, str], List[str]] = collections.defaultdict(list)
    for (anchor, factor, state) in counts.keys():
        by_vf_states[(anchor, factor)].append(state)

    for (anchor, factor), states in tqdm(list(by_vf_states.items()), desc="Per (anchor,factor)", unit="vf"):
        states = sorted(set(states))
        if len(states) < 2:
            continue

        support_full = allowed_support.get((anchor, factor), [])
        if not support_full:
            continue

        support_index_full = {cell: i for i, cell in enumerate(support_full)}

        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                sA = states[i]
                sB = states[j]

                keyA = (anchor, factor, sA)
                keyB = (anchor, factor, sB)

                nA = int(Ns[keyA])
                nB = int(Ns[keyB])

                if nA == 0 or nB == 0:
                    continue

                vecA_full = np.zeros((len(support_full),), dtype=np.float64)
                vecB_full = np.zeros((len(support_full),), dtype=np.float64)

                for cell, c in counts[keyA].items():
                    idx = support_index_full.get(cell)
                    if idx is not None:
                        vecA_full[idx] = float(c)
                for cell, c in counts[keyB].items():
                    idx = support_index_full.get(cell)
                    if idx is not None:
                        vecB_full[idx] = float(c)

                # pair-specific pruning
                keep = (vecA_full > 0) | (vecB_full > 0)
                if not np.any(keep):
                    continue

                vecA = vecA_full[keep]
                vecB = vecB_full[keep]
                support_pair = [support_full[idx] for idx, k in enumerate(keep) if k]

                summ = posterior_summary_and_deltas(
                    counts_a=vecA,
                    counts_b=vecB,
                    rng=rng,
                    lam=args.lambda_prior,
                    n_draws=args.draws,
                )

                pA_mean = summ["pA_mean"]
                pB_mean = summ["pB_mean"]
                d_mean = summ["delta_mean"]
                d_q025 = summ["delta_q025"]
                d_q975 = summ["delta_q975"]

                if pA_mean is None or pB_mean is None:
                    continue

                joint_ot_fast_mean, within_cost, across_cost = fast_structured_joint_ot(
                    p_joint=pA_mean,
                    q_joint=pB_mean,
                    support=support_pair,
                    rel_costs=rel_costs,
                    alpha_value=args.alpha_value,
                    beta_relation=args.beta_relation,
                )

                # marginals
                partners, relations, idxs_by_partner, idxs_by_relation = build_support_metadata(support_pair)
                pA_value = marginalize_by_groups(pA_mean, partners, idxs_by_partner)
                pB_value = marginalize_by_groups(pB_mean, partners, idxs_by_partner)
                pA_rel = marginalize_by_groups(pA_mean, relations, idxs_by_relation)
                pB_rel = marginalize_by_groups(pB_mean, relations, idxs_by_relation)

                value_jsd_mean = js_divergence(pA_value, pB_value)
                relation_jsd_mean = js_divergence(pA_rel, pB_rel)

                row = {
                    "anchor_value": anchor,
                    "factor": factor,
                    "state_a": sA,
                    "state_b": sB,
                    "N_a": nA,
                    "N_b": nB,
                    "low_n_flag": int((nA < args.min_n_per_state) or (nB < args.min_n_per_state)),
                    "support_size_pair": len(support_pair),
                    "top_k_partners": args.top_k_partners,
                    "min_partner_count": args.min_partner_count,
                    "lambda_prior": args.lambda_prior,
                    "draws": args.draws,
                    "seed": args.seed,
                    "alpha_value": args.alpha_value,
                    "beta_relation": args.beta_relation,
                    "joint_ot_fast_mean": joint_ot_fast_mean,
                    "joint_ot_fast_within_relation": within_cost,
                    "joint_ot_fast_across_partner": across_cost,
                    "joint_jsd_mean": summ["joint_jsd_mean"],
                    "joint_jsd_q025": summ["joint_jsd_q025"],
                    "joint_jsd_q975": summ["joint_jsd_q975"],
                    "value_jsd_mean": value_jsd_mean,
                    "relation_jsd_mean": relation_jsd_mean,
                }
                rows_out.append(row)

                for idx, (partner, rel) in enumerate(support_pair):
                    rows_delta.append({
                        "anchor_value": anchor,
                        "factor": factor,
                        "state_a": sA,
                        "state_b": sB,
                        "partner_value": partner,
                        "relation": rel,
                        "pA_mean": float(pA_mean[idx]),
                        "pB_mean": float(pB_mean[idx]),
                        "delta_mean": float(d_mean[idx]),
                        "delta_q025": float(d_q025[idx]),
                        "delta_q975": float(d_q975[idx]),
                        "N_a": nA,
                        "N_b": nB,
                        "support_size_pair": len(support_pair),
                    })

    df = pd.DataFrame(rows_out)
    if len(df) == 0:
        raise RuntimeError("No summary rows produced.")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.sort_values(
        ["anchor_value", "factor", "joint_ot_fast_mean"],
        ascending=[True, True, False],
    ).to_csv(args.out_csv, index=False)
    print(f"[done] wrote summary: {args.out_csv} (rows={len(df)})")

    df_delta = pd.DataFrame(rows_delta)
    if len(df_delta) > 0:
        os.makedirs(os.path.dirname(args.out_delta_csv) or ".", exist_ok=True)
        df_delta.sort_values(
            ["anchor_value", "factor", "state_a", "state_b", "delta_mean"],
            ascending=[True, True, True, True, False],
        ).to_csv(args.out_delta_csv, index=False)
        print(f"[done] wrote deltas: {args.out_delta_csv} (rows={len(df_delta)})")


if __name__ == "__main__":
    main()
