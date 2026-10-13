#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_value_relations_kg.py

FAST LLM-based VALUE-VALUE relation extraction (KG-style, schema-constrained).

Changes for speed
- NO per-batch/local tqdm (only ONE global tqdm).
- NO evidence_quotes and NO confidence fields (shorter outputs, faster decoding).
- No batching pipeline: processes row-by-row (simpler, lower overhead; still flushes periodically).

Input JSONL (one record per line) example:
{
  "idx": 0,
  "text_type": "...",
  "full_text": "...",
  "target_sentence": "...",
  "value_items": [{"value_item": "informed decision-making", "score": 85}, ...]
}

Max LLM calls per row: 1 (extractor) + optional 1 (verifier) = 2

Output field added:
value_relation_graph = {
  "values": [{"value_item": str, "status": ...}],
  "edges": [{"source": str, "target": str, "relation": ..., "direction": ..., "polarity": ...}],
  "notes": str,
  "meta": {...}
}

"""

from __future__ import annotations

import os
import re
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# -------------------------------
# Inference backend
# -------------------------------
# The inference backend is imported when the extractor is constructed.


# ============================================================
# Relation schema (fixed labels)
# ============================================================

RELATION_LABELS: List[str] = [
    "CO_OCCURS",
    "SYNERGY",
    "MEANS_END",
    "PREREQUISITE",
    "JUSTIFIES",
    "CONFLICT",
    "TRADEOFF",
    "CONSTRAINT",
    "PRIORITIZES",
    "RESOLVES_BY",
    "SUBSTITUTES",
]

VALUE_STATUS: List[str] = ["EXPRESSED", "ACTIVATED", "MENTIONED", "RELATED", "ABSENT"]


# ============================================================
# Prompting (Extractor + Verifier)
# ============================================================

def build_system_prompt_extractor() -> str:
    return (
        "You are an information extraction system that builds a SMALL knowledge graph about HUMAN VALUES.\n"
        "You will be given:\n"
        "- TARGET_SENTENCE (primary focus)\n"
        "- EVIDENCE_TEXT (target + optional context window)\n"
        "- VALUE_ITEMS (candidate value items)\n\n"

        "IMPORTANT CONTEXT\n"
        "- VALUE_ITEMS are PRE-FILTERED as relevant to this text.\n"

        "TASK A: VALUE STATUS (FOR EACH VALUE_ITEM)\n"
        "Assign exactly ONE status using this decision procedure (in order):\n"
        "1) EXPRESSED: the text takes a stance (endorses/condemns/praises/blames/argues for/against) about the value.\n"
        "2) MENTIONED: the value is explicitly referenced (named) but no stance is taken.\n"
        "3) ACTIVATED: the value is strongly implicated for interpreting the claim, but not named.\n"
        "4) RELATED: weak/indirect topical connection.\n"
        "5) ABSENT: clearly irrelevant OR contradicted.\n\n"

        "TASK B: VALUE–VALUE EDGES (ALLOW LIGHT IMPLICIT INFERENCE)\n"
        "Extract relationships between TWO values using TARGET_SENTENCE + EVIDENCE_TEXT.\n"
        "You may include edges that are not directly stated word-for-word, but ONLY when they are strongly supported by context.\n"
        "Do NOT add edges based on generic ideology or typical associations.\n\n"

        "EDGE GATING (TWO TIERS)\n"
        "You may output an edge if EITHER Tier-1 or Tier-2 holds.\n\n"

        "Tier-1 (EXPLICIT):\n"
        "- EVIDENCE_TEXT contains an explicit relational cue that links the two values (cause/means/require/contrast/priority/etc.).\n"

        "Tier-2 (IMPLIED):\n"
        "- The sentence/context expresses a clear argumentative structure (justification, requirement, contrast, trade-off,\n"
        "  constraint, prioritization, substitution), AND\n"
        "- BOTH values are at least ACTIVATED (or stronger) in EVIDENCE_TEXT, AND\n"
        "- The relation can be inferred without outside knowledge (only from EVIDENCE_TEXT).\n\n"

        "IMPLICIT EDGE RULES (Tier-2)\n"
        "- If the relation is merely 'these two values often go together', DO NOT add an edge.\n"
        "- If unsure, omit the edge.\n"

        "RELATION LABELS (choose exactly one per edge)\n"
        f"{', '.join(RELATION_LABELS)}\n\n"

        "LABEL DEFINITIONS\n"
        "- CO_OCCURS: both values are salient in EVIDENCE_TEXT but no specific interaction is supported.\n"
        "- SYNERGY: one value is presented as reinforcing/supporting the other.\n"
        "- MEANS_END: one value is used as a means to achieve the other.\n"
        "- PREREQUISITE: one value is necessary/required for the other.\n"
        "- JUSTIFIES: one value is used as a normative reason to permit/legitimize the other.\n"
        "- CONFLICT: one value undermines/incompatible with the other.\n"
        "- TRADEOFF: explicit or strongly implied balancing/sacrifice between them.\n"
        "- CONSTRAINT: one value limits the scope/conditions of the other.\n"
        "- PRIORITIZES: an ordering/ranking is asserted or strongly implied.\n"
        "- RESOLVES_BY: a resolution is proposed using one value to address tension with another.\n"
        "- SUBSTITUTES: one value replaces the other (either-or framing).\n\n"

        "FIELD-SAFETY RULES (AVOID LABEL MIXUPS)\n"
        "- status MUST be one of: EXPRESSED, ACTIVATED, MENTIONED, RELATED, ABSENT.\n"
        "- relation MUST be one of: " + ", ".join(RELATION_LABELS) + ".\n"
        "- NEVER use a relation label as a status, and NEVER use a status label as a relation.\n\n"

        "OUTPUT FORMAT (STRICT)\n"
        "Return ONLY valid JSON (no markdown, no extra keys, no commentary).\n"
        "The JSON MUST match this schema exactly:\n"
        "{\n"
        '  \"values\": [\n'
        '    {\"value_item\": str, \"status\": one of [EXPRESSED,ACTIVATED,MENTIONED,RELATED,ABSENT]}\n'
        "  ],\n"
        '  \"edges\": [\n'
        "    {\n"
        '      \"source\": str,\n'
        '      \"target\": str,\n'
        f'      \"relation\": one of [{",".join(RELATION_LABELS)}],\n'
        '      \"direction\": one of [\"source_to_target\",\"target_to_source\",\"none\"],\n'
        '      \"polarity\": one of [\"supports\",\"opposes\",\"mixed\",\"neutral\"]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"

        "HARD OUTPUT RULES\n"
        "- Output MUST be a SINGLE JSON object and NOTHING ELSE.\n"
        "- Do NOT include any keys other than: values, edges\n"
        "- values MUST include ALL candidate VALUE_ITEMS exactly once each, using exact strings.\n"
        "- edges MUST use ONLY value_item strings from VALUE_ITEMS.\n"
    )


def build_user_prompt_extractor(target_sentence: str, evidence_text: str, value_items: List[str]) -> str:
    vi_block = "\n".join([f"- {v}" for v in value_items]) if value_items else "(none)"
    return (
        "TARGET_SENTENCE:\n<<<\n"
        f"{target_sentence}\n"
        ">>>\n\n"
        "EVIDENCE_TEXT:\n<<<\n"
        f"{evidence_text}\n"
        ">>>\n\n"
        "VALUE_ITEMS (candidates):\n"
        f"{vi_block}\n"
    )


def build_system_prompt_verifier() -> str:
    return (
        "You are a verifier/cleaner for a value-relation graph extracted from text.\n"
        "You will be given:\n"
        "- EVIDENCE_TEXT\n"
        "- CANDIDATE_GRAPH (values + edges)\n\n"
        "TASK\n"
        "- Keep an edge ONLY if the relation is explicitly stated in EVIDENCE_TEXT.\n"
        "- Drop any edge that is not explicitly supported.\n"
        "- Do NOT add new edges.\n"
        "- Do NOT change value statuses.\n\n"
        "OUTPUT FORMAT (STRICT)\n"
        "Return ONLY valid JSON with the same schema:\n"
        "{\n"
        '  "values": [...],\n'
        '  "edges": [...],\n'
        '  "notes": str\n'
        "}\n"
    )


def build_user_prompt_verifier(evidence_text: str, candidate_graph: Dict[str, Any]) -> str:
    return (
        "EVIDENCE_TEXT:\n<<<\n"
        f"{evidence_text}\n"
        ">>>\n\n"
        "CANDIDATE_GRAPH:\n"
        f"{json.dumps(candidate_graph, ensure_ascii=False)}\n"
    )


# ============================================================
# Utilities
# ============================================================

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def _safe_truncate(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return s
    if len(s) <= max_chars:
        return s
    return s[:max_chars]


def _extract_context_window(full_text: str, target_sentence: str, window_chars: int) -> str:
    if not full_text:
        return ""
    if window_chars <= 0:
        return str(full_text)

    ft = str(full_text)
    ts = str(target_sentence or "")
    if ts and ts in ft:
        pos = ft.find(ts)
        half = max(0, window_chars // 2)
        start = max(0, pos - half)
        end = min(len(ft), pos + len(ts) + half)
        return ft[start:end]
    return ft[:window_chars]


def _normalize_raw(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    s = raw.replace("\\r", "\r").replace("\\n", "\n").replace("\\t", "\t")
    s = re.sub(r"^\s*```.*?$", "", s, flags=re.MULTILINE)
    s = re.sub(r"\s*```$", "", s, flags=re.MULTILINE)
    return s.strip()


def _score_to_float(score: Any) -> Optional[float]:
    if score is None:
        return None
    try:
        return float(score)
    except Exception:
        return None


def _parse_value_items_with_scores(x: Any) -> List[Dict[str, Any]]:
    if x is None:
        return []
    if isinstance(x, list):
        out: List[Dict[str, Any]] = []
        for it in x:
            if isinstance(it, dict):
                vi = str(it.get("value_item", "") or "").strip()
                if not vi:
                    continue
                out.append({"value_item": vi, "score": it.get("score", None)})
            else:
                s = str(it).strip()
                if s:
                    out.append({"value_item": s, "score": None})
        return out
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            j = json.loads(s)
            return _parse_value_items_with_scores(j)
        except Exception:
            return [{"value_item": s, "score": None}]
    s = str(x).strip()
    return [{"value_item": s, "score": None}] if s else []


def _dedup_value_items_keep_max_score(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for it in items:
        vi = str(it.get("value_item", "") or "").strip()
        if not vi:
            continue
        sc = _score_to_float(it.get("score", None))
        if vi not in best:
            best[vi] = it
        else:
            prev_sc = _score_to_float(best[vi].get("score", None))
            if prev_sc is None or (sc is not None and sc > prev_sc):
                best[vi] = it
    return list(best.values())


def _top_k_value_items(items: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    if k <= 0:
        return []
    if len(items) <= k:
        return items

    scored: List[Tuple[int, float]] = []
    unscored: List[int] = []
    for i, it in enumerate(items):
        sc = _score_to_float(it.get("score", None))
        if sc is None:
            unscored.append(i)
        else:
            scored.append((i, sc))

    scored_sorted = sorted(scored, key=lambda t: t[1], reverse=True)
    picked_idx: List[int] = [i for (i, _) in scored_sorted[:k]]
    if len(picked_idx) < k:
        need = k - len(picked_idx)
        picked_idx.extend(unscored[:need])

    picked_set = set(picked_idx)
    return [it for j, it in enumerate(items) if j in picked_set][:k]


def _iter_jsonl(path: str, max_rows: int = 0):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        n = 0
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            yield line_idx, obj
            n += 1
            if max_rows and max_rows > 0 and n >= max_rows:
                break


def _count_jsonl_records(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


# ============================================================
# JSON parsing + minimal repair
# ============================================================

def _extract_first_json_object(text: str) -> Optional[str]:
    s = _normalize_raw(text)
    if not s:
        return None

    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
    return None


def _safe_json_load(raw: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    s = _normalize_raw(raw)
    if not s:
        return None, ["empty_output"]

    try:
        return json.loads(s), warnings
    except Exception:
        warnings.append("json_parse_failed")
        sub = _extract_first_json_object(s)
        if not sub:
            warnings.append("no_json_object_found")
            return None, warnings
        try:
            return json.loads(sub), warnings + ["extracted_first_json_object"]
        except Exception:
            warnings.append("json_parse_failed_on_extracted")
            return None, warnings


def _validate_graph_schema(graph: Dict[str, Any], allowed_value_items: List[str]) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    allowed_set = {v.strip() for v in allowed_value_items if v.strip()}

    out: Dict[str, Any] = {"values": [], "edges": [], "notes": ""}

    out["notes"] = graph.get("notes", "") if isinstance(graph.get("notes", ""), str) else ""

    # values
    vals = graph.get("values", [])
    if not isinstance(vals, list):
        warnings.append("values_not_list")
        vals = []

    seen = set()
    for v in vals:
        if not isinstance(v, dict):
            continue
        vi = str(v.get("value_item", "") or "").strip()
        if not vi:
            continue
        if vi not in allowed_set:
            warnings.append(f"value_item_not_allowed_in_values:{vi}")
            continue

        status = str(v.get("status", "") or "").strip().upper()
        if status not in VALUE_STATUS:
            warnings.append(f"bad_value_status:{vi}:{status}")
            status = "ABSENT"

        if vi in seen:
            continue
        seen.add(vi)
        out["values"].append({"value_item": vi, "status": status})

    # ensure all candidates included
    present = {x["value_item"] for x in out["values"]}
    for vi in allowed_value_items:
        if vi not in present:
            out["values"].append({"value_item": vi, "status": "ABSENT"})
            warnings.append(f"missing_value_filled_as_absent:{vi}")

    # edges
    edges = graph.get("edges", [])
    if not isinstance(edges, list):
        warnings.append("edges_not_list")
        edges = []

    for e in edges:
        if not isinstance(e, dict):
            continue
        src = str(e.get("source", "") or "").strip()
        tgt = str(e.get("target", "") or "").strip()
        if not src or not tgt:
            continue
        if src not in allowed_set or tgt not in allowed_set:
            warnings.append(f"edge_value_item_not_allowed:{src}->{tgt}")
            continue

        rel = str(e.get("relation", "") or "").strip().upper()
        if rel not in RELATION_LABELS:
            warnings.append(f"bad_relation_label:{rel}")
            continue

        direction = str(e.get("direction", "") or "").strip()
        if direction not in ["source_to_target", "target_to_source", "none"]:
            direction = "none"

        polarity = str(e.get("polarity", "") or "").strip()
        if polarity not in ["supports", "opposes", "mixed", "neutral"]:
            polarity = "neutral"

        out["edges"].append(
            {
                "source": src,
                "target": tgt,
                "relation": rel,
                "direction": direction,
                "polarity": polarity,
            }
        )

    # de-dup identical edges
    uniq = []
    seen_e = set()
    for e in out["edges"]:
        key = (e["source"], e["target"], e["relation"], e["direction"], e["polarity"])
        if key in seen_e:
            continue
        seen_e.add(key)
        uniq.append(e)
    out["edges"] = uniq

    return out, warnings


# ============================================================
# LLM wrappers
# ============================================================

class LLMRunner:
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        system_prompt: str,
        device: str = "cuda",
        dtype: str = "auto",
    ):
        from model import LLMModel
        self.model = LLMModel(
            model=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
            dtype=dtype,
            system_prompt=system_prompt,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.top_p = top_p

    def run_one(self, prompt: str) -> str:
        out = self.model([prompt])[0]
        return str(out) if out is not None else ""


def verify_graph_two(
    verifier: LLMRunner,
    evidence_text: str,
    candidate_graph: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    prompt = build_user_prompt_verifier(evidence_text, candidate_graph)
    raw = verifier.run_one(prompt)
    obj, warns = _safe_json_load(raw)
    return obj, {"raw": _normalize_raw(raw), "warnings": warns}


# ============================================================
# Evidence builder
# ============================================================

def build_evidence_text(
    target_sentence: str,
    full_text: str,
    use_context: bool,
    context_window_chars: int,
    evidence_max_chars: int,
) -> str:
    ts = str(target_sentence or "")
    if not use_context:
        return _safe_truncate(ts, evidence_max_chars)

    ctx = _extract_context_window(full_text or "", ts, context_window_chars)
    if ts and ts not in ctx:
        ctx = (ts + "\n" + ctx) if ctx else ts
    return _safe_truncate(ctx, evidence_max_chars)


# ============================================================
# CLI
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser("Extract value-value relations (KG-style) from JSONL (fast, minimal schema).")
    ap.add_argument("--input_jsonl", type=str, required=True)

    ap.add_argument("--output_jsonl_original", type=str, required=True)
    ap.add_argument("--output_jsonl", type=str, default=None)

    ap.add_argument("--idx_col", type=str, default="idx")
    ap.add_argument("--target_sentence_key", type=str, default="target_sentence")
    ap.add_argument("--full_text_key", type=str, default="full_text")
    ap.add_argument("--jsonl_value_items_key", type=str, default="value_items")

    ap.add_argument("--top_k_value_items", type=int, default=6)
    ap.add_argument("--max_rows", type=int, default=0)

    ap.add_argument("--save_every_rows", type=int, default=500)

    # global count tqdm
    ap.add_argument("--count_total_first", action="store_true", help="Pre-count rows for global tqdm total.")

    # context/evidence controls
    ap.add_argument("--use_context", action="store_true")
    ap.add_argument("--context_window_chars", type=int, default=1600)
    ap.add_argument("--evidence_max_chars", type=int, default=1800)
    ap.add_argument("--target_max_chars", type=int, default=600)

    # LLM configs
    ap.add_argument("--llm_model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=220)  # smaller for speed
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="auto")

    # verifier toggle (2nd call)
    ap.add_argument("--use_verifier", action="store_true")
    ap.add_argument("--verifier_max_new_tokens", type=int, default=180)

    # output toggles
    ap.add_argument("--save_raw", action="store_true")
    ap.add_argument("--keep_warnings", action="store_true")

    return ap.parse_args()


def main():
    args = parse_args()
    _ensure_dir(args.output_jsonl_original)
    if args.output_jsonl:
        _ensure_dir(args.output_jsonl)

    # ---- global count tqdm ----
    total_lines: Optional[int] = None
    if args.count_total_first:
        print("[info] Counting total JSONL records for global tqdm total...", flush=True)
        total_lines = _count_jsonl_records(args.input_jsonl)
        if args.max_rows and args.max_rows > 0:
            total_lines = min(total_lines, int(args.max_rows))
        print(f"[info] Total rows (approx): {total_lines}", flush=True)

    extractor = LLMRunner(
        model_name=args.llm_model,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        system_prompt=build_system_prompt_extractor(),
        device=args.device,
        dtype=args.dtype,
    )

    verifier: Optional[LLMRunner] = None
    if args.use_verifier:
        verifier = LLMRunner(
            model_name=args.llm_model,
            tensor_parallel_size=args.tensor_parallel_size,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=args.verifier_max_new_tokens,
            system_prompt=build_system_prompt_verifier(),
            device=args.device,
            dtype=args.dtype,
        )

    f_orig = open(args.output_jsonl_original, "w", encoding="utf-8")
    f_norm = open(args.output_jsonl, "w", encoding="utf-8") if args.output_jsonl else None

    save_every = max(1, int(args.save_every_rows))
    n_since_flush = 0

    try:
        global_pbar = tqdm(total=total_lines, desc="rows(total)", unit="rows")

        for line_idx, obj in _iter_jsonl(args.input_jsonl, max_rows=args.max_rows):
            idx = obj.get(args.idx_col, obj.get("idx", line_idx))

            target_sentence = str(obj.get(args.target_sentence_key, "") or "")
            full_text = str(obj.get(args.full_text_key, "") or "")

            value_items_raw = obj.get(args.jsonl_value_items_key, [])
            vis = _parse_value_items_with_scores(value_items_raw)
            vis = _dedup_value_items_keep_max_score(vis)
            vis = _top_k_value_items(vis, int(args.top_k_value_items))
            value_items = [
                str(it.get("value_item", "") or "").strip()
                for it in vis
                if str(it.get("value_item", "") or "").strip()
            ]

            # If no candidates, still produce empty graph quickly (no LLM call)
            evidence_text = build_evidence_text(
                target_sentence=target_sentence,
                full_text=full_text,
                use_context=bool(args.use_context),
                context_window_chars=int(args.context_window_chars),
                evidence_max_chars=int(args.evidence_max_chars),
            )
            ts_trunc = _safe_truncate(target_sentence, int(args.target_max_chars))

            if not value_items:
                graph = {"values": [], "edges": [], "notes": "no_value_items"}
                schema_warns = []
                debug_extractor = {"raw": "", "warnings": ["skipped_no_value_items"]}
                debug_verifier = None
            else:
                prompt = build_user_prompt_extractor(ts_trunc, evidence_text, value_items)
                raw1 = extractor.run_one(prompt)
                parsed1, parse_warns1 = _safe_json_load(raw1)
                debug_extractor = {"raw": _normalize_raw(raw1), "warnings": parse_warns1}

                if parsed1 is None:
                    graph = {
                        "values": [{"value_item": vi, "status": "ABSENT"} for vi in value_items],
                        "edges": [],
                        "notes": "extraction_failed",
                    }
                    schema_warns = ["extractor_returned_no_json"]
                    debug_verifier = None
                else:
                    graph, schema_warns = _validate_graph_schema(parsed1, value_items)
                    debug_verifier = None

                    if verifier is not None:
                        obj2, dbg2 = verify_graph_two(verifier, evidence_text, graph)
                        debug_verifier = dbg2
                        if obj2 is not None:
                            graph2, schema_warns2 = _validate_graph_schema(obj2, value_items)
                            graph = graph2
                            schema_warns.extend(schema_warns2)
                        else:
                            schema_warns.append("verifier_failed_no_json_used_extractor_graph")

            out_obj = dict(obj)
            out_obj["value_relation_graph"] = {
                "scope": {
                    "target_sentence": ts_trunc,
                    "use_context": bool(args.use_context),
                    "context_window_chars": int(args.context_window_chars),
                },
                "values": graph.get("values", []),
                "edges": graph.get("edges", []),
                "notes": graph.get("notes", ""),
                "meta": {
                    "relation_labels": RELATION_LABELS,
                    "value_status_labels": VALUE_STATUS,
                    "llm_calls": (2 if verifier is not None else 1) if value_items else 0,
                },
            }

            if args.keep_warnings:
                out_obj["value_relation_graph"]["warnings"] = schema_warns
                out_obj["value_relation_graph"]["parse_warnings"] = {
                    "extractor": debug_extractor.get("warnings", []),
                    "verifier": (debug_verifier.get("warnings", []) if debug_verifier else []),
                }

            if args.save_raw:
                out_obj["value_relation_graph_raw"] = {
                    "extractor": debug_extractor.get("raw", ""),
                    "verifier": (debug_verifier.get("raw", "") if debug_verifier else ""),
                }

            f_orig.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

            if f_norm is not None:
                norm = {
                    args.idx_col: idx,
                    "target_sentence": ts_trunc,
                    "value_items_used": value_items,
                    "n_edges": len(graph.get("edges", [])),
                    "value_relation_graph": graph,
                }
                f_norm.write(json.dumps(norm, ensure_ascii=False) + "\n")

            n_since_flush += 1
            if n_since_flush >= save_every:
                f_orig.flush()
                if f_norm is not None:
                    f_norm.flush()
                n_since_flush = 0

            global_pbar.update(1)

        global_pbar.close()

    finally:
        f_orig.close()
        if f_norm is not None:
            f_norm.close()

    print("[Done]", flush=True)
    print(f"- Saved: {args.output_jsonl_original}", flush=True)
    if args.output_jsonl:
        print(f"- Saved: {args.output_jsonl}", flush=True)


if __name__ == "__main__":
    main()
