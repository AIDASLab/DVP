#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_context_states_from_factors.py

UPDATED (two-stage, new JSONL input) + SINGLE LLM INSTANCE + COUNT + CHUNKED STREAMING + PERIODIC FLUSH
------------------------------------------------------------------------------------------------------
- Supports value-relations JSONL input where value items + ABSENT status are in:
    obj["value_relation_graph"]["values"] = [{"value_item": ..., "status": ...}, ...]
- Only consider value items whose status != "ABSENT"
- Two-stage extraction:
  (1) Activate factors: ask LLM which factors are inferable/activated from text
  (2) For each activated factor: ask exact state
- Uses TARGET_TEXT + optional CONTEXT_TEXT (full/window text) for extraction

IMPORTANT CHANGE (memory / speed):
- Uses ONE underlying model instance (e.g., ONE vLLM engine).
- Switches system prompt between stages by mutating: llm.model.model.system_prompt
- Processes INPUT IN CHUNKS (row_chunk_size) so you can stop mid-run without losing everything,
  while still keeping BATCHED LLM calls (keeps vLLM throughput).

NEW (progress + robustness):
- Counts target rows for tqdm total (JSONL / sentences JSONL by counting non-empty lines; CSV via len()).
- Writes JSONL outputs incrementally (append-per-row, flush every K rows).
- Optional --resume for JSONL mode: skips rows whose idx already exists in output JSONL.

Legacy modes (CSV / sentences JSONL) are preserved.
"""

from __future__ import annotations

import os
import re
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple, Iterable, Set

import pandas as pd
from tqdm import tqdm

# -------------------------------
# Imports from your codebase
# -------------------------------
# The inference backend is imported when the shared model is constructed.


# ============================================================
# Prompting
# ============================================================

def build_system_prompt_state(include_evidence: bool) -> str:
    if include_evidence:
        return (
            "You are a strict extractor for CONTEXT STATES in value profiling.\n"
            "You will be given:\n"
            "- TARGET_TEXT (a single sentence/segment)\n"
            "- CONTEXT_TEXT (optional wider context)\n"
            "- VALUE_ITEM (string)\n"
            "- ONE CONTEXT FACTOR with: group, factor name, definition, allowed states\n\n"
            "TASK\n"
            "Select the SINGLE best matching STATE from the allowed states.\n"
            "If the state cannot be inferred from the text, output STATE: NONE.\n\n"
            "RULES (HIGH PRECISION)\n"
            "- Prefer TARGET_TEXT. Use CONTEXT_TEXT only to confirm/clarify, not to invent.\n"
            "- Do NOT guess. If ambiguous or not mentioned -> NONE.\n"
            "- Do NOT output a state that is not in the provided allowed states.\n"
            "- Evidence must be a short quote (<= 120 chars) copied from TARGET_TEXT or CONTEXT_TEXT, or NONE.\n\n"
            "OUTPUT FORMAT (STRICT)\n"
            "Exactly 5 lines, no extra text:\n"
            "VALUE_ITEM: <value item>\n"
            "FACTOR: <factor>\n"
            "STATE: <one of allowed states OR NONE>\n"
            "EVIDENCE: <quote OR NONE>\n"
            "CONFIDENCE: <integer 0..100>\n"
        )
    else:
        return (
            "You are a strict extractor for CONTEXT STATES in value profiling.\n"
            "You will be given:\n"
            "- TARGET_TEXT (a single sentence/segment)\n"
            "- CONTEXT_TEXT (optional wider context)\n"
            "- VALUE_ITEM (string)\n"
            "- ONE CONTEXT FACTOR with: group, factor name, definition, allowed states\n\n"
            "TASK\n"
            "Select the SINGLE best matching STATE from the allowed states.\n"
            "If the state cannot be inferred from the text, output STATE: NONE.\n\n"
            "RULES (HIGH PRECISION)\n"
            "- Prefer TARGET_TEXT. Use CONTEXT_TEXT only to confirm/clarify, not to invent.\n"
            "- Do NOT guess. If ambiguous or not mentioned -> NONE.\n"
            "- Do NOT output a state that is not in the provided allowed states.\n\n"
            "OUTPUT FORMAT (STRICT)\n"
            "Exactly 4 lines, no extra text:\n"
            "VALUE_ITEM: <value item>\n"
            "FACTOR: <factor>\n"
            "STATE: <one of allowed states OR NONE>\n"
            "CONFIDENCE: <integer 0..100>\n"
        )


def build_system_prompt_activate() -> str:
    return (
        "You are a strict selector for ACTIVATED CONTEXT FACTORS in value profiling.\n"
        "You will be given:\n"
        "- VALUE_ITEM (string)\n"
        "- TARGET_TEXT (a single sentence/segment)\n"
        "- CONTEXT_TEXT (optional wider context)\n"
        "- A LIST of candidate FACTORS, each with a short definition\n\n"
        "TASK\n"
        "Select ONLY the factors that are inferable from the text.\n"
        "If none are inferable, output NONE.\n\n"
        "RULES (HIGH PRECISION)\n"
        "- Prefer TARGET_TEXT. Use CONTEXT_TEXT only to confirm/clarify.\n"
        "- Do NOT guess.\n"
        "- Only select factor names that appear in the provided candidate list.\n\n"
        "OUTPUT FORMAT (STRICT)\n"
        "Exactly 3 lines, no extra text:\n"
        "VALUE_ITEM: <value item>\n"
        "ACTIVATED_FACTORS: <factor1 | factor2 | ... OR NONE>\n"
        "CONFIDENCE: <integer 0..100>\n"
    )


USER_TEMPLATE_STATE = (
    "VALUE_ITEM: {value_item}\n"
    "FACTOR_GROUP: {group}\n"
    "FACTOR: {factor}\n"
    "DEFINITION: {definition}\n"
    "ALLOWED_STATES: {allowed_states}\n\n"
    "TARGET_TEXT:\n"
    "<<<\n"
    "{target_text}\n"
    ">>>\n\n"
    "CONTEXT_TEXT:\n"
    "<<<\n"
    "{context_text}\n"
    ">>>\n"
)

USER_TEMPLATE_ACTIVATE = (
    "VALUE_ITEM: {value_item}\n\n"
    "CANDIDATE_FACTORS (name :: definition):\n"
    "{factor_block}\n\n"
    "TARGET_TEXT:\n"
    "<<<\n"
    "{target_text}\n"
    ">>>\n\n"
    "CONTEXT_TEXT:\n"
    "<<<\n"
    "{context_text}\n"
    ">>>\n"
)


# ============================================================
# Parsing helpers
# ============================================================

_RE_VI = re.compile(r"^\s*VALUE_ITEM\s*:\s*(.+?)\s*$", re.IGNORECASE)
_RE_FAC = re.compile(r"^\s*FACTOR\s*:\s*([a-z0-9_]+)\s*$", re.IGNORECASE)
_RE_STATE = re.compile(r"^\s*STATE\s*:\s*(.+?)\s*$", re.IGNORECASE)
_RE_EVI = re.compile(r"^\s*EVIDENCE\s*:\s*(.+?)\s*$", re.IGNORECASE)
_RE_CONF = re.compile(r"^\s*CONFIDENCE\s*:\s*(\d{1,3})\s*$", re.IGNORECASE)
_RE_ACT_FACTORS = re.compile(r"^\s*ACTIVATED_FACTORS\s*:\s*(.+?)\s*$", re.IGNORECASE)


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def _normalize_raw(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    s = raw.replace("\\r", "\r").replace("\\n", "\n").replace("\\t", "\t")
    s = re.sub(r"^\s*```.*?$", "", s, flags=re.MULTILINE)
    return s.strip()


def _clip_int(x: Any, lo: int, hi: int, default: int = 0) -> int:
    try:
        v = int(x)
    except Exception:
        return default
    return max(lo, min(hi, v))


def _parse_state_output(
    raw: str,
    expected_value_item: str,
    expected_factor: str,
    allowed_states: List[str],
    include_evidence: bool,
) -> Tuple[Dict[str, Any], List[str]]:
    s = _normalize_raw(raw)
    warnings: List[str] = []

    if not s:
        parsed = {"selected_state": "NONE", "confidence": 0}
        if include_evidence:
            parsed["evidence"] = "NONE"
        return parsed, ["empty_output"]

    vi = fac = state = evi = conf = None
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    for ln in lines:
        m = _RE_VI.match(ln)
        if m:
            vi = m.group(1).strip()
        m = _RE_FAC.match(ln)
        if m:
            fac = m.group(1).strip()
        m = _RE_STATE.match(ln)
        if m:
            state = m.group(1).strip()
        if include_evidence:
            m = _RE_EVI.match(ln)
            if m:
                evi = m.group(1).strip()
        m = _RE_CONF.match(ln)
        if m:
            conf = m.group(1).strip()

    if vi and vi.strip().lower() != (expected_value_item or "").strip().lower():
        warnings.append(f"value_item_mismatch: got='{vi}' expected='{expected_value_item}'")
    if fac and fac.strip().lower() != (expected_factor or "").strip().lower():
        warnings.append(f"factor_mismatch: got='{fac}' expected='{expected_factor}'")

    allowed_set = {x.strip().lower() for x in allowed_states}
    if state is None:
        warnings.append("missing_state")
        state_norm = "none"
    else:
        state_norm = state.strip()

    if state_norm.strip().lower() == "none":
        selected = "NONE"
    else:
        if state_norm.strip().lower() not in allowed_set:
            warnings.append(f"state_not_allowed: '{state_norm}'")
            selected = "NONE"
        else:
            low = state_norm.strip().lower()
            selected = next((x for x in allowed_states if x.strip().lower() == low), state_norm)

    parsed: Dict[str, Any] = {"selected_state": selected}

    if include_evidence:
        evidence = (evi or "").strip()
        if not evidence:
            evidence = "NONE"
            warnings.append("missing_evidence")
        parsed["evidence"] = evidence

    confidence = _clip_int(conf, 0, 100, default=0)
    if conf is None:
        warnings.append("missing_confidence")
    parsed["confidence"] = confidence

    return parsed, warnings


def _parse_activate_output(
    raw: str,
    expected_value_item: str,
    allowed_factors: List[str],
) -> Tuple[List[str], int, List[str]]:
    s = _normalize_raw(raw)
    warnings: List[str] = []

    if not s:
        return [], 0, ["empty_output"]

    vi = af = conf = None
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    for ln in lines:
        m = _RE_VI.match(ln)
        if m:
            vi = m.group(1).strip()
        m = _RE_ACT_FACTORS.match(ln)
        if m:
            af = m.group(1).strip()
        m = _RE_CONF.match(ln)
        if m:
            conf = m.group(1).strip()

    if vi and vi.strip().lower() != (expected_value_item or "").strip().lower():
        warnings.append(f"value_item_mismatch: got='{vi}' expected='{expected_value_item}'")

    confidence = _clip_int(conf, 0, 100, default=0)
    if conf is None:
        warnings.append("missing_confidence")

    if af is None:
        warnings.append("missing_activated_factors")
        return [], confidence, warnings

    af_norm = af.strip()
    if af_norm.lower() == "none":
        return [], confidence, warnings

    candidates = [x.strip() for x in af_norm.split("|") if x.strip()]
    allowed_set = {x.strip().lower() for x in allowed_factors}
    kept: List[str] = []
    for c in candidates:
        if c.strip().lower() in allowed_set:
            kept.append(next((x for x in allowed_factors if x.strip().lower() == c.strip().lower()), c))
        else:
            warnings.append(f"factor_not_allowed: '{c}'")

    seen = set()
    out = []
    for x in kept:
        k = x.strip().lower()
        if k not in seen:
            out.append(x)
            seen.add(k)

    return out, confidence, warnings


# ============================================================
# SINGLE LLM wrapper (switch system prompt per stage)
# ============================================================

class SingleLLM:
    """
    Holds exactly ONE underlying model instance (e.g., one vLLM engine).
    Switches system_prompt by mutating:
      llm.model.model.system_prompt
    """
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        include_evidence: bool,
        device: str = "cuda",
        dtype: str = "auto",
        model_dir: Optional[str] = None,
    ):
        self.system_prompt_activate = build_system_prompt_activate()
        self.system_prompt_state = build_system_prompt_state(include_evidence=include_evidence)

        from model import LLMModel
        self.model = LLMModel(
            model=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
            dtype=dtype,
            model_dir=model_dir,
            system_prompt=self.system_prompt_state,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.top_p = top_p

    def set_stage(self, stage: str) -> None:
        if stage == "activate":
            self.model.model.system_prompt = self.system_prompt_activate
        elif stage == "state":
            self.model.model.system_prompt = self.system_prompt_state
        else:
            raise ValueError(f"unknown stage: {stage}")

    def run(self, prompts: List[str], *, max_tokens: Optional[int] = None) -> List[str]:
        kwargs: Dict[str, Any] = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = int(max_tokens)
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        outs = self.model(prompts, **kwargs)
        return [str(o) if o is not None else "" for o in outs]


# ============================================================
# Value-items parsing
# ============================================================

def _parse_value_item_cell(x: Any) -> List[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    s = str(x).strip()
    if not s:
        return []
    try:
        j = json.loads(s)
        if isinstance(j, list):
            return [str(v).strip() for v in j if str(v).strip()]
        if isinstance(j, str) and j.strip():
            return [j.strip()]
    except Exception:
        pass
    return [s]


def _parse_value_items_field(value_items_field: Any) -> List[str]:
    if value_items_field is None:
        return []
    if isinstance(value_items_field, list):
        out: List[str] = []
        for v in value_items_field:
            if isinstance(v, dict):
                vi = str(v.get("value_item", "") or "").strip()
                if vi:
                    out.append(vi)
            else:
                s = str(v).strip()
                if s:
                    out.append(s)
        return out
    return _parse_value_item_cell(value_items_field)


def _value_items_from_value_relation_graph(obj: Dict[str, Any]) -> Optional[List[str]]:
    vrg = obj.get("value_relation_graph")
    if not isinstance(vrg, dict):
        return None
    vals = vrg.get("values")
    if not isinstance(vals, list):
        return None

    out: List[str] = []
    for v in vals:
        if not isinstance(v, dict):
            continue
        vi = str(v.get("value_item", "") or "").strip()
        st = str(v.get("status", "") or "").strip()
        if not vi:
            continue
        if st.upper() == "ABSENT":
            continue
        out.append(vi)

    seen = set()
    dedup = []
    for x in out:
        k = x.lower()
        if k not in seen:
            dedup.append(x)
            seen.add(k)
    return dedup


def _infer_context_text(obj: Dict[str, Any], full_text: str) -> str:
    vrg = obj.get("value_relation_graph")
    if isinstance(vrg, dict):
        scope = vrg.get("scope")
        if isinstance(scope, dict):
            use_context = scope.get("use_context", True)
            if use_context is False:
                return ""
            return (full_text or "")[:2000]
    return (full_text or "")[:2000]


# ============================================================
# Input loading (generators)
# ============================================================

def _iter_rows_from_csv_mode(args) -> Iterable[Dict[str, Any]]:
    df_t = pd.read_csv(args.target_csv)
    df_f = pd.read_csv(args.full_csv)

    for col in [args.target_original_idx_col, args.target_text_col, args.target_value_items_col]:
        if col not in df_t.columns:
            raise ValueError(f"target_csv missing column '{col}'. columns={list(df_t.columns)}")
    if args.full_text_col not in df_f.columns:
        raise ValueError(f"full_csv missing text column '{args.full_text_col}'. columns={list(df_f.columns)}")

    if args.full_idx_col == "__row_index__":
        full_lookup = {int(i): str(df_f.loc[i, args.full_text_col] or "") for i in range(len(df_f))}
    else:
        if args.full_idx_col not in df_f.columns:
            raise ValueError(f"full_csv missing idx column '{args.full_idx_col}'. columns={list(df_f.columns)}")
        full_lookup: Dict[int, str] = {}
        for _, row in df_f.iterrows():
            try:
                k = int(row[args.full_idx_col])
            except Exception:
                continue
            full_lookup[k] = str(row.get(args.full_text_col, "") or "")

    if args.max_rows and args.max_rows > 0:
        df_t = df_t.head(args.max_rows).copy()

    for ridx, row in df_t.iterrows():
        try:
            orig_idx = int(row[args.target_original_idx_col])
        except Exception:
            orig_idx = None

        idx_val = int(row.get(args.target_idx_col, ridx)) if args.target_idx_col in df_t.columns else int(ridx)
        target_text = str(row.get(args.target_text_col, "") or "")
        value_items = _parse_value_item_cell(row.get(args.target_value_items_col))

        full_text = ""
        if orig_idx is not None and orig_idx in full_lookup:
            full_text = str(full_lookup[orig_idx] or "")

        yield {
            "_row_key": int(ridx),
            "idx": idx_val,
            "original_text_idx": orig_idx,
            "target_text": target_text,
            "value_items": value_items,
            "full_text": full_text,
            "_obj": {"full_text": full_text},
            "_extra": {},
        }


def _iter_rows_from_jsonl_mode(args) -> Iterable[Dict[str, Any]]:
    path = args.input_jsonl
    if not path:
        raise ValueError("input_jsonl is required for JSONL mode.")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if args.max_rows and args.max_rows > 0 and n >= args.max_rows:
                break

            raw_idx = obj.get(args.target_idx_col, obj.get("idx", line_idx))
            try:
                idx_val = int(raw_idx)
            except Exception:
                idx_val = int(line_idx)

            full_text = str(obj.get("full_text", "") or "")
            target_field = args.target_text_col or "target_sentence"
            target_text = str(obj.get(target_field, obj.get("target_sentence", "")) or "")

            vis = _value_items_from_value_relation_graph(obj)
            if vis is None:
                vi_field = obj.get("value_items", None)
                if vi_field is None and args.target_value_items_col:
                    vi_field = obj.get(args.target_value_items_col, None)
                vis = _parse_value_items_field(vi_field)

            original_text_idx = idx_val

            yield {
                "_row_key": int(line_idx),
                "idx": idx_val,
                "original_text_idx": original_text_idx,
                "target_text": target_text,
                "value_items": vis,
                "full_text": full_text,
                "_obj": obj,
                "_extra": {k: v for k, v in obj.items()
                           if k not in {"full_text", "value_items", "target_sentence", target_field}},
            }
            n += 1


def _iter_rows_from_sentences_jsonl_mode(args) -> Iterable[Dict[str, Any]]:
    path = args.sentences_jsonl
    if not path:
        raise ValueError("sentences_jsonl is required for SENTENCE JSONL mode.")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if args.max_rows and args.max_rows > 0 and n >= args.max_rows:
                break

            vi = str(obj.get("value_item", "") or "").strip()
            sent = str(obj.get("sentence", "") or "").strip()
            if not vi or not sent:
                continue

            raw_idx = obj.get(args.target_idx_col, obj.get("idx", line_idx))
            try:
                idx_val = int(raw_idx)
            except Exception:
                idx_val = int(line_idx)

            yield {
                "_row_key": int(line_idx),
                "idx": idx_val,
                "original_text_idx": idx_val,
                "target_text": sent,
                "value_items": [vi],
                "full_text": "",
                "_obj": obj,
                "_extra": {k: v for k, v in obj.items() if k not in {"value_item", "sentence"}},
            }
            n += 1


# ============================================================
# Counting helpers (for tqdm total)
# ============================================================

def _count_nonempty_jsonl_lines(path: str, *, max_rows: int = 0) -> int:
    cnt = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cnt += 1
                if max_rows and max_rows > 0 and cnt >= max_rows:
                    break
    return cnt


def _load_done_ids_from_output_jsonl(path: str, *, id_key: str = "idx") -> Set[int]:
    done: Set[int] = set()
    if not path or (not os.path.exists(path)):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            v = obj.get(id_key, None)
            try:
                done.add(int(v))
            except Exception:
                continue
    return done


# ============================================================
# CLI
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser("Extract per-factor context states for each (text,value_item) using factors ontology.")

    in_grp = ap.add_mutually_exclusive_group(required=True)
    in_grp.add_argument("--input_jsonl", type=str, default=None,
                        help="Mode B: JSONL where each line contains full_text, target_sentence, value_items OR value_relation_graph.")
    in_grp.add_argument("--target_csv", type=str, default=None,
                        help="Mode A: legacy target CSV containing (original_text_idx, text, value_item, ...).")
    in_grp.add_argument("--sentences_jsonl", type=str, default=None,
                        help="Mode C: sentence JSONL with {value_item, sentence, ...}.")

    ap.add_argument("--full_csv", type=str, default=None, help="Mode A only. Required when using --target_csv.")

    ap.add_argument("--factors_json", type=str, required=True)
    ap.add_argument("--output_path", type=str, required=True, help=".jsonl recommended; .json also supported")

    ap.add_argument("--target_idx_col", type=str, default="idx")
    ap.add_argument("--target_original_idx_col", type=str, default="original_text_idx")

    ap.add_argument("--target_text_col", type=str, default="target_sentence",
                    help="Mode A: usually 'text'. Mode B: usually 'target_sentence'. Mode C: ignored (always uses 'sentence').")
    ap.add_argument("--target_value_items_col", type=str, default="value_item",
                    help="Mode A: column containing JSON-string list of value items. Mode B: fallback key if 'value_items' missing.")

    ap.add_argument("--full_idx_col", type=str, default="__row_index__",
                    help="Mode A: Column for original_text_idx lookup. Use '__row_index__' to use full_csv row index.")
    ap.add_argument("--full_text_col", type=str, default="text")

    ap.add_argument("--max_rows", type=int, default=0, help="0=all")
    ap.add_argument("--max_factors_per_item", type=int, default=0, help="0=all factors")

    ap.add_argument("--llm_model", type=str, default="Qwen/Qwen3-32B")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=200)  # stage2; stage1 fixed at 180
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="auto")
    ap.add_argument("--model_dir", type=str, default=None)

    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--save_raw", action="store_true")
    ap.add_argument("--keep_warnings", action="store_true")
    ap.add_argument("--keep_full_text", action="store_true")
    ap.add_argument("--include_evidence", action="store_true")

    ap.add_argument("--output_format", type=str, default="jsonl", choices=["jsonl", "json"])

    ap.add_argument("--two_stage", action="store_true")
    ap.add_argument("--one_stage", action="store_true")

    ap.add_argument("--save_normalized", action="store_true",
                    help="Also write <output_path>.normalized.jsonl with compact factor->state mapping.")

    ap.add_argument("--save_every_rows", type=int, default=5000,
                    help="Flush outputs every K completed rows (JSONL flush; JSON snapshot write). 0 disables.")

    ap.add_argument("--row_chunk_size", type=int, default=2000,
                    help="Process rows in chunks to allow safe interruption while keeping batched LLM calls.")

    ap.add_argument("--resume", action="store_true",
                    help="JSONL only: if output_path exists, skip rows whose idx already appears in it, and append new rows.")

    return ap.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    _ensure_dir(args.output_path)

    if args.target_csv is not None and not args.full_csv:
        raise ValueError("CSV mode requires --full_csv.")

    # stage mode
    use_two_stage = True
    if args.one_stage:
        use_two_stage = False
    if args.two_stage:
        use_two_stage = True

    # load factors
    with open(args.factors_json, "r", encoding="utf-8") as f:
        factors_data = json.load(f)

    items = factors_data.get("items", {})
    if not isinstance(items, dict):
        raise ValueError("factors_json must have top-level key 'items' as a dict.")

    def _get_factors_for_value_item(vi: str) -> List[Dict[str, Any]]:
        vi_entry = items.get(vi)
        if not vi_entry or not isinstance(vi_entry, dict):
            return []
        factors = vi_entry.get("factors", [])
        if not isinstance(factors, list):
            return []
        if args.max_factors_per_item and args.max_factors_per_item > 0:
            factors = factors[:args.max_factors_per_item]
        cleaned: List[Dict[str, Any]] = []
        for fac_obj in factors:
            group = str(fac_obj.get("group", "") or "")
            factor = str(fac_obj.get("factor", "") or "")
            definition = str(fac_obj.get("definition", "") or "")
            allowed_states = fac_obj.get("states", [])
            if not factor or not isinstance(allowed_states, list) or len(allowed_states) == 0:
                continue
            cleaned.append({
                "group": group,
                "factor": factor,
                "definition": definition,
                "allowed_states": [str(s) for s in allowed_states],
            })
        return cleaned

    # total row count for tqdm
    total_rows: Optional[int] = None
    if args.input_jsonl:
        total_rows = _count_nonempty_jsonl_lines(args.input_jsonl, max_rows=args.max_rows)
    elif args.sentences_jsonl:
        total_rows = _count_nonempty_jsonl_lines(args.sentences_jsonl, max_rows=args.max_rows)
    elif args.target_csv:
        df_t = pd.read_csv(args.target_csv)
        total_rows = int(min(len(df_t), args.max_rows)) if args.max_rows and args.max_rows > 0 else int(len(df_t))

    # build row iterator
    if args.sentences_jsonl:
        row_iter = _iter_rows_from_sentences_jsonl_mode(args)
    elif args.input_jsonl:
        row_iter = _iter_rows_from_jsonl_mode(args)
    else:
        row_iter = _iter_rows_from_csv_mode(args)

    # model (single instance)
    llm = SingleLLM(
        model_name=args.llm_model,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        include_evidence=args.include_evidence,
        device=args.device,
        dtype=args.dtype,
        model_dir=args.model_dir,
    )

    # output writers
    as_jsonl = (args.output_format == "jsonl") or args.output_path.lower().endswith(".jsonl")
    if not as_jsonl and args.resume:
        raise ValueError("--resume is supported only for JSONL output format.")

    norm_path = args.output_path + ".normalized.jsonl"
    want_norm = bool(args.save_normalized)

    done_ids: Set[int] = set()
    if as_jsonl and args.resume and os.path.exists(args.output_path):
        done_ids = _load_done_ids_from_output_jsonl(args.output_path, id_key="idx")

    out_mode = "a" if (as_jsonl and args.resume and os.path.exists(args.output_path)) else "w"
    f_out = open(args.output_path, out_mode, encoding="utf-8") if as_jsonl else None
    f_norm = None
    if as_jsonl and want_norm:
        norm_mode = "a" if (args.resume and os.path.exists(norm_path)) else "w"
        f_norm = open(norm_path, norm_mode, encoding="utf-8")

    # counters
    llm_calls_activate = 0
    llm_calls_state = 0
    written_rows = 0
    processed_rows = 0
    skipped_rows = 0
    save_every = int(args.save_every_rows or 0)
    row_chunk_size = max(1, int(args.row_chunk_size or 1))
    bs = max(1, int(args.batch_size or 1))

    def _make_norm_record(rec: Dict[str, Any]) -> Dict[str, Any]:
        compact: Dict[str, Any] = {
            "idx": rec.get("idx"),
            "original_text_idx": rec.get("original_text_idx"),
            "target_text": rec.get("target_text"),
            "value_items": rec.get("value_items"),
            "context_states": {},
        }
        if "meta" in rec:
            compact["meta"] = rec["meta"]

        for vi, entries in (rec.get("context_states", {}) or {}).items():
            fac_map: Dict[str, Any] = {}
            for e in entries:
                fac = e.get("factor")
                if not fac:
                    continue
                fac_map[fac] = {
                    "selected_state": e.get("selected_state", "NONE"),
                    "confidence": e.get("confidence", 0),
                }
                if args.include_evidence:
                    fac_map[fac]["evidence"] = e.get("evidence", "NONE")
            compact["context_states"][vi] = fac_map
        return compact

    def _write_record_jsonl(rec: Dict[str, Any]) -> None:
        nonlocal written_rows
        assert f_out is not None
        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if f_norm is not None:
            f_norm.write(json.dumps(_make_norm_record(rec), ensure_ascii=False) + "\n")
        written_rows += 1
        if save_every > 0 and (written_rows % save_every == 0):
            f_out.flush()
            if f_norm is not None:
                f_norm.flush()

    # JSON (non-JSONL) fallback: keep in memory + periodic snapshots
    all_records: List[Dict[str, Any]] = []
    def _write_record_json(rec: Dict[str, Any]) -> None:
        nonlocal written_rows
        all_records.append(rec)
        written_rows += 1
        if save_every > 0 and (written_rows % save_every == 0):
            tmp_path = args.output_path + ".partial.json"
            with open(tmp_path, "w", encoding="utf-8") as wf:
                json.dump(all_records, wf, ensure_ascii=False, indent=2)

    # ============================================================
    # Chunk processing
    # ============================================================

    pbar = tqdm(total=total_rows, desc="rows", dynamic_ncols=True)

    pending: List[Dict[str, Any]] = []

    def process_chunk(chunk_rows: List[Dict[str, Any]]) -> None:
        nonlocal llm_calls_activate, llm_calls_state, processed_rows, skipped_rows

        # Build per-row records for this chunk
        by_row: Dict[int, Dict[str, Any]] = {}
        row_objs: Dict[int, Dict[str, Any]] = {}

        for rm in chunk_rows:
            idx_val = rm.get("idx", None)
            try:
                idx_int = int(idx_val)
            except Exception:
                idx_int = None

            # resume skip (JSONL only)
            if as_jsonl and args.resume and (idx_int is not None) and (idx_int in done_ids):
                skipped_rows += 1
                processed_rows += 1
                pbar.update(1)
                continue

            rec: Dict[str, Any] = {
                "idx": rm["idx"],
                "original_text_idx": rm["original_text_idx"],
                "target_text": rm["target_text"],
                "value_items": rm["value_items"],
                "context_states": {},
            }
            extra = rm.get("_extra", {}) or {}
            if extra:
                rec["meta"] = extra
            if args.keep_full_text and rm.get("full_text"):
                rec["full_text"] = rm["full_text"]
            if args.save_raw:
                rec["raw_model_outputs"] = {"activate": {}, "state": {}}

            by_row[rm["_row_key"]] = rec
            row_objs[rm["_row_key"]] = rm.get("_obj", {}) or {}

        if not by_row:
            return

        # -------------------------
        # Stage 1: activate factors
        # -------------------------
        if use_two_stage:
            activate_jobs: List[Dict[str, Any]] = []
            for row_key, rec in by_row.items():
                obj = row_objs.get(row_key, {})
                full_text = str(obj.get("full_text", "") or "")
                target_text = str(rec.get("target_text", "") or "")
                context_text = _infer_context_text(obj, full_text)

                for vi in rec.get("value_items", []) or []:
                    facs = _get_factors_for_value_item(vi)
                    if not facs:
                        continue

                    factor_block = "\n".join([f"- {fo['factor']} :: {fo['definition']}" for fo in facs])
                    prompt = USER_TEMPLATE_ACTIVATE.format(
                        value_item=vi,
                        factor_block=factor_block[:6000],
                        target_text=target_text[:1500],
                        context_text=context_text[:2000],
                    )
                    activate_jobs.append({
                        "_row_key": row_key,
                        "value_item": vi,
                        "allowed_factors": [fo["factor"] for fo in facs],
                        "prompt": prompt,
                        "_factors": facs,
                    })

            if activate_jobs:
                llm.set_stage("activate")
                for start in range(0, len(activate_jobs), bs):
                    chunk = activate_jobs[start:start + bs]
                    prompts = [j["prompt"] for j in chunk]
                    raws = llm.run(prompts, max_tokens=180)
                    llm_calls_activate += len(prompts)

                    for job, raw in zip(chunk, raws):
                        row_key = job["_row_key"]
                        vi = job["value_item"]
                        allowed_factors = job["allowed_factors"]

                        activated, conf, warns = _parse_activate_output(
                            raw=raw,
                            expected_value_item=vi,
                            allowed_factors=allowed_factors,
                        )

                        by_row[row_key].setdefault("_activated_factors", {})[vi] = activated
                        by_row[row_key].setdefault("_activate_confidence", {})[vi] = conf
                        by_row[row_key].setdefault("_factors_by_vi", {})[vi] = {fo["factor"]: fo for fo in job["_factors"]}

                        if args.keep_warnings and warns:
                            by_row[row_key].setdefault("_activate_warnings", {})[vi] = warns

                        if args.save_raw:
                            by_row[row_key]["raw_model_outputs"]["activate"][vi] = _normalize_raw(raw)
        else:
            # one-stage: mark all factors activated
            for row_key, rec in by_row.items():
                for vi in rec.get("value_items", []) or []:
                    facs = _get_factors_for_value_item(vi)
                    if not facs:
                        continue
                    rec.setdefault("_activated_factors", {})[vi] = [fo["factor"] for fo in facs]
                    rec.setdefault("_factors_by_vi", {})[vi] = {fo["factor"]: fo for fo in facs}

        # -------------------------
        # Stage 2: state per activated factor
        # -------------------------
        state_jobs: List[Dict[str, Any]] = []
        for row_key, rec in by_row.items():
            obj = row_objs.get(row_key, {})
            full_text = str(obj.get("full_text", "") or "")
            target_text = str(rec.get("target_text", "") or "")
            context_text = _infer_context_text(obj, full_text)

            factors_by_vi = rec.get("_factors_by_vi", {}) or {}
            activated_map = rec.get("_activated_factors", {}) or {}

            for vi in rec.get("value_items", []) or []:
                fac_map = factors_by_vi.get(vi, {}) or {}
                activated = activated_map.get(vi, [])

                if use_two_stage and not activated:
                    continue

                for factor in activated:
                    fo = fac_map.get(factor)
                    if not fo:
                        continue
                    prompt = USER_TEMPLATE_STATE.format(
                        value_item=vi,
                        group=fo["group"],
                        factor=fo["factor"],
                        definition=fo["definition"],
                        allowed_states=" | ".join([str(s) for s in fo["allowed_states"]]),
                        target_text=target_text[:1500],
                        context_text=context_text[:2000],
                    )
                    state_jobs.append({
                        "_row_key": row_key,
                        "value_item": vi,
                        "factor": fo["factor"],
                        "group": fo["group"],
                        "definition": fo["definition"],
                        "allowed_states": fo["allowed_states"],
                        "prompt": prompt,
                    })

        if state_jobs:
            llm.set_stage("state")
            for start in range(0, len(state_jobs), bs):
                chunk = state_jobs[start:start + bs]
                prompts = [j["prompt"] for j in chunk]
                raws = llm.run(prompts, max_tokens=args.max_new_tokens)
                llm_calls_state += len(prompts)

                for job, raw in zip(chunk, raws):
                    row_key = job["_row_key"]
                    vi = job["value_item"]
                    factor = job["factor"]
                    allowed_states = job["allowed_states"]

                    parsed, warnings = _parse_state_output(
                        raw=raw,
                        expected_value_item=vi,
                        expected_factor=factor,
                        allowed_states=allowed_states,
                        include_evidence=args.include_evidence,
                    )

                    entry = {
                        "group": job["group"],
                        "factor": factor,
                        "definition": job["definition"],
                        "states": allowed_states,
                        "selected_state": parsed["selected_state"],
                        "confidence": parsed["confidence"],
                    }
                    if args.include_evidence:
                        entry["evidence"] = parsed.get("evidence", "NONE")
                    if args.keep_warnings and warnings:
                        entry["warnings"] = warnings

                    by_row[row_key].setdefault("context_states", {}).setdefault(vi, []).append(entry)

                    if args.save_raw:
                        by_row[row_key]["raw_model_outputs"]["state"].setdefault(vi, {})
                        by_row[row_key]["raw_model_outputs"]["state"][vi][factor] = _normalize_raw(raw)

        # cleanup + write + pbar update
        for row_key, rec in by_row.items():
            rec.pop("_activated_factors", None)
            rec.pop("_activate_confidence", None)
            rec.pop("_activate_warnings", None)
            rec.pop("_factors_by_vi", None)

            if as_jsonl:
                _write_record_jsonl(rec)
            else:
                _write_record_json(rec)

            processed_rows += 1
            pbar.update(1)

    # stream rows -> chunk -> process
    for rm in row_iter:
        pending.append(rm)
        if len(pending) >= row_chunk_size:
            process_chunk(pending)
            pending = []

    if pending:
        process_chunk(pending)
        pending = []

    pbar.close()

    # finalize writers
    if as_jsonl:
        assert f_out is not None
        f_out.flush()
        f_out.close()
        if f_norm is not None:
            f_norm.flush()
            f_norm.close()
    else:
        with open(args.output_path, "w", encoding="utf-8") as wf:
            json.dump(all_records, wf, ensure_ascii=False, indent=2)
        if want_norm:
            with open(norm_path, "w", encoding="utf-8") as wf:
                norms = [_make_norm_record(r) for r in all_records]
                json.dump(norms, wf, ensure_ascii=False, indent=2)

    print(f"Saved: {args.output_path}")
    if want_norm:
        if as_jsonl:
            print(f"Saved (normalized): {norm_path}")
        else:
            print(f"Saved (normalized JSON): {norm_path}")
    print(f"Two-stage: {'ON' if use_two_stage else 'OFF'}")
    print(f"row_chunk_size: {row_chunk_size} | batch_size: {bs} | save_every_rows: {int(args.save_every_rows or 0)}")
    print(f"Resume: {'ON' if args.resume else 'OFF'} | skipped_rows: {skipped_rows} | processed_rows: {processed_rows}")
    print(f"LLM calls (activate): {llm_calls_activate}")
    print(f"LLM calls (state): {llm_calls_state}")
    print(f"LLM calls (total): {llm_calls_activate + llm_calls_state}")
    print(f"Evidence: {'ON' if args.include_evidence else 'OFF'}")


if __name__ == "__main__":
    main()
