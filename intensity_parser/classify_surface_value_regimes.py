#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classify value expressions using value definitions and optional surrounding context.

--top_k_value_items (default 4) selects the top-k scored value_items in each row.
Items without scores rank last and can be included when fewer scored items exist.
Category signals are interpreted as hints rather than strict matching rules.

Input JSONL example:
{
  "idx": 0,
  "text_type": "...",
  "full_text": "...",
  "target_sentence": "...",
  "expressive_type": "...",
  "expressive_score": 60,
  "value_items": [{"value_item": "religious authority", "score": 90}, ...],
  ...
}

Outputs:
- output_jsonl_original: original record + surface_value_regimes (+ optional raw)
- output_jsonl (optional): normalized per row summary
"""

from __future__ import annotations

import os
import re
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# -------------------------------
# Imports from your codebase
# -------------------------------
# The inference backend is imported when the classifier is constructed.


# ===============================
# Prompting
# ===============================

def build_system_prompt() -> str:
    # Reserve Not Invoked for unlinked values; treat category signals as hints.
    return (
        "You are a classifier for SURFACE VALUE REGIMES.\n"
        "You will be given:\n"
        "- VALUE_ITEM (string)\n"
        "- VALUE_ITEM_DEFINITION (short definition; may be empty)\n"
        "- A set of CATEGORIES, each with description and example signals\n"
        "- TARGET_SENTENCE (the sentence to classify)\n"
        "- CONTEXT (optional; surrounding text to help interpret TARGET_SENTENCE)\n\n"
        "IMPORTANT\n"
        "The provided 'signals' are NUANCED HINTS (soft cues), not hard rules.\n\n"
        "TASK\n"
        "Choose EXACTLY ONE category name from the provided categories.\n"
        "Decide based on how TARGET_SENTENCE positions/evaluates/argues about VALUE_ITEM.\n"
        "You MAY use CONTEXT only to disambiguate what TARGET_SENTENCE refers to.\n\n"
        "GUIDANCE\n"
        "- Prefer a non-'Not Invoked' category if TARGET_SENTENCE clearly evaluates, justifies, condemns,\n"
        "  endorses, rejects, constrains, or trades off the VALUE_ITEM (directly or by clear implication).\n"
        "- Choose 'Not Invoked' ONLY when there is no reasonable link between TARGET_SENTENCE and VALUE_ITEM,\n"
        "  even after considering CONTEXT.\n"
        "- If multiple categories fit, pick the single best-supported by TARGET_SENTENCE.\n\n"
        "RULES\n"
        "- Do NOT invent facts not supported by TARGET_SENTENCE (and CONTEXT only for reference resolution).\n"
        "- CATEGORY must be one of the provided names (verbatim).\n\n"
        "OUTPUT FORMAT (STRICT)\n"
        "Exactly 3 lines, no extra text:\n"
        "VALUE_ITEM: <value item>\n"
        "CATEGORY: <one category name>\n"
        "CONFIDENCE: <integer 0..100>\n"
    )


def _format_categories_block(categories: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for c in categories:
        name = str(c.get("name", "") or "").strip()
        desc = str(c.get("description", "") or "").strip()
        sigs = c.get("signals", [])
        if not isinstance(sigs, list):
            sigs = []
        sigs = [str(s).strip() for s in sigs if str(s).strip()]
        sigs = sigs[:5]
        sig_str = "; ".join(sigs) if sigs else "NONE"
        lines.append(f"- {name}\n  desc: {desc}\n  signals (hints, not rules): {sig_str}")
    return "\n".join(lines)


USER_TEMPLATE_WITH_CONTEXT = (
    "VALUE_ITEM: {value_item}\n"
    "VALUE_ITEM_DEFINITION: {value_def}\n\n"
    "CATEGORIES:\n"
    "{categories_block}\n\n"
    "TARGET_SENTENCE:\n"
    "<<<\n"
    "{target_sentence}\n"
    ">>>\n\n"
    "CONTEXT (optional):\n"
    "<<<\n"
    "{context_text}\n"
    ">>>\n"
)

USER_TEMPLATE_NO_CONTEXT = (
    "VALUE_ITEM: {value_item}\n"
    "VALUE_ITEM_DEFINITION: {value_def}\n\n"
    "CATEGORIES:\n"
    "{categories_block}\n\n"
    "TARGET_SENTENCE:\n"
    "<<<\n"
    "{target_sentence}\n"
    ">>>\n"
)


# ===============================
# Parsing helpers
# ===============================

_RE_VI = re.compile(r"^\s*VALUE_ITEM\s*:\s*(.+?)\s*$", re.IGNORECASE)
_RE_CAT = re.compile(r"^\s*CATEGORY\s*:\s*(.+?)\s*$", re.IGNORECASE)
_RE_CONF = re.compile(r"^\s*CONFIDENCE\s*:\s*(\d{1,3})\s*$", re.IGNORECASE)


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


def _parse_category_output(
    raw: str,
    expected_value_item: str,
    allowed_categories: List[str],
) -> Tuple[Dict[str, Any], List[str]]:
    s = _normalize_raw(raw)
    warnings: List[str] = []

    if not s:
        return {"category": "UNKNOWN", "confidence": 0}, ["empty_output"]

    vi = cat = conf = None
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    for ln in lines:
        m = _RE_VI.match(ln)
        if m:
            vi = m.group(1).strip()
        m = _RE_CAT.match(ln)
        if m:
            cat = m.group(1).strip()
        m = _RE_CONF.match(ln)
        if m:
            conf = m.group(1).strip()

    if vi and vi.strip().lower() != (expected_value_item or "").strip().lower():
        warnings.append(f"value_item_mismatch: got='{vi}' expected='{expected_value_item}'")

    allowed_set = {x.strip().lower() for x in allowed_categories}
    if cat is None:
        warnings.append("missing_category")
        selected = "UNKNOWN"
    else:
        low = cat.strip().lower()
        if low not in allowed_set:
            warnings.append(f"category_not_allowed: '{cat}'")
            selected = "UNKNOWN"
        else:
            selected = next((x for x in allowed_categories if x.strip().lower() == low), cat)

    confidence = _clip_int(conf, 0, 100, default=0)
    if conf is None:
        warnings.append("missing_confidence")

    return {"category": selected, "confidence": confidence}, warnings

def _dedup_value_items_keep_max_score(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate value_items by value_item string.
    If duplicates exist, keep the one with the highest numeric score.
    Items without scores are treated as lowest priority.
    """
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
            # replace if current score is higher
            if prev_sc is None or (sc is not None and sc > prev_sc):
                best[vi] = it

    return list(best.values())

# ===============================
# LLM wrapper
# ===============================

class SurfaceRegimeClassifier:
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        device: str = "cuda",
        dtype: str = "auto",
        model_dir: Optional[str] = None,
    ):
        from model import LLMModel
        self.model = LLMModel(
            model=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
            dtype=dtype,
            model_dir=model_dir,
            system_prompt=build_system_prompt(),
            tensor_parallel_size=tensor_parallel_size,
        )
        self.top_p = top_p
        self.model_dir = model_dir

    def run(self, prompts: List[str]) -> List[str]:
        outs = self.model(prompts)
        return [str(o) if o is not None else "" for o in outs]


# ===============================
# Input helpers
# ===============================

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


def _score_to_float(score: Any) -> Optional[float]:
    if score is None:
        return None
    try:
        return float(score)
    except Exception:
        return None


def _top_k_value_items(items: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """
    Keep only top-k by score (descending). Missing/unparseable scores go last.
    Stable-ish: ties preserve relative order among scored items after sort.
    """
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

    # If not enough scored items, fill with unscored in original order
    if len(picked_idx) < k:
        need = k - len(picked_idx)
        picked_idx.extend(unscored[:need])

    # Keep original order among picked indices to avoid shuffling too much
    picked_set = set(picked_idx)
    return [it for j, it in enumerate(items) if j in picked_set][:k]


def _load_regimes_jsonl(regimes_jsonl: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(regimes_jsonl):
        raise FileNotFoundError(regimes_jsonl)

    regimes: Dict[str, Dict[str, Any]] = {}
    with open(regimes_jsonl, "r", encoding="utf-8") as f:
        for _, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            vi = str(obj.get("value_item", "") or "").strip()
            cats = obj.get("categories", [])
            if not vi or not isinstance(cats, list) or not cats:
                continue
            regimes[vi] = {
                "categories": cats,
                "meta": obj.get("meta", {}),
            }
    return regimes


def _load_value_defs_json(path: str) -> Dict[str, str]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"value_defs_json must be a JSON object (dict), got: {type(obj)}")
    out: Dict[str, str] = {}
    for k, v in obj.items():
        kk = str(k).strip()
        vv = "" if v is None else str(v).strip()
        if kk:
            out[kk] = vv
    return out


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
        start = max(0, pos - window_chars // 2)
        end = min(len(ft), pos + len(ts) + window_chars // 2)
        return ft[start:end]
    return ft[:window_chars]


# ===============================
# Robust LLM calling with fallback
# ===============================

def _run_with_fallback(
    clf: SurfaceRegimeClassifier,
    jobs: List[Dict[str, Any]],
    batch_size: int,
    prefer_context: bool,
    keep_warnings: bool,
) -> Tuple[List[str], List[List[str]]]:
    raws: List[str] = [""] * len(jobs)
    call_warnings: List[List[str]] = [[] for _ in jobs]

    def pick_prompt(j: Dict[str, Any], use_context: bool) -> str:
        if use_context:
            return j.get("prompt_with_context") or j.get("prompt_no_context") or ""
        return j.get("prompt_no_context") or ""

    def run_batch(indices: List[int], use_context: bool, bs: int):
        if not indices:
            return
        prompts = [pick_prompt(jobs[i], use_context) for i in indices]
        try:
            outs = clf.run(prompts)
            for i, out in zip(indices, outs):
                raws[i] = str(out) if out is not None else ""
        except Exception:
            if bs <= 1 or len(indices) == 1:
                i = indices[0]
                if use_context:
                    if keep_warnings:
                        call_warnings[i].append("llm_error_with_context_fallback_no_context")
                    try:
                        out2 = clf.run([pick_prompt(jobs[i], use_context=False)])[0]
                        raws[i] = str(out2) if out2 is not None else ""
                        return
                    except Exception:
                        if keep_warnings:
                            call_warnings[i].append("llm_error_no_context")
                        raws[i] = ""
                        return
                else:
                    if keep_warnings:
                        call_warnings[i].append("llm_error_no_context")
                    raws[i] = ""
                    return

            mid = len(indices) // 2
            run_batch(indices[:mid], use_context, max(1, bs // 2))
            run_batch(indices[mid:], use_context, max(1, bs // 2))

    all_idx = list(range(len(jobs)))
    run_batch(all_idx, use_context=prefer_context, bs=max(1, int(batch_size)))
    return raws, call_warnings


# ===============================
# CLI
# ===============================

def parse_args():
    ap = argparse.ArgumentParser(
        "Classify each (target_sentence,value_item) into a surface value regime category using regimes JSONL. (JSONL input only)"
    )

    ap.add_argument("--input_jsonl", type=str, required=True)
    ap.add_argument("--regimes_jsonl", type=str, required=True)

    ap.add_argument(
        "--value_defs_json",
        type=str,
        default=None,
        help="Path to unique_value_item_definitions.json (dict: value_item -> definition).",
    )

    ap.add_argument(
        "--output_jsonl_original",
        type=str,
        required=True,
        help="Write original JSONL records with added fields: surface_value_regimes (+ optional raw).",
    )
    ap.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
        help="Optional: also save normalized JSONL per row (structured summary).",
    )

    ap.add_argument("--idx_col", type=str, default="idx")
    ap.add_argument("--original_text_idx_col", type=str, default="original_text_idx")

    ap.add_argument("--target_sentence_key", type=str, default="target_sentence")
    ap.add_argument("--full_text_key", type=str, default="full_text")

    ap.add_argument(
        "--jsonl_value_items_key",
        type=str,
        default="value_items",
        help="Which JSON key contains value items (default: value_items; expects list[dict] with value_item).",
    )

    ap.add_argument(
        "--top_k_value_items",
        type=int,
        default=4,
        help="Only consider top-k value_items per row by score (descending). Default=4. Set 0 to skip all.",
    )

    ap.add_argument("--max_rows", type=int, default=0, help="0=all")

    ap.add_argument(
        "--save_every_rows",
        type=int,
        default=1000,
        help="Flush/write outputs every N input rows (buffered).",
    )

    ap.add_argument(
        "--count_total_first",
        action="store_true",
        help="Count total input JSONL lines first (gives ETA, costs one extra pass).",
    )

    # Context controls
    ap.add_argument(
        "--use_context",
        action="store_true",
        help="If set, include a context window from full_text around target_sentence (fallback to head slice).",
    )
    ap.add_argument(
        "--context_window_chars",
        type=int,
        default=1400,
        help="Context window size (chars). If target_sentence found in full_text, extracts window around it.",
    )
    ap.add_argument(
        "--context_max_chars",
        type=int,
        default=1800,
        help="Hard truncate for context text inserted into prompt (chars).",
    )
    ap.add_argument(
        "--target_max_chars",
        type=int,
        default=600,
        help="Hard truncate for target_sentence inserted into prompt (chars).",
    )
    ap.add_argument(
        "--def_max_chars",
        type=int,
        default=300,
        help="Hard truncate for value item definition inserted into prompt (chars).",
    )

    # LLM configs
    ap.add_argument("--llm_model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="auto")
    ap.add_argument("--model_dir", type=str, default=None)

    # LLM batching
    ap.add_argument("--batch_size", type=int, default=64)

    # output toggles
    ap.add_argument("--save_raw", action="store_true", help="Save raw model outputs per (row,value_item).")
    ap.add_argument("--keep_warnings", action="store_true", help="Keep parser warnings per (row,value_item).")

    return ap.parse_args()


def main():
    args = parse_args()
    _ensure_dir(args.output_jsonl_original)
    if args.output_jsonl:
        _ensure_dir(args.output_jsonl)

    print("[1/9] Loading regimes JSONL...", flush=True)
    regimes = _load_regimes_jsonl(args.regimes_jsonl)
    if not regimes:
        raise ValueError(f"No regimes loaded from: {args.regimes_jsonl}")
    print(f"  - regimes loaded: {len(regimes)} value_items", flush=True)

    print("[2/9] Loading value definitions JSON (optional)...", flush=True)
    value_defs = _load_value_defs_json(args.value_defs_json) if args.value_defs_json else {}
    print(f"  - value definitions loaded: {len(value_defs)}", flush=True)

    print("[3/9] Loading LLM model...", flush=True)
    clf = SurfaceRegimeClassifier(
        model_name=args.llm_model,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
        model_dir=args.model_dir,
    )
    print("  - model ready", flush=True)

    print("[4/9] Streaming input + classifying...", flush=True)
    bs = max(1, int(args.batch_size))
    row_batch_size = max(1, int(args.save_every_rows))

    total_lines: Optional[int] = None
    if args.count_total_first:
        print("  - counting total lines for global tqdm...", flush=True)
        total_lines = _count_jsonl_records(args.input_jsonl)
        if args.max_rows and args.max_rows > 0:
            total_lines = min(total_lines, int(args.max_rows))
        print(f"  - total rows (approx): {total_lines}", flush=True)

    f_orig = open(args.output_jsonl_original, "w", encoding="utf-8")
    f_norm = open(args.output_jsonl, "w", encoding="utf-8") if args.output_jsonl else None

    try:
        buffer_rows: List[Dict[str, Any]] = []
        buffer_orig_objs: List[Dict[str, Any]] = []

        def flush_buffer(chunk_id: int, global_pbar: tqdm):
            nonlocal buffer_rows, buffer_orig_objs
            if not buffer_rows:
                return

            flat_jobs: List[Dict[str, Any]] = []

            for local_i, r in enumerate(buffer_rows):
                target_sentence = str(r.get("target_sentence", "") or "")
                target_sentence = _safe_truncate(target_sentence, args.target_max_chars)

                context_text = ""
                if args.use_context:
                    full_text = str(r.get("full_text", "") or "")
                    ctx = _extract_context_window(full_text, target_sentence, args.context_window_chars)
                    context_text = _safe_truncate(ctx, args.context_max_chars)

                vis: List[Dict[str, Any]] = r.get("value_items", []) or []

                # 1) deduplicate same value_item within the row (keep max score)
                vis = _dedup_value_items_keep_max_score(vis)

                # 2) apply TOP-K per row
                vis = _top_k_value_items(vis, int(args.top_k_value_items))

                for it in vis:
                    vi = str(it.get("value_item", "") or "").strip()
                    if not vi:
                        continue

                    regime = regimes.get(vi)
                    if not regime:
                        flat_jobs.append({
                            "local_i": local_i,
                            "value_item": vi,
                            "missing_regime": True,
                            "allowed_categories": ["UNKNOWN"],
                            "prompt_with_context": "",
                            "prompt_no_context": "",
                        })
                        continue

                    categories = regime["categories"]
                    allowed = [
                        str(c.get("name", "") or "").strip()
                        for c in categories
                        if str(c.get("name", "") or "").strip()
                    ]
                    allowed = [a for a in allowed if a]
                    if not allowed:
                        flat_jobs.append({
                            "local_i": local_i,
                            "value_item": vi,
                            "missing_regime": True,
                            "allowed_categories": ["UNKNOWN"],
                            "prompt_with_context": "",
                            "prompt_no_context": "",
                        })
                        continue

                    vdef = value_defs.get(vi, "")
                    vdef = _safe_truncate(vdef, args.def_max_chars)

                    categories_block = _format_categories_block(categories)

                    prompt_no_ctx = USER_TEMPLATE_NO_CONTEXT.format(
                        value_item=vi,
                        value_def=(vdef or "N/A"),
                        categories_block=categories_block,
                        target_sentence=target_sentence,
                    )

                    prompt_with_ctx = USER_TEMPLATE_WITH_CONTEXT.format(
                        value_item=vi,
                        value_def=(vdef or "N/A"),
                        categories_block=categories_block,
                        target_sentence=target_sentence,
                        context_text=(context_text or "N/A"),
                    )

                    flat_jobs.append({
                        "local_i": local_i,
                        "value_item": vi,
                        "missing_regime": False,
                        "allowed_categories": allowed,
                        "prompt_with_context": prompt_with_ctx,
                        "prompt_no_context": prompt_no_ctx,
                    })

            jobs_with_prompts = [j for j in flat_jobs if not j.get("missing_regime")]
            jobs_missing = [j for j in flat_jobs if j.get("missing_regime")]

            per_row_regimes: List[Dict[str, Any]] = [dict() for _ in buffer_rows]
            per_row_raw: List[Dict[str, Any]] = [dict() for _ in buffer_rows] if args.save_raw else []

            # Missing regime definition
            for j in jobs_missing:
                li = int(j["local_i"])
                vi = j["value_item"]
                entry: Dict[str, Any] = {"category": "UNKNOWN", "confidence": 0}
                if args.keep_warnings:
                    entry["warnings"] = ["missing_regime_definition"]
                per_row_regimes[li][vi] = entry

            # Run LLM with robust fallback
            with tqdm(total=len(jobs_with_prompts), desc=f"llm(chunk={chunk_id})", leave=False) as llm_pbar:
                for start in range(0, len(jobs_with_prompts), bs):
                    sub_jobs = jobs_with_prompts[start:start + bs]
                    raws, call_warns = _run_with_fallback(
                        clf=clf,
                        jobs=sub_jobs,
                        batch_size=len(sub_jobs),
                        prefer_context=bool(args.use_context),
                        keep_warnings=bool(args.keep_warnings),
                    )

                    for job, raw, cw in zip(sub_jobs, raws, call_warns):
                        li = int(job["local_i"])
                        vi = job["value_item"]
                        allowed = job["allowed_categories"]

                        parsed, parse_warns = _parse_category_output(
                            raw=raw,
                            expected_value_item=vi,
                            allowed_categories=allowed,
                        )

                        entry: Dict[str, Any] = {
                            "category": parsed["category"],
                            "confidence": parsed["confidence"],
                        }

                        if args.keep_warnings:
                            merged_warns: List[str] = []
                            if cw:
                                merged_warns.extend(cw)
                            if parse_warns:
                                merged_warns.extend(parse_warns)
                            if merged_warns:
                                entry["warnings"] = merged_warns

                        per_row_regimes[li][vi] = entry
                        if args.save_raw:
                            per_row_raw[li][vi] = _normalize_raw(raw)

                    llm_pbar.update(len(sub_jobs))

            # Write outputs
            for li, (r, orig) in enumerate(zip(buffer_rows, buffer_orig_objs)):
                out_obj = dict(orig)
                out_obj["surface_value_regimes"] = per_row_regimes[li]
                if args.save_raw:
                    out_obj["surface_value_regimes_raw"] = per_row_raw[li]
                f_orig.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

                if f_norm is not None:
                    # keep the original value_items (not top-k filtered) for traceability
                    rec: Dict[str, Any] = {
                        args.idx_col: r.get(args.idx_col, None),
                        args.original_text_idx_col: r.get(args.original_text_idx_col, None),
                        "target_sentence": r.get("target_sentence", ""),
                        "full_text_present": bool(r.get("full_text", "")),
                        "value_items": r.get("value_items", []),
                        "top_k_value_items_used": int(args.top_k_value_items),
                        "surface_value_regimes": per_row_regimes[li],
                    }
                    if args.save_raw:
                        rec["surface_value_regimes_raw"] = per_row_raw[li]
                    f_norm.write(json.dumps(rec, ensure_ascii=False) + "\n")

            f_orig.flush()
            if f_norm is not None:
                f_norm.flush()

            global_pbar.update(len(buffer_rows))
            buffer_rows = []
            buffer_orig_objs = []

        global_pbar = tqdm(total=total_lines, desc="rows(total)", unit="rows")

        chunk_id = 0
        for line_idx, obj in _iter_jsonl(args.input_jsonl, max_rows=args.max_rows):
            idx = obj.get(args.idx_col, obj.get("idx", line_idx))
            original_text_idx = obj.get(args.original_text_idx_col, obj.get("original_text_idx", None))

            target_sentence = obj.get(args.target_sentence_key, "")
            target_sentence = "" if target_sentence is None else str(target_sentence)

            full_text = obj.get(args.full_text_key, "")
            full_text = "" if full_text is None else str(full_text)

            value_items_raw = obj.get(args.jsonl_value_items_key, [])
            vis = _parse_value_items_with_scores(value_items_raw)

            buffer_orig_objs.append(obj)
            buffer_rows.append({
                args.idx_col: idx,
                args.original_text_idx_col: original_text_idx,
                "target_sentence": target_sentence,
                "full_text": full_text,
                "value_items": vis,
            })

            if len(buffer_rows) >= row_batch_size:
                chunk_id += 1
                flush_buffer(chunk_id=chunk_id, global_pbar=global_pbar)

        if buffer_rows:
            chunk_id += 1
            flush_buffer(chunk_id=chunk_id, global_pbar=global_pbar)

        global_pbar.close()

    finally:
        f_orig.close()
        if f_norm is not None:
            f_norm.close()

    print("[9/9] Done.", flush=True)
    print(f"  - Saved ORIGINAL+RESULT JSONL: {args.output_jsonl_original}", flush=True)
    if args.output_jsonl:
        print(f"  - Saved NORMALIZED JSONL: {args.output_jsonl}", flush=True)


if __name__ == "__main__":
    main()
