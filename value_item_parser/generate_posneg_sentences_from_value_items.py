#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate positive and negative sentence examples for each value item.

Generation controls
-------------------
- A requested count of zero disables generation and uniqueness constraints for
  that label.
- Per-value run limits and consecutive no-progress limits stop generation for
  a value item and continue with the next item.
- --start_index / --end_index select a slice of the value catalog.
- --resume_from_output counts existing positive/negative records, skips
  completed values, and generates the remaining counts for incomplete values.
  Sentence deduplication applies within the current run only.

Output format
-------------
JSONL, one line per sentence:
{
  "value_item": ...,
  "label": "pos"|"neg",
  "gen_id": ...,
  "sentence": ...,
  "run_id": ...,
  "seed": ...,
  "temperature": ...,
  "top_p": ...,
  "value_item_index": ...
}
"""

import os
import re
import json
import argparse
import random
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from tqdm import tqdm

# The inference backend is imported when the generator is constructed.


# ===============================
# Prompts
# ===============================

SYSTEM_PROMPT = (
    "You are a dataset generator.\n"
    "Given a VALUE_ITEM, you will generate short, SINGLE-SENTENCE examples.\n"
    "Some sentences must be POSITIVE (support/endorse the value item), and some NEGATIVE "
    "(oppose/reject/criticize the value item).\n"
    "\n"
    "RULES\n"
    "- Output MUST be valid JSON only (no markdown, no commentary).\n"
    "- Each sentence must be exactly ONE sentence.\n"
    "- Keep each sentence <= 30 words.\n"
    "- Avoid slurs, hate, or explicit violence.\n"
    "- Avoid using 'positive/negative' words explicitly; just express stance naturally.\n"
    "- Ensure diversity (different speakers/roles/settings), but DO NOT mention 'context factors'.\n"
)

# braces escaped for .format()
USER_TEMPLATE = (
    "VALUE_ITEM: {value_item}\n"
    "N_POS: {n_pos}\n"
    "N_NEG: {n_neg}\n"
    "\n"
    "Generate JSON with EXACT keys:\n"
    "{{\n"
    '  "value_item": "<same as input>",\n'
    '  "pos": ["...", "...", ...],\n'
    '  "neg": ["...", "...", ...]\n'
    "}}\n"
    "\n"
    "Constraints:\n"
    "- pos must contain EXACTLY N_POS sentences.\n"
    "- neg must contain EXACTLY N_NEG sentences.\n"
    "- Each sentence must be ONE sentence.\n"
    "- Do not include additional keys.\n"
)


# ===============================
# Helpers
# ===============================

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


def _safe_json_load(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _try_extract_json_substring(raw: str) -> Optional[Dict[str, Any]]:
    raw_n = _normalize_raw(raw)
    j = _safe_json_load(raw_n)
    if isinstance(j, dict):
        return j
    m = re.search(r"\{.*\}", raw_n, flags=re.DOTALL)
    if m:
        return _safe_json_load(m.group(0))
    return None


def _dedupe_preserve_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        x2 = (x or "").strip()
        if not x2:
            continue
        if x2 not in seen:
            seen.add(x2)
            out.append(x2)
    return out


def _validate_one_sentence(s: str) -> bool:
    if not s or not isinstance(s, str):
        return False
    s = s.strip()
    if "\n" in s or "\r" in s:
        return False
    if len(s.split()) > 60:
        return False
    # discourage multi-sentence
    if s.count(".") >= 2 or s.count("!") >= 2 or s.count("?") >= 2:
        return False
    return True


def _sanitize_filename(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w\-.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:180] if len(s) > 180 else s


def _load_existing_counts(output_path: str) -> Dict[str, Dict[str, int]]:
    """
    Reads an existing JSONL output and returns counts per value_item:
      counts[value_item]["pos"], counts[value_item]["neg"]
    Silently ignores malformed lines.
    """
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"pos": 0, "neg": 0})
    if not output_path or not os.path.exists(output_path):
        return counts

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            vi = rec.get("value_item")
            lab = rec.get("label")
            if not isinstance(vi, str) or lab not in ("pos", "neg"):
                continue
            counts[vi][lab] += 1
    return counts


# ===============================
# LLM wrapper (passes kwargs -> SamplingParams)
# ===============================

class PosNegGenerator:
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        device: str = "cuda",
        dtype: str = "auto",
        system_prompt: str = SYSTEM_PROMPT,
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
            system_prompt=system_prompt,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.base_temperature = float(temperature)
        self.base_top_p = float(top_p)
        self.model_dir = model_dir

    def run(self, prompt: str, **kwargs) -> str:
        out = self.model([prompt], **kwargs)[0]
        return str(out) if out is not None else ""


# ===============================
# CLI
# ===============================

def parse_args():
    ap = argparse.ArgumentParser("Generate POS/NEG sentences per value item (seeded vLLM).")
    ap.add_argument("--factors_json", type=str, required=True, help="Used ONLY to enumerate unique value items.")
    ap.add_argument("--output_path", type=str, required=True, help=".jsonl recommended")

    # LLM configs
    ap.add_argument("--llm_model", type=str, default="Qwen/Qwen3-32B")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=800)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="auto")
    ap.add_argument("--model_dir", type=str, default=None)

    # targets per value item (allow 0)
    ap.add_argument("--pos_per_value_item", type=int, default=50)
    ap.add_argument("--neg_per_value_item", type=int, default=50)

    # per-run chunking
    ap.add_argument("--chunk_pos", type=int, default=5)
    ap.add_argument("--chunk_neg", type=int, default=5)

    # value item slicing (begin from middle)
    ap.add_argument("--max_value_items", type=int, default=0, help="0=all")
    ap.add_argument("--start_index", type=int, default=0, help="Start index in sorted value_items (0-based).")
    ap.add_argument("--end_index", type=int, default=0, help="End index (exclusive) in sorted value_items; 0=all.")

    # resume mode (skip already completed items by reading an existing output)
    ap.add_argument(
        "--resume_from_output",
        type=str,
        default=None,
        help="If set, read this JSONL and skip value items that already have enough pos/neg. "
             "If equal to --output_path, the script will append to it.",
    )

    # diversity knobs
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--seed_step", type=int, default=97)
    ap.add_argument("--temp_jitter", type=float, default=0.15)
    ap.add_argument("--min_unique_ratio", type=float, default=0.85)
    ap.add_argument("--max_retries_per_chunk", type=int, default=2)

    # skip / safety
    ap.add_argument("--max_runs_per_value_item", type=int, default=500)
    ap.add_argument("--max_no_progress_runs", type=int, default=30)

    # output
    ap.add_argument("--save_raw", action="store_true")
    ap.add_argument("--save_per_value_item", action="store_true")

    return ap.parse_args()


def _consume_chunk(
    gen: PosNegGenerator,
    vi: str,
    need_pos: int,
    need_neg: int,
    n_pos: int,
    n_neg: int,
    seed_i: int,
    temp_i: float,
    top_p: float,
    max_retries: int,
    min_unique_ratio: float,
    already_pos: set,
    already_neg: set,
) -> Tuple[List[str], List[str], str, int]:
    """
    Returns: (pos_sents, neg_sents, raw, used_seed)
    """
    prompt = USER_TEMPLATE.format(value_item=vi, n_pos=n_pos, n_neg=n_neg)

    best_pos: List[str] = []
    best_neg: List[str] = []
    best_raw = ""

    used_seed = seed_i

    for attempt in range(max_retries + 1):
        raw = gen.run(prompt, seed=used_seed, temperature=temp_i, top_p=top_p)
        raw_n = _normalize_raw(raw)
        best_raw = raw_n

        j = _try_extract_json_substring(raw_n)

        pos: List[str] = []
        neg: List[str] = []
        if isinstance(j, dict):
            p = j.get("pos", [])
            n = j.get("neg", [])
            if isinstance(p, list):
                pos = [str(x).strip() for x in p]
            if isinstance(n, list):
                neg = [str(x).strip() for x in n]

        pos = [s for s in pos if _validate_one_sentence(s)]
        neg = [s for s in neg if _validate_one_sentence(s)]

        # remove duplicates vs collected
        pos = [s for s in pos if s not in already_pos]
        neg = [s for s in neg if s not in already_neg]
        pos = _dedupe_preserve_order(pos)
        neg = _dedupe_preserve_order(neg)

        # A zero requested count has no uniqueness or collection constraint.
        pos_ratio = (len(pos) / n_pos) if n_pos > 0 else 1.0
        neg_ratio = (len(neg) / n_neg) if n_neg > 0 else 1.0

        pos_ok = (n_pos == 0) or (len(pos) >= min(n_pos, need_pos) and pos_ratio >= min_unique_ratio)
        neg_ok = (n_neg == 0) or (len(neg) >= min(n_neg, need_neg) and neg_ratio >= min_unique_ratio)

        ok = pos_ok and neg_ok

        if ok:
            return pos[:min(n_pos, need_pos)], neg[:min(n_neg, need_neg)], best_raw, used_seed

        # keep best effort
        if len(pos) + len(neg) > len(best_pos) + len(best_neg):
            best_pos, best_neg = pos, neg

        # retry: bump seed deterministically
        used_seed = seed_i + 10_000 * (attempt + 1)

    return best_pos[:min(n_pos, need_pos)], best_neg[:min(n_neg, need_neg)], best_raw, used_seed


def main():
    args = parse_args()
    _ensure_dir(args.output_path)

    with open(args.factors_json, "r", encoding="utf-8") as f:
        factors_data = json.load(f)

    items = factors_data.get("items", {})
    if not isinstance(items, dict) or not items:
        raise ValueError("factors_json must have non-empty top-level key 'items' as a dict.")

    value_items = [str(v).strip() for v in items.keys() if str(v).strip()]
    value_items.sort()

    if args.max_value_items and args.max_value_items > 0:
        value_items = value_items[:args.max_value_items]

    # slicing (begin from middle)
    sidx = max(0, int(args.start_index))
    eidx = int(args.end_index) if args.end_index and int(args.end_index) > 0 else len(value_items)
    value_items = value_items[sidx:eidx]
    if not value_items:
        raise ValueError("No value items selected after slicing.")

    # validate chunk sizes and totals (allow 0 totals)
    if args.pos_per_value_item < 0 or args.neg_per_value_item < 0:
        raise ValueError("--pos_per_value_item and --neg_per_value_item must be >= 0.")
    if args.pos_per_value_item > 0 and args.chunk_pos <= 0:
        raise ValueError("--chunk_pos must be positive when --pos_per_value_item > 0.")
    if args.neg_per_value_item > 0 and args.chunk_neg <= 0:
        raise ValueError("--chunk_neg must be positive when --neg_per_value_item > 0.")

    gen = PosNegGenerator(
        model_name=args.llm_model,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
        model_dir=args.model_dir,
    )

    rng = random.Random(args.seed)

    # resume counts
    existing_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"pos": 0, "neg": 0})
    if args.resume_from_output:
        existing_counts = _load_existing_counts(args.resume_from_output)

    # open output file: append if resuming into same file, else overwrite
    write_mode = "a" if (args.resume_from_output and os.path.abspath(args.resume_from_output) == os.path.abspath(args.output_path)) else "w"
    f_out = open(args.output_path, write_mode, encoding="utf-8")

    per_dir = None
    if args.save_per_value_item:
        out_dir = os.path.dirname(os.path.abspath(args.output_path)) or "."
        per_dir = os.path.join(out_dir, "per_value_item")
        os.makedirs(per_dir, exist_ok=True)

    total_written = 0
    processed = 0
    skipped = 0

    for vi_idx, vi in enumerate(tqdm(value_items, desc="value_items")):
        # Resume skip: if already enough, continue
        if args.resume_from_output:
            have_pos = int(existing_counts.get(vi, {}).get("pos", 0))
            have_neg = int(existing_counts.get(vi, {}).get("neg", 0))
            if have_pos >= args.pos_per_value_item and have_neg >= args.neg_per_value_item:
                continue

        # Collect this run's additions separately from the existing record counts.
        pos_collected: List[str] = []
        neg_collected: List[str] = []
        pos_set, neg_set = set(), set()

        # Resume loads counts, not prior sentences; deduplication is run-local.
        # TODO: Load prior sentences into these sets for cross-run deduplication.
        run_id = 0
        no_progress_runs = 0

        # per-value file
        f_vi = None
        if per_dir is not None:
            vi_slug = _sanitize_filename(vi) or f"vi_{vi_idx + sidx}"
            vi_path = os.path.join(per_dir, f"{vi_slug}.jsonl")
            # if resuming, append, else overwrite
            vi_mode = "a" if (args.resume_from_output and os.path.exists(vi_path)) else "w"
            f_vi = open(vi_path, vi_mode, encoding="utf-8")

        # figure out remaining targets considering existing counts
        already_pos_n = int(existing_counts.get(vi, {}).get("pos", 0)) if args.resume_from_output else 0
        already_neg_n = int(existing_counts.get(vi, {}).get("neg", 0)) if args.resume_from_output else 0

        target_pos_total = max(0, args.pos_per_value_item - already_pos_n)
        target_neg_total = max(0, args.neg_per_value_item - already_neg_n)

        # nothing to do
        if target_pos_total == 0 and target_neg_total == 0:
            if f_vi is not None:
                f_vi.close()
            continue

        while len(pos_collected) < target_pos_total or len(neg_collected) < target_neg_total:
            # safety caps
            if run_id >= int(args.max_runs_per_value_item):
                print(f"[SKIP] value_item='{vi}' hit max_runs_per_value_item={args.max_runs_per_value_item}")
                skipped += 1
                break
            if no_progress_runs >= int(args.max_no_progress_runs):
                print(f"[SKIP] value_item='{vi}' hit max_no_progress_runs={args.max_no_progress_runs}")
                skipped += 1
                break

            need_pos = target_pos_total - len(pos_collected)
            need_neg = target_neg_total - len(neg_collected)

            n_pos = min(args.chunk_pos, need_pos) if need_pos > 0 else 0
            n_neg = min(args.chunk_neg, need_neg) if need_neg > 0 else 0

            seed_i = int(args.seed + run_id * args.seed_step)
            temp_i = float(args.temperature + rng.uniform(-args.temp_jitter, args.temp_jitter))
            temp_i = max(0.05, min(1.5, temp_i))

            got_pos, got_neg, raw, used_seed = _consume_chunk(
                gen=gen,
                vi=vi,
                need_pos=need_pos,
                need_neg=need_neg,
                n_pos=n_pos,
                n_neg=n_neg,
                seed_i=seed_i,
                temp_i=temp_i,
                top_p=float(args.top_p),
                max_retries=int(args.max_retries_per_chunk),
                min_unique_ratio=float(args.min_unique_ratio),
                already_pos=pos_set,
                already_neg=neg_set,
            )

            if len(got_pos) == 0 and len(got_neg) == 0:
                no_progress_runs += 1
            else:
                no_progress_runs = 0

            # write out
            for s in got_pos:
                pos_collected.append(s)
                pos_set.add(s)
                rec: Dict[str, Any] = {
                    "value_item": vi,
                    "label": "pos",
                    # gen_id should reflect total across all runs; include offset for resume
                    "gen_id": already_pos_n + (len(pos_collected) - 1),
                    "sentence": s,
                    "run_id": run_id,
                    "seed": int(used_seed),
                    "temperature": float(temp_i),
                    "top_p": float(args.top_p),
                    "value_item_index": int(vi_idx + sidx),
                }
                if args.save_raw:
                    rec["raw"] = raw
                line = json.dumps(rec, ensure_ascii=False) + "\n"
                f_out.write(line)
                if f_vi is not None:
                    f_vi.write(line)
                total_written += 1

            for s in got_neg:
                neg_collected.append(s)
                neg_set.add(s)
                rec = {
                    "value_item": vi,
                    "label": "neg",
                    "gen_id": already_neg_n + (len(neg_collected) - 1),
                    "sentence": s,
                    "run_id": run_id,
                    "seed": int(used_seed),
                    "temperature": float(temp_i),
                    "top_p": float(args.top_p),
                    "value_item_index": int(vi_idx + sidx),
                }
                if args.save_raw:
                    rec["raw"] = raw
                line = json.dumps(rec, ensure_ascii=False) + "\n"
                f_out.write(line)
                if f_vi is not None:
                    f_vi.write(line)
                total_written += 1

            run_id += 1

        # flush after each value item
        f_out.flush()
        os.fsync(f_out.fileno())
        if f_vi is not None:
            f_vi.flush()
            os.fsync(f_vi.fileno())
            f_vi.close()

        processed += 1

    f_out.flush()
    os.fsync(f_out.fileno())
    f_out.close()

    print(f"Saved: {args.output_path}")
    if per_dir is not None:
        print(f"Per-value-item dir: {per_dir}")
    print(f"Total sentences written (this run): {total_written}")
    print(f"Value items iterated (slice): {len(value_items)}")
    print(f"Value items processed (attempted): {processed}")
    print(f"Value items skipped (safety): {skipped}")
    print(f"Per value item target: pos={args.pos_per_value_item}, neg={args.neg_per_value_item}")
    if args.resume_from_output:
        print(f"Resume source: {args.resume_from_output}")
        print("NOTE: Dedupe is only within this run. If you need global dedupe, load prior sentences too.")


if __name__ == "__main__":
    main()
