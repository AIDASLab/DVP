#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_value_items_from_expressive_jsonl.py

Given expressive-sentence jsonl like:
{
  "idx": 0,
  "text": "...",
  "text_type": "comment",
  "annotations": [{"sentence": "...", "score": 60, "type": "opinion"}, ...],
  ...
}

Goal:
- Drop rows where annotations == []
- For each annotation sentence:
  - prompt LLM with:
      (A) full text
      (B) target sentence
    and return VALUE ITEMS with relevance scores.

STRICT OUTPUT (model):
- Must output ONLY valid JSON:
  [
    {"value_item": "<short phrase>", "score": <0..100>},
    ...
  ]
- If none applies, output exactly:
  []

Outputs:
- JSONL only: one record per (idx, annotation sentence)

Example output record:
{
  "idx": 0,
  "text_type": "comment",
  "full_text": "...",
  "target_sentence": "...",
  "expressive_type": "opinion",
  "expressive_score": 60,
  "value_items": [{"value_item":"critical thinking","score":72}, ...]
}
"""

import os
import re
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# -------------------------------
# Imports from your codebase
# -------------------------------
# The inference backend is imported when the extractor is constructed.


# ===============================
# Prompting
# ===============================

VALUE_SYSTEM_PROMPT = (
    "You are a strict multilingual VALUE-ITEM extractor.\n"
    "Your job is HIGH PRECISION: only output value items that are clearly supported by the input.\n\n"
    "TASK\n"
    "Given:\n"
    "  - FULL_TEXT: the entire original text\n"
    "  - TARGET_SENTENCE: an extracted expressive sentence from the full text\n"
    "Return a list of value-related items that are relevant to TARGET_SENTENCE (and supported by FULL_TEXT context).\n\n"
    "VALUE ITEMS include short phrases spanning:\n"
    "- human values (e.g., 'honesty', 'self-reliance')\n"
    "- moral norms (e.g., 'fairness', 'harm avoidance')\n"
    "- duties/obligations (e.g., 'duty to care for family')\n"
    "- rights/entitlements (e.g., 'freedom of expression')\n"
    "- cultural values/customs (e.g., 'respect for elders')\n"
    "- professional/occupational codes (e.g., 'journalistic integrity')\n\n"
    "SCORING (0-100)\n"
    "Score expresses how strongly/relevantly the value item is implicated by TARGET_SENTENCE.\n"
    "- 0-20: barely related / very weak implication\n"
    "- 21-50: somewhat related\n"
    "- 51-80: clearly related\n"
    "- 81-100: central / explicit\n\n"
    "RULES\n"
    "1) Output ONLY valid JSON (no markdown, no comments, no extra keys).\n"
    "2) Output must be a JSON array of objects: {\"value_item\": string, \"score\": int}.\n"
    "3) value_item must be a short phrase (2-6 words typically), lowercase preferred unless proper noun.\n"
    "4) Do NOT invent unrelated items. If nothing applies, output exactly: []\n"
    "5) Return at most 8 items (prefer fewer, high-precision).\n"
)

VALUE_USER_TEMPLATE = (
    "FULL_TEXT:\n"
    "<<<\n"
    "{full_text}\n"
    ">>>\n\n"
    "TARGET_SENTENCE:\n"
    "<<<\n"
    "{target_sentence}\n"
    ">>>\n\n"
    "Return ONLY the JSON array."
)


# ===============================
# Parsing helpers (robust JSON extraction)
# ===============================

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def _normalize_raw(raw: Any) -> str:
    if raw is None:
        return ""
    s = str(raw)
    # normalize escaped newlines if they appear literally
    s = s.replace("\\r", "\r").replace("\\n", "\n").replace("\\t", "\t")
    # strip common code fences if any
    s = re.sub(r"^\s*```.*?$", "", s, flags=re.MULTILINE)
    return s.strip()


def _extract_json_array(text: str) -> Optional[str]:
    """
    Try to extract the first top-level JSON array substring from text.
    This lets us recover when a model accidentally adds prefixes.
    """
    s = _normalize_raw(text)
    if not s:
        return None

    # If already valid array, quick return
    if s.startswith("[") and s.endswith("]"):
        return s

    # Find first '[' and attempt to match until corresponding ']'
    start = s.find("[")
    if start < 0:
        return None

    # simple stack scan
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
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]

    return None


def _clip_score(x: Any) -> int:
    try:
        v = int(x)
    except Exception:
        return 0
    return max(0, min(100, v))


def parse_value_items(raw: Any) -> List[Dict[str, Any]]:
    """
    Parse model output into:
      [{"value_item": str, "score": int}, ...]
    If parsing fails -> []
    """
    s = _normalize_raw(raw)
    if not s:
        return []

    arr_txt = _extract_json_array(s)
    if arr_txt is None:
        return []

    try:
        data = json.loads(arr_txt)
    except Exception:
        return []

    if not isinstance(data, list):
        return []

    out: List[Dict[str, Any]] = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        vi = obj.get("value_item", None)
        sc = obj.get("score", None)
        if not isinstance(vi, str):
            continue
        vi2 = vi.strip()
        if not vi2:
            continue
        out.append({"value_item": vi2, "score": _clip_score(sc)})

    # Dedup by value_item keeping max score
    best: Dict[str, int] = {}
    for x in out:
        k = x["value_item"]
        best[k] = max(best.get(k, 0), int(x["score"]))
    merged = [{"value_item": k, "score": v} for k, v in best.items()]
    merged.sort(key=lambda d: int(d["score"]), reverse=True)

    # limit to 8 (per prompt)
    return merged[:8]


# ===============================
# LLM wrapper
# ===============================

class ValueItemExtractor:
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
            system_prompt=VALUE_SYSTEM_PROMPT,
            tensor_parallel_size=tensor_parallel_size,
        )

    def extract_batch(
        self,
        pairs: List[Tuple[str, str]],
        max_full_text_chars: int = 8000,
    ) -> List[Dict[str, Any]]:
        prompts = []
        for full_text, target_sentence in pairs:
            ft = (full_text or "")[:max_full_text_chars]
            ts = (target_sentence or "")
            prompts.append(VALUE_USER_TEMPLATE.format(full_text=ft, target_sentence=ts))

        raw_outputs = self.model(prompts)

        results: List[Dict[str, Any]] = []
        for r in raw_outputs:
            r_str = str(r) if r is not None else ""
            items = parse_value_items(r_str)
            # IMPORTANT: we do NOT keep raw model output anymore
            results.append({"value_items": items})
        return results


# ===============================
# IO helpers
# ===============================

def read_jsonl(path: str, max_rows: int = 0) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows and max_rows > 0 and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ===============================
# CLI
# ===============================

def parse_args():
    ap = argparse.ArgumentParser("Assign value items to expressive sentences in a jsonl file.")
    ap.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Path to expressive jsonl (output of extract_valued_text.py).",
    )
    ap.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="Path to output jsonl (one record per annotation sentence).",
    )

    ap.add_argument("--max_rows", type=int, default=0, help="Read at most N input rows (0 = all).")
    ap.add_argument("--min_expressive_score", type=int, default=0, help="Only use annotations with score >= this.")
    ap.add_argument("--max_items_per_sentence", type=int, default=8, help="Hard cap after parsing (default 8).")

    # LLM configs
    ap.add_argument("--llm_model", type=str, default="Qwen/Qwen3-32B")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="auto")
    ap.add_argument("--model_dir", type=str, default=None)

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_full_text_chars", type=int, default=8000, help="Truncate FULL_TEXT to this many chars.")
    return ap.parse_args()


def main():
    args = parse_args()
    _ensure_dir(args.output_jsonl)

    if args.min_expressive_score < 0 or args.min_expressive_score > 100:
        raise ValueError("--min_expressive_score must be in 0..100")

    # 1) load model
    extractor = ValueItemExtractor(
        model_name=args.llm_model,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
        model_dir=args.model_dir,
    )

    # 2) load jsonl
    rows_in = read_jsonl(args.input_jsonl, max_rows=args.max_rows)

    # 3) expand into per-annotation tasks (filter NONE annotations)
    tasks: List[Dict[str, Any]] = []
    for r in rows_in:
        anns = r.get("annotations", []) or []
        if not anns:
            continue

        idx = r.get("idx", None)
        full_text = r.get("text", "")
        text_type = r.get("text_type", "")

        for a in anns:
            sent = (a.get("sentence", "") or "").strip()
            etype = (a.get("type", "") or "").strip()
            escore = a.get("score", 0)

            if not sent:
                continue
            if int(escore) < int(args.min_expressive_score):
                continue

            tasks.append(
                {
                    "idx": idx,
                    "text_type": text_type,
                    "full_text": full_text,
                    "target_sentence": sent,
                    "expressive_type": etype,
                    "expressive_score": int(escore),
                }
            )

    out_rows: List[Dict[str, Any]] = []

    # 4) batch prompt
    bs = max(1, int(args.batch_size))
    for start in tqdm(range(0, len(tasks), bs), desc="extract_value_items"):
        batch = tasks[start : start + bs]
        pairs = [(x["full_text"], x["target_sentence"]) for x in batch]
        outputs = extractor.extract_batch(pairs, max_full_text_chars=args.max_full_text_chars)

        for x, o in zip(batch, outputs):
            items = o.get("value_items", []) or []
            # hard cap
            items = items[: max(1, int(args.max_items_per_sentence))]

            out_rows.append(
                {
                    **x,
                    "value_items": items,
                }
            )

    # 5) save (JSONL only)
    write_jsonl(args.output_jsonl, out_rows)

    print(f"Input rows read: {len(rows_in)}")
    print(f"Annotation tasks processed: {len(tasks)}")
    print(f"Saved JSONL: {args.output_jsonl}")


if __name__ == "__main__":
    main()
