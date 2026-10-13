#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_valued_text.py

Filter + extract "expressive" (normative / preference-bearing) sentences from a CSV,
AND assign an expressibility score (0~100) per extracted sentence.

Expressive = decisions / opinions / values-norms / preferences-tradeoffs.

Output (JSONL or JSON)
----------------------
Each row -> one record:
{
  "idx": <int>,
  "text_col": "text",
  "text_type": "plain|comment|dialogue",
  "text": <original text>,
  "annotations": [
     {"sentence": "...", "score": 0-100, "type": "decision|opinion|value_norm|preference_tradeoff|mixed"},
     ...
  ],
  "raw_model_output": <str>
}

MODEL OUTPUT FORMAT (STRICT)
----------------------------
The model must output ONLY records in this exact format, one per extracted sentence:

SENTENCE: <sentence text>
TYPE: <decision|opinion|value_norm|preference_tradeoff|mixed>
SCORE: <0..100>

Records are separated by a blank line.

If there is NONE, output exactly: NONE
"""

import os
import re
import json
import argparse
from typing import List, Dict, Any, Optional

import pandas as pd
from tqdm import tqdm

# -------------------------------
# Inference backend
# -------------------------------
# The inference backend is imported when the extractor is constructed.


# ===============================
# Prompting
# ===============================

SYSTEM_PROMPT = (
    "You are a strict multilingual annotator for VALUE-EXPRESSIVE sentences.\n"
    "Your job is HIGH PRECISION: when unsure, DO NOT extract.\n\n"

    "TASK\n"
    "Extract ONLY sentences that explicitly express at least one of the following:\n"
    "  (A) decision/intent, (B) opinion/belief, (C) value/norm/ought, (D) preference/trade-off.\n\n"

    "IMPORTANT: Many texts contain emotions, dialogue, actions, or events. Those are NOT automatically expressive.\n"
    "A sentence must contain an explicit stance cue or normative cue to be extracted.\n\n"

    "TYPES (choose exactly one per extracted sentence)\n"
    "1) decision: explicit decision / intention / plan / refusal.\n"
    "   - MUST contain a decision cue (e.g., decided/chose/will/won't/plan/try/need to/must for self)\n"
    "     or Korean cues like: \"겠다\", \"할래\", \"할 거야\", \"하기로\", \"결정\", \"선택\", \"~해야겠\", \"~할 필요\".\n\n"
    "2) opinion: explicit belief/judgment/view.\n"
    "   - MUST contain an opinion cue (e.g., I think/I believe/in my view/seems)\n"
    "     or Korean cues like: \"~라고 생각\", \"~라고 봐\", \"~인 것 같\", \"내 의견\", \"~라 믿\".\n\n"
    "3) value_norm: explicit norm/should/ought/important/right/wrong.\n"
    "   - MUST contain a normative cue such as: should/ought/must (general), important, wrong/right\n"
    "     or Korean cues like: \"~해야 한다\", \"~해야만\", \"~돼야\", \"중요\", \"옳/그르\", \"바람직\", \"필요하다(일반 규범)\".\n\n"
    "4) preference_tradeoff: explicit preference or trade-off.\n"
    "   - MUST contain preference/trade-off cue such as: prefer/rather/priority/over/even if/at the cost of\n"
    "     or Korean cues like: \"~보다 ~가 낫\", \"차라리\", \"선호\", \"우선\", \"~하더라도\", \"감수\".\n\n"
    "5) mixed: ONLY if a SINGLE sentence clearly contains TWO OR MORE of the above types.\n"
    "   - Do NOT use mixed for conditional predictions or general commentary.\n\n"

    "HARD EXCLUSION RULES (DO NOT extract these even if they mention charged topics)\n"
    "- Pure facts or reports: news-like event descriptions, who-did-what, dates, numbers, approvals, announcements.\n"
    "- Headline/title fragments, slogans, hashtags, or quote-only titles.\n"
    "- Narrative/scene/action-only sentences: physical actions, descriptions (\"I looked\", \"he kissed\", \"she walked\").\n"
    "- Questions that merely ask information or invite action, unless they explicitly contain a norm/preference "
    "  (e.g., \"Should we...?\", \"Would you rather...?\" with explicit cues).\n"
    "- Imperatives/requests (\"Don't run away\") unless they explicitly express a norm/value (e.g., \"You must...\").\n"
    "- Procedural instructions without stance.\n"
    "- Corrupted/garbled text or model error messages.\n\n"

    "SCORING (0-100)\n"
    "Score expresses how EXPLICIT and STRONG the stance is.\n"
    "- 0-20: should NOT be extracted (if extracted by mistake).\n"
    "- 21-50: weak stance; mild/hedged, vague.\n"
    "- 51-80: clear stance with explicit cue.\n"
    "- 81-100: very strong/explicit commitment, normative insistence, or clear trade-off.\n"
    "IMPORTANT: If a sentence is mostly factual, do NOT extract it instead of giving a low score.\n\n"

    "OUTPUT FORMAT (STRICT)\n"
    "- Output ONLY extracted items in the format below.\n"
    "- One record per extracted sentence, exactly 3 lines:\n"
    "  SENTENCE: <sentence text>\n"
    "  TYPE: <decision|opinion|value_norm|preference_tradeoff|mixed>\n"
    "  SCORE: <integer 0..100>\n"
    "- Separate records with a blank line.\n"
    "- If there is NONE, output exactly: NONE\n\n"

    "FINAL CHECK BEFORE OUTPUT\n"
    "For each extracted sentence, confirm internally:\n"
    "  (1) It contains an explicit cue for the chosen type.\n"
    "  (2) It is NOT a pure report/action/headline.\n"
    "If either fails, DO NOT output that sentence.\n"
)

USER_TEMPLATE = (
    "TEXT_TYPE: {text_type}\n"
    "GUIDELINES FOR TEXT_TYPE:\n"
    "- plain: standard prose\n"
    "- comment: may be informal/short/emotional; still extract only stance sentences with explicit cues\n"
    "- dialogue: multi-speaker; extract stance sentences from any speaker; keep only the sentence text\n"
    "- Always split into sentences; ignore title fragments and broken strings.\n\n"
    "TEXT:\n"
    "<<<\n"
    "{text}\n"
    ">>>"
)

VALID_TYPES = {"decision", "opinion", "value_norm", "preference_tradeoff", "mixed"}


# ===============================
# Parsing helpers
# ===============================

_SENT_RE = re.compile(r"^\s*SENTENCE\s*:\s*(.+?)\s*$", re.IGNORECASE)
_TYPE_RE = re.compile(r"^\s*TYPE\s*:\s*([A-Za-z_]+)\s*$", re.IGNORECASE)
_SCORE_RE = re.compile(r"^\s*SCORE\s*:\s*(\d{1,3})\s*$", re.IGNORECASE)

# Optional single-line fallback (many models like to do this):
# SENTENCE: ... | TYPE: ... | SCORE: ...
_ONE_LINE_RE = re.compile(
    r"(?i)SENTENCE\s*:\s*(?P<sent>.+?)\s*(?:\||;|,)\s*TYPE\s*:\s*(?P<typ>[A-Za-z_]+)\s*(?:\||;|,)\s*SCORE\s*:\s*(?P<score>\d{1,3})\s*$"
)


def _normalize_raw(raw: str) -> str:
    """Normalize common escaped sequences and strip code-fences."""
    if not isinstance(raw, str):
        return ""
    s = raw

    # If the model returned literal escape sequences, normalize.
    # This helps when upstream serializes strings or models output "\\n", "\\t".
    s = s.replace("\\r", "\r").replace("\\n", "\n").replace("\\t", "\t")

    # Remove common code fences lines
    s = re.sub(r"^\s*```.*?$", "", s, flags=re.MULTILINE)
    s = s.strip()

    return s


def _coerce_type(typ: str) -> str:
    t = (typ or "").strip().lower()

    # Some models may output close variants; map them.
    alias = {
        "value": "value_norm",
        "norm": "value_norm",
        "values_norms": "value_norm",
        "values": "value_norm",
        "preference": "preference_tradeoff",
        "tradeoff": "preference_tradeoff",
        "preference-tradeoff": "preference_tradeoff",
        "preference_trade-off": "preference_tradeoff",
        "decision_choice": "decision",
        "belief": "opinion",
    }
    t = alias.get(t, t)

    if t not in VALID_TYPES:
        return "mixed"
    return t


def _clip_score(score: int) -> int:
    try:
        score = int(score)
    except Exception:
        return 0
    return max(0, min(100, score))


def _parse_records(raw: str) -> List[Dict[str, Any]]:
    """
    Parse raw model output into list of:
      {"sentence": ..., "score": int, "type": ...}

    Expected record format (3 lines):
      SENTENCE: ...
      TYPE: ...
      SCORE: ...
    Records separated by blank line.

    Also supports one-line fallback:
      SENTENCE: ... | TYPE: ... | SCORE: ...
    """
    s = _normalize_raw(raw)
    if not s or s.upper() == "NONE":
        return []

    anns: List[Dict[str, Any]] = []

    # 1) Try to parse one-line records first (robust for models that won't do multi-line)
    for ln in s.splitlines():
        ln = ln.strip()
        if not ln:
            continue

        # remove bullet/numbering prefixes if present
        ln2 = re.sub(r"^\s*[-*•]\s*", "", ln)
        ln2 = re.sub(r"^\s*\d+[\).\]]\s*", "", ln2)

        m = _ONE_LINE_RE.match(ln2)
        if m:
            sent = m.group("sent").strip().strip('"').strip("'")
            typ = _coerce_type(m.group("typ"))
            score = _clip_score(int(m.group("score")))
            if sent:
                anns.append({"sentence": sent, "score": score, "type": typ})

    # Retain parsed one-line entries and also attempt block parsing.
    # (block parsing may add more).
    # 2) Block parsing: split by blank lines into record chunks.
    blocks = re.split(r"\n\s*\n+", s)
    for blk in blocks:
        blk = blk.strip()
        if not blk:
            continue
        if blk.upper() == "NONE":
            continue

        # Skip single-line records already included above.
        if _ONE_LINE_RE.match(blk.replace("\n", " ").strip()):
            # already covered by one-line parsing
            continue

        lines = [re.sub(r"^\s*[-*•]\s*", "", x).strip() for x in blk.splitlines() if x.strip()]
        if len(lines) < 3:
            # Sometimes models omit blank lines; try to recover by scanning within block
            # by picking first occurrences of SENTENCE/TYPE/SCORE anywhere.
            sent = typ = None
            score = None
            for x in lines:
                m1 = _SENT_RE.match(x)
                if m1 and sent is None:
                    sent = m1.group(1).strip()
                    continue
                m2 = _TYPE_RE.match(x)
                if m2 and typ is None:
                    typ = m2.group(1).strip()
                    continue
                m3 = _SCORE_RE.match(x)
                if m3 and score is None:
                    score = m3.group(1).strip()
                    continue

            if sent and typ and score is not None:
                sent = sent.strip().strip('"').strip("'")
                typ = _coerce_type(typ)
                score_i = _clip_score(int(score))
                if sent:
                    anns.append({"sentence": sent, "score": score_i, "type": typ})
            continue

        # Normal case: read line-by-line and extract fields
        sent = typ = None
        score = None

        for x in lines:
            m1 = _SENT_RE.match(x)
            if m1:
                sent = m1.group(1).strip()
                continue
            m2 = _TYPE_RE.match(x)
            if m2:
                typ = m2.group(1).strip()
                continue
            m3 = _SCORE_RE.match(x)
            if m3:
                score = m3.group(1).strip()
                continue

        if sent and typ and score is not None:
            sent = sent.strip().strip('"').strip("'")
            typ = _coerce_type(typ)
            score_i = _clip_score(int(score))
            if sent:
                anns.append({"sentence": sent, "score": score_i, "type": typ})

    # Deduplicate (some models might produce duplicates across parsing paths)
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for a in anns:
        key = (a.get("sentence", ""), a.get("type", ""), int(a.get("score", 0)))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(a)

    return uniq


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


# ===============================
# LLM extractor (persistent)
# ===============================

class ExpressiveSentenceExtractor:
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
            system_prompt=SYSTEM_PROMPT,
            tensor_parallel_size=tensor_parallel_size,
        )

    def extract(self, texts: List[str], text_type: str) -> List[Dict[str, Any]]:
        prompts = [USER_TEMPLATE.format(text_type=text_type, text=(t or "")[:8000]) for t in texts]
        raw = self.model(prompts)

        results: List[Dict[str, Any]] = []
        for r in raw:
            r_str = str(r) if r is not None else ""
            results.append(
                {
                    "raw_model_output": r_str,
                    "annotations": _parse_records(r_str),
                }
            )
        return results


# ===============================
# CLI
# ===============================

def parse_args():
    ap = argparse.ArgumentParser("Extract expressive sentences + score (0~100) from CSV text.")
    ap.add_argument("--input_csv", type=str, required=True, help="Path to a single CSV file.")
    ap.add_argument("--output_path", type=str, required=True, help="Path to output .jsonl (recommended) or .json")

    ap.add_argument("--text_col", type=str, default="text", help="Which column in CSV to use as text (default: text).")
    ap.add_argument("--max_rows", type=int, default=0, help="Max rows to process from top (0 = all).")

    ap.add_argument(
        "--text_type",
        type=str,
        default="plain",
        choices=["plain", "comment", "dialogue"],
        help="Prompt framing by text type.",
    )

    # LLM configs
    ap.add_argument("--llm_model", type=str, default="Qwen/Qwen3-32B", help="Model name for LLMModel.")
    ap.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size.")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    ap.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling.")
    ap.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens for extraction output.")
    ap.add_argument("--device", type=str, default="cuda", help="Device for model (default: cuda).")
    ap.add_argument("--dtype", type=str, default="auto", help="dtype for model (default: auto).")
    ap.add_argument("--model_dir", type=str, default=None, help="Optional local model dir.")

    # batching / robustness
    ap.add_argument("--batch_size", type=int, default=16, help="LLM batch size.")
    ap.add_argument("--skip_empty", action="store_true", help="Skip rows with empty/non-string text.")

    # output format
    ap.add_argument(
        "--output_format",
        type=str,
        default="jsonl",
        choices=["jsonl", "json"],
        help="Output format (jsonl recommended).",
    )

    # optional: keep only sentences above threshold
    ap.add_argument(
        "--min_score",
        type=int,
        default=0,
        help="Keep only extracted sentences with score >= min_score (0..100). Default 0 keeps all.",
    )

    return ap.parse_args()


def main():
    args = parse_args()
    _ensure_dir(args.output_path)

    if args.min_score < 0 or args.min_score > 100:
        raise ValueError("--min_score must be in 0..100")

    # 1) Load model
    extractor = ExpressiveSentenceExtractor(
        model_name=args.llm_model,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
        model_dir=args.model_dir,
    )

    # 2) Load CSV + truncate
    df = pd.read_csv(args.input_csv)
    if args.text_col not in df.columns:
        raise ValueError(f"--text_col '{args.text_col}' not found in columns: {list(df.columns)}")

    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    texts = df[args.text_col].tolist()

    # Output
    as_jsonl = (args.output_format == "jsonl") or args.output_path.lower().endswith(".jsonl")
    records_all: List[Dict[str, Any]] = []

    f = open(args.output_path, "w", encoding="utf-8") if as_jsonl else None

    try:
        batch_size = max(1, args.batch_size)
        for start in tqdm(range(0, len(texts), batch_size), desc="extract_expressive_scored"):
            chunk = texts[start:start + batch_size]
            idxs = list(range(start, min(start + batch_size, len(texts))))

            # Optional skip empty
            if args.skip_empty:
                filtered = []
                filtered_idxs = []
                for i, t in zip(idxs, chunk):
                    if isinstance(t, str) and t.strip():
                        filtered.append(t)
                        filtered_idxs.append(i)
                chunk = filtered
                idxs = filtered_idxs

            if not chunk:
                continue

            try:
                outputs = extractor.extract(chunk, text_type=args.text_type)
            except Exception as e:
                # skip this single row
                i = idxs[0]
                t = chunk[0]

                rec = {
                    "idx": int(i),
                    "text_col": args.text_col,
                    "text_type": args.text_type,
                    "text": t,
                    "annotations": [],
                    "raw_model_output": f"__ERROR__: {type(e).__name__}: {e}",
                }

                if as_jsonl:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                else:
                    records_all.append(rec)

                continue

            for i, t, out in zip(idxs, chunk, outputs):
                anns = out["annotations"]
                if args.min_score > 0:
                    anns = [a for a in anns if int(a.get("score", 0)) >= args.min_score]

                rec = {
                    "idx": int(i),
                    "text_col": args.text_col,
                    "text_type": args.text_type,
                    "text": t,
                    "annotations": anns,
                    "raw_model_output": out["raw_model_output"],
                }

                if as_jsonl:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                else:
                    records_all.append(rec)

        if not as_jsonl:
            with open(args.output_path, "w", encoding="utf-8") as wf:
                json.dump(records_all, wf, ensure_ascii=False, indent=2)

    finally:
        if f is not None:
            f.close()

    print(f"Saved: {args.output_path}")
    print(f"Rows processed: {len(df)}")
    print(f"Output format: {'jsonl' if as_jsonl else 'json'}")
    if args.min_score > 0:
        print(f"Filtered annotations with min_score >= {args.min_score}")


if __name__ == "__main__":
    main()
