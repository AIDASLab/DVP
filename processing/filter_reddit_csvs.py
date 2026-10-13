#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import pandas as pd


USER_LINE_RE = re.compile(r'^\[(?P<user>[^\]\n]{1,128})\]\s', flags=re.UNICODE)
AUTOMOD_NAMES = {"automoderator"}  # case-insensitive check


def split_lines(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.split("\n")


def join_lines(lines: List[str]) -> str:
    return "\n".join(lines).strip()


def is_user_line(line: str) -> Tuple[bool, str]:
    """Return (is_user_line, username) for lines like: [username] comment..."""
    m = USER_LINE_RE.match(line)
    if not m:
        return False, ""
    return True, m.group("user")


def count_comment_lines(lines: List[str]) -> int:
    """Count lines that look like '[user] ...'."""
    c = 0
    for ln in lines:
        is_user, _ = is_user_line(ln)
        if is_user:
            c += 1
    return c


def remove_automod(lines: List[str]) -> List[str]:
    """Remove any lines authored by AutoModerator (case-insensitive)."""
    kept = []
    for ln in lines:
        is_user, user = is_user_line(ln)
        if is_user and user.strip().lower() in AUTOMOD_NAMES:
            continue
        kept.append(ln)
    return kept


def remap_users_inline(lines: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Remap per-row unique users to [user_1], [user_2], ...
    Only remap on lines that start with [user] .
    Return new lines and the mapping (original_user -> tag).
    """
    mapping: Dict[str, str] = {}
    next_id = 1
    out = []

    for ln in lines:
        is_user, user = is_user_line(ln)
        if not is_user:
            out.append(ln)
            continue

        key = user.strip()
        # Skip AutoModerator here too (should have been removed already, but be safe)
        if key.lower() in AUTOMOD_NAMES:
            # drop the line completely
            continue

        if key not in mapping:
            mapping[key] = f"user_{next_id}"
            next_id += 1

        # Replace only the leading tag
        out.append(USER_LINE_RE.sub(f"[{mapping[key]}] ", ln, count=1))

    return out, mapping


def process_text_cell(text: str, min_comments: int) -> Tuple[str, int]:
    """
    Process one 'text' cell:
      - remove automod lines
      - remap users
      - count comments after removal
      - return (new_text, num_comments_kept)
    """
    lines = split_lines(text)
    # Remove AutoModerator lines
    lines = remove_automod(lines)
    # Remap users
    lines, _ = remap_users_inline(lines)
    # Count kept comments
    kept = count_comment_lines(lines)
    # Rebuild text
    new_text = join_lines(lines)
    return new_text, kept


def process_dataframe(df: pd.DataFrame, min_comments: int) -> pd.DataFrame:
    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column.")
    if "num_comments" not in df.columns:
        # There is no existing count to preserve in num_comments_original.
        has_num_comments = False
    else:
        has_num_comments = True

    processed_texts = []
    kept_counts = []

    # Iterate rows; vectorizing this kind of parsing offers little benefit
    for _, row in df.iterrows():
        t = row.get("text", "")
        new_text, kept = process_text_cell(t, min_comments=min_comments)
        processed_texts.append(new_text)
        kept_counts.append(kept)

    df = df.copy()
    df["text"] = processed_texts
    df["num_comments_kept"] = kept_counts

    # Filter rows with too few comments AFTER processing
    df = df[df["num_comments_kept"] >= min_comments].reset_index(drop=True)

    # Preserve an existing count separately, then report the retained count.
    if has_num_comments:
        df["num_comments_original"] = df["num_comments"]
    df["num_comments"] = df["num_comments_kept"]

    return df


def resolve_outputs(input_path: Path, output_root: Path, base_inputs: List[Path]) -> Path:
    """
    Mirror the input directory layout under output_root if possible.
    If input was a file, put it under output_root / relative_parent / filename.
    """
    # Find a base ancestor among provided input roots to compute a relative path
    chosen_base = None
    for base in base_inputs:
        try:
            rel = input_path.relative_to(base)
            chosen_base = base
            break
        except ValueError:
            continue

    if chosen_base is not None:
        rel = input_path.relative_to(chosen_base)
        return (output_root / rel).with_suffix(".csv")

    # Fallback: just place in output_root with the file name
    return (output_root / input_path.name).with_suffix(".csv")


def collect_csvs(inputs: List[Path]) -> List[Path]:
    """
    Collect only CSV files that end with '_doc.csv' from the given inputs.
    Recursively scans directories.
    """
    files: List[Path] = []
    for p in inputs:
        if p.is_file() and p.name.endswith("_doc.csv"):
            files.append(p)
        elif p.is_dir():
            for f in p.rglob("*_doc.csv"):
                if f.is_file():
                    files.append(f)
    return sorted(files)

def main():
    ap = argparse.ArgumentParser(
        description="Filter Reddit doc CSVs: drop short threads, remove AutoModerator, and anonymize users."
    )
    ap.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more CSV files or directories to process (recursively).",
    )
    ap.add_argument(
        "--output_dir",
        required=True,
        help="Directory where filtered CSVs will be written, mirroring input structure.",
    )
    ap.add_argument(
        "--min_comments",
        type=int,
        default=5,
        help="Minimum number of comments required to keep a row (default: 5).",
    )
    args = ap.parse_args()

    input_paths = [Path(x).resolve() for x in args.input]
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    csvs = collect_csvs(input_paths)
    if not csvs:
        print("[WARN] No CSV files found in the provided paths.", file=sys.stderr)
        sys.exit(0)

    for in_csv in csvs:
        try:
            df = pd.read_csv(in_csv)
        except Exception as e:
            print(f"[SKIP] Could not read {in_csv}: {e}", file=sys.stderr)
            continue

        try:
            out_df = process_dataframe(df, min_comments=args.min_comments)
        except Exception as e:
            print(f"[ERROR] Failed to process {in_csv}: {e}", file=sys.stderr)
            continue

        # Ensure parent dirs exist
        out_path = resolve_outputs(in_csv, output_root, input_paths)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"[OK] {in_csv} -> {out_path} (rows kept={len(out_df)})")


if __name__ == "__main__":
    main()
