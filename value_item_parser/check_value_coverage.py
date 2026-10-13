"""Check native mapped value rows against existing catalogs; never rewrite labels."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def check_coverage(input_jsonl, value_items_json, value_defs_json, factors_json, regimes_jsonl):
    known = set(json.loads(Path(value_items_json).read_text(encoding="utf-8")))
    defined = set(json.loads(Path(value_defs_json).read_text(encoding="utf-8")))
    factors = set(json.loads(Path(factors_json).read_text(encoding="utf-8"))["items"])
    with Path(regimes_jsonl).open(encoding="utf-8") as stream:
        regimes = {json.loads(line)["value_item"] for line in stream if line.strip()}
    counts = Counter(rows=0, value_occurrences=0, invalid_rows=0, invalid_value_items=0,
                     unknown_values=0, missing_factors=0, missing_regimes=0,
                     without_continuous_definitions=0)
    observed = set()
    with Path(input_jsonl).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            counts["rows"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                counts["invalid_rows"] += 1
                continue
            if not isinstance(row, dict) or not isinstance(row.get("value_items"), list):
                counts["invalid_rows"] += 1
                continue
            if not isinstance(row.get("target_sentence"), str):
                counts["invalid_rows"] += 1
                continue
            for item in row["value_items"]:
                if not isinstance(item, dict):
                    counts["invalid_value_items"] += 1
                    continue
                value = item.get("value_item")
                if not isinstance(value, str) or not value.strip():
                    counts["invalid_value_items"] += 1
                    continue
                # The connected native output uses exact canonical strings.
                # Do not invent aliasing, drop records, or remap unknown values.
                counts["value_occurrences"] += 1
                observed.add(value)
                counts["unknown_values"] += value not in known
                counts["missing_factors"] += value not in factors
                counts["missing_regimes"] += value not in regimes
                counts["without_continuous_definitions"] += value not in defined
    errors = sum(counts[key] for key in ("invalid_rows", "invalid_value_items", "unknown_values",
                                        "missing_factors", "missing_regimes"))
    passed = bool(counts["rows"] and counts["value_occurrences"] and not errors)
    return {"status": "PASS" if passed else "FAIL", **dict(counts),
            "unique_values": len(observed), "catalog_values": len(known),
            "definition_values": len(defined)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--value_items_json", required=True)
    parser.add_argument("--value_defs_json", required=True)
    parser.add_argument("--factors_json", required=True)
    parser.add_argument("--regimes_jsonl", required=True)
    args = parser.parse_args()
    try:
        result = check_coverage(**vars(args))
    except (OSError, ValueError, KeyError, TypeError):
        # No corpus text or unknown labels are echoed into logs.
        parser.exit(1, "Coverage check failed: unreadable or invalid input/catalog file.\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        parser.exit(1, "Mapped values are empty, malformed, or outside the selected catalogs.\n")


if __name__ == "__main__":
    main()
